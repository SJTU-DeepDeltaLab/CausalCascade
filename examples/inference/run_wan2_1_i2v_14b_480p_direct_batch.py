import argparse
import inspect
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

from common import (
    append_manifest,
    base_metadata,
    case_frames_dir,
    case_metadata,
    case_video_path,
    collect_cases,
    default_project_root,
    export_frames_from_video,
    parse_gpu_list,
    read_prompt,
    record_failure,
    choose_seed,
    setup_logging,
    utc_now,
)


MODEL_NAME = "Wan2.1-I2V-14B-480P"


def str_to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    return value.lower() in ("1", "true", "yes", "y", "on")


def parse_args():
    root = default_project_root()
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_dir", type=Path, default=root / "data" / "inputs" / "L1")
    p.add_argument("--model_dir", type=Path, default=root / MODEL_NAME)
    p.add_argument("--code_dir", type=Path, default=root / "ModelCode" / "Wan2.1")
    p.add_argument("--output_dir", type=Path, default=root / MODEL_NAME / "output")

    p.add_argument("--task", type=str, default="i2v-14B")
    p.add_argument("--size", type=str, default="832*480")
    p.add_argument("--frame_num", type=int, default=81, help="Wan generate.py says frame_num should be 4n+1; I2V default is 81.")
    p.add_argument("--fps", type=int, default=16, help="Wan2.1 I2V native/recommended save fps is 16.")
    p.add_argument("--sample_steps", type=int, default=None, help="If omitted, use cfg.sample_steps if present, otherwise 40 for I2V.")
    p.add_argument("--sample_solver", type=str, default="unipc")
    p.add_argument("--sample_shift", type=float, default=None, help="If omitted, use cfg.sample_shift if present, otherwise 3.0 for 480P I2V.")
    p.add_argument("--sample_guide_scale", type=float, default=None, help="If omitted, use cfg.sample_guide_scale if present, otherwise 5.0.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seed_strategy",
        type=str,
        default="case_hash",
        choices=("case_hash", "sequential", "same"),
        help=(
            "case_hash: deterministic seed from base seed and case_id; recommended for benchmarks. "
            "sequential: seed = base_seed + case_index - 1. "
            "same: all cases use the same base seed."
        ),
    )
    p.add_argument("--random_seed", action="store_true")

    # Official multi-GPU / memory options.
    p.add_argument("--t5_fsdp", action="store_true")
    p.add_argument("--dit_fsdp", action="store_true")
    p.add_argument("--ulysses_size", type=int, default=1)
    p.add_argument("--ring_size", type=int, default=1)
    p.add_argument("--t5_cpu", action="store_true")
    p.add_argument("--convert_model_dtype", action="store_true")
    p.add_argument("--offload_model", type=str, default=None, choices=("true", "false"))

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only_series", nargs="*", default=None)
    p.add_argument("--save_frames", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def validate_args(args, world_size: int):
    if args.frame_num <= 0:
        raise ValueError("--frame_num must be positive")
    if (args.frame_num - 1) % 4 != 0:
        logging.warning("Wan official generate.py says frame_num should be 4n+1; got %s", args.frame_num)
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if world_size > 1:
        if args.ulysses_size > 1 and args.ulysses_size != world_size:
            raise ValueError("--ulysses_size must equal WORLD_SIZE when using Ulysses")
        if args.dit_fsdp is False:
            logging.warning("WORLD_SIZE > 1 but --dit_fsdp is not set. Official multi-GPU practice usually uses --dit_fsdp.")
        if args.t5_fsdp is False and not args.t5_cpu:
            logging.warning("WORLD_SIZE > 1 but neither --t5_fsdp nor --t5_cpu is set.")
    else:
        if args.t5_fsdp or args.dit_fsdp or args.ulysses_size > 1 or args.ring_size > 1:
            raise ValueError("FSDP/Ulysses/Ring require torchrun with WORLD_SIZE > 1")


def main():
    setup_logging()
    args = parse_args()

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if rank != 0:
        logging.getLogger().setLevel(logging.ERROR)

    validate_args(args, world_size)

    if not args.code_dir.exists():
        raise FileNotFoundError(f"code_dir does not exist: {args.code_dir}")
    if not args.model_dir.exists():
        raise FileNotFoundError(f"model_dir does not exist: {args.model_dir}")

    sys.path.insert(0, str(args.code_dir.resolve()))

    import torch
    import torch.distributed as dist
    from PIL import Image

    import wan
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import cache_video

    cases = collect_cases(args.dataset_dir, args.only_series, args.limit)
    manifest_path = args.output_dir / "manifest.json"

    if rank == 0:
        logging.info("Found %d cases", len(cases))
        logging.info(
            "Config: task=%s size=%s frame_num=%s fps=%s seed=%s seed_strategy=%s",
            args.task,
            args.size,
            args.frame_num,
            args.fps,
            args.seed,
            args.seed_strategy,
        )
        if args.dry_run:
            rng = random.Random(args.seed)
            for idx, case in enumerate(cases, start=1):
                seed = choose_seed(args, case, idx, rng)
                logging.info("%s seed=%s -> %s", case.case_id, seed, case_video_path(args.output_dir, case))
            return
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # In dry_run under torchrun, non-rank0 should also return.
    if args.dry_run:
        return

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        torch.cuda.set_device(0 if torch.cuda.is_available() else "cpu")

    if args.task not in WAN_CONFIGS:
        raise ValueError(f"Unknown Wan task {args.task}. Available: {list(WAN_CONFIGS.keys())}")

    if args.size not in MAX_AREA_CONFIGS:
        raise ValueError(f"Unknown size {args.size}. Available sizes include: {list(MAX_AREA_CONFIGS.keys())}")

    cfg = WAN_CONFIGS[args.task]

    # Official Wan2.1 generate.py initializes xFuser sequence-parallel groups
    # before constructing WanI2V/WanT2V when Ulysses/Ring is enabled.
    # Without this, WanI2V.__init__ will fail at get_sequence_parallel_world_size().
    if world_size > 1 and (args.ulysses_size > 1 or args.ring_size > 1):
        if args.ulysses_size * args.ring_size != world_size:
            raise ValueError(
                f"--ulysses_size * --ring_size must equal WORLD_SIZE. "
                f"Got {args.ulysses_size} * {args.ring_size} != {world_size}"
            )

        if args.ulysses_size > 1 and hasattr(cfg, "num_heads"):
            if cfg.num_heads % args.ulysses_size != 0:
                raise ValueError(
                    f"cfg.num_heads={cfg.num_heads} must be divisible by "
                    f"--ulysses_size={args.ulysses_size}"
                )

        from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel

        init_distributed_environment(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
        )
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

        if rank == 0:
            logging.info(
                "Initialized xFuser model parallel: sequence_parallel_degree=%s ring_degree=%s ulysses_degree=%s",
                dist.get_world_size(),
                args.ring_size,
                args.ulysses_size,
            )

    # Match official defaults for I2V when config fields are absent.
    sample_steps = args.sample_steps
    if sample_steps is None:
        sample_steps = getattr(cfg, "sample_steps", 40)

    sample_shift = args.sample_shift
    if sample_shift is None:
        sample_shift = getattr(cfg, "sample_shift", 3.0 if args.size in ("832*480", "480*832") else 5.0)

    sample_guide_scale = args.sample_guide_scale
    if sample_guide_scale is None:
        sample_guide_scale = getattr(cfg, "sample_guide_scale", 5.0)

    cfg_sample_fps = getattr(cfg, "sample_fps", args.fps)

    offload_model = str_to_bool(args.offload_model)
    if offload_model is None:
        # Single-GPU inference offloads the model; distributed inference keeps it on GPU.
        offload_model = False if world_size > 1 else True

    use_sp = args.ulysses_size > 1 or args.ring_size > 1

    if rank == 0:
        logging.info("Creating WanI2V once per rank for batch inference")
        logging.info("Prompt extension: disabled")
        logging.info(
            "Sampling: steps=%s shift=%s guide_scale=%s solver=%s offload_model=%s cfg_sample_fps=%s save_fps=%s",
            sample_steps,
            sample_shift,
            sample_guide_scale,
            args.sample_solver,
            offload_model,
            cfg_sample_fps,
            args.fps,
        )
        logging.info(
            "Distributed: world_size=%s local_rank=%s dit_fsdp=%s t5_fsdp=%s ulysses_size=%s ring_size=%s use_sp=%s",
            world_size,
            local_rank,
            args.dit_fsdp,
            args.t5_fsdp,
            args.ulysses_size,
            args.ring_size,
            use_sp,
        )

    # Pass the sequence-parallel flag accepted by the installed WanI2V constructor.
    init_sig = inspect.signature(wan.WanI2V.__init__)
    init_params = init_sig.parameters

    candidate_kwargs = {
        "config": cfg,
        "checkpoint_dir": str(args.model_dir.resolve()),
        "device_id": local_rank,
        "rank": rank,
        "t5_fsdp": args.t5_fsdp,
        "dit_fsdp": args.dit_fsdp,
        "use_usp": use_sp,
        "use_sp": use_sp,
        "t5_cpu": args.t5_cpu,
        "convert_model_dtype": args.convert_model_dtype,
    }
    ctor_kwargs = {k: v for k, v in candidate_kwargs.items() if k in init_params}

    if rank == 0:
        logging.info("WanI2V.__init__ signature: %s", init_sig)
        logging.info("WanI2V ctor kwargs actually used: %s", sorted(ctor_kwargs.keys()))

    model = wan.WanI2V(**ctor_kwargs)

    rng = random.Random(args.seed)

    run_record = {
        **base_metadata(
            MODEL_NAME,
            args.model_dir,
            args.dataset_dir,
            args.output_dir,
            args,
            "direct official WanI2V batch runner with one model load per rank",
        ),
        "started_at_utc": utc_now(),
        "seed_policy": {
            "base_seed": args.seed,
            "seed_strategy": args.seed_strategy,
            "random_seed": args.random_seed,
        },
        "generation_config": {
            "task": args.task,
            "size": args.size,
            "frame_num": args.frame_num,
            "fps": args.fps,
            "cfg_sample_fps": cfg_sample_fps,
            "sample_steps": sample_steps,
            "sample_shift": sample_shift,
            "sample_guide_scale": sample_guide_scale,
            "sample_solver": args.sample_solver,
            "offload_model": offload_model,
        },
        "multi_gpu": {
            "world_size": world_size,
            "dit_fsdp": args.dit_fsdp,
            "t5_fsdp": args.t5_fsdp,
            "ulysses_size": args.ulysses_size,
            "ring_size": args.ring_size,
            "use_sp": use_sp,
        },
        "cases": [],
    }

    batch_start = time.time()

    for idx, case in enumerate(cases, start=1):
        video_path = case_video_path(args.output_dir, case)
        frames_dir = case_frames_dir(args.output_dir, case) if args.save_frames else None
        seed = choose_seed(args, case, idx, rng)

        # All ranks must agree on skipping; rank0 decides and broadcasts.
        if rank == 0:
            skip = video_path.exists() and not args.overwrite
        else:
            skip = None

        if world_size > 1:
            obj = [skip]
            dist.broadcast_object_list(obj, src=0)
            skip = bool(obj[0])

        if skip:
            if rank == 0:
                prompt_for_meta = read_prompt(case)
                meta = case_metadata(case, prompt_for_meta, video_path, frames_dir)
                meta.update(
                    {
                        "index": idx,
                        "status": "skipped_existing",
                        "seed": seed,
                        "base_seed": args.seed,
                        "seed_strategy": "random_seed" if args.random_seed else args.seed_strategy,
                        "completed_at_utc": utc_now(),
                    }
                )
                run_record["cases"].append(meta)
                logging.info("[%d/%d] skip existing %s", idx, len(cases), case.case_id)
            continue

        case_start = time.time()

        try:
            prompt = read_prompt(case)
            image = Image.open(case.image_path).convert("RGB")

            if rank == 0:
                video_path.parent.mkdir(parents=True, exist_ok=True)
                logging.info(
                    "[%d/%d] generating %s seed=%s",
                    idx,
                    len(cases),
                    case.case_id,
                    seed,
                )

            video = model.generate(
                prompt,
                image,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=sample_steps,
                guide_scale=sample_guide_scale,
                seed=seed,
                offload_model=offload_model,
            )

            if rank == 0:
                cache_video(
                    tensor=video[None],
                    save_file=str(video_path),
                    fps=args.fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )

                meta = case_metadata(case, prompt, video_path, frames_dir)
                meta.update(
                    {
                        "index": idx,
                        "status": "ok",
                        "seed": seed,
                        "base_seed": args.seed,
                        "seed_strategy": "random_seed" if args.random_seed else args.seed_strategy,
                        "task": args.task,
                        "size": args.size,
                        "frame_num": args.frame_num,
                        "fps": args.fps,
                        "cfg_sample_fps": cfg_sample_fps,
                        "sample_steps": sample_steps,
                        "sample_shift": sample_shift,
                        "sample_guide_scale": sample_guide_scale,
                        "sample_solver": args.sample_solver,
                        "elapsed_seconds": time.time() - case_start,
                        "completed_at_utc": utc_now(),
                    }
                )

                if frames_dir:
                    meta["exported_frame_count"] = export_frames_from_video(video_path, frames_dir, args.fps)

                run_record["cases"].append(meta)

                logging.info(
                    "[%d/%d] completed %s seed=%s elapsed=%.1fs -> %s",
                    idx,
                    len(cases),
                    case.case_id,
                    seed,
                    time.time() - case_start,
                    video_path,
                )

            del video

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if world_size > 1:
                dist.barrier()

        except Exception as exc:
            if rank == 0:
                logging.exception("[%d/%d] failed %s seed=%s", idx, len(cases), case.case_id, seed)
                prompt_for_meta = read_prompt(case) if case.prompt_path.exists() else ""
                meta = case_metadata(case, prompt_for_meta, video_path, frames_dir)
                meta.update(
                    {
                        "index": idx,
                        "status": "failed",
                        "seed": seed,
                        "base_seed": args.seed,
                        "seed_strategy": "random_seed" if args.random_seed else args.seed_strategy,
                        "elapsed_seconds": time.time() - case_start,
                        **record_failure(exc),
                        "completed_at_utc": utc_now(),
                    }
                )
                run_record["cases"].append(meta)

            # In distributed mode, abort the whole torchrun job on failure to avoid rank hangs.
            if world_size > 1:
                raise

    if rank == 0:
        run_record["completed_at_utc"] = utc_now()
        run_record["elapsed_seconds"] = time.time() - batch_start
        append_manifest(manifest_path, run_record)
        logging.info("Batch complete: %s elapsed=%.1fs", args.output_dir, run_record["elapsed_seconds"])

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
