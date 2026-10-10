import argparse
import logging
import os
import random
import sys
from pathlib import Path

from common import (
    append_manifest,
    base_metadata,
    case_frames_dir,
    case_metadata,
    case_video_path,
    collect_cases,
    default_project_root,
    export_frames_from_video,
    read_prompt,
    record_failure,
    choose_seed,
    setup_logging,
    utc_now,
)


MODEL_NAME = "Wan2.2-TI2V-5B"
torch = None


def parse_args():
    root = default_project_root()
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_dir", type=Path, default=root / "data" / "inputs" / "L1")
    p.add_argument("--model_dir", type=Path, default=root / MODEL_NAME)
    p.add_argument(
        "--code_dir",
        type=Path,
        default=root / "ModelCode" / "Wan2.2",
        help="Official Wan2.2 repository. Falls back to model_dir if it contains the wan package.",
    )
    p.add_argument("--output_dir", type=Path, default=root / MODEL_NAME / "output")

    p.add_argument("--frame_num", type=int, default=121)
    p.add_argument("--size", type=str, default="1280*704")

    p.add_argument(
        "--fps",
        type=int,
        default=None,
        help=(
            "Output video FPS. If not set, use cfg.sample_fps from Wan config. "
            "For Wan2.2-TI2V-5B, native cfg.sample_fps is usually 24."
        ),
    )

    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--seed_strategy",
        type=str,
        default="case_hash",
        choices=("case_hash", "sequential", "same"),
        help=(
            "Seed strategy when --random_seed is not used. "
            "case_hash: deterministic seed from base seed and case_id; recommended for benchmarks. "
            "sequential: seed = base_seed + case_index - 1. "
            "same: all cases use the same base seed."
        ),
    )
    p.add_argument("--random_seed", action="store_true")

    p.add_argument("--sample_steps", "--sampling_steps", dest="sample_steps", type=int, default=None)
    p.add_argument("--sample_solver", type=str, default="unipc")
    p.add_argument("--sample_shift", type=float, default=None)
    p.add_argument("--sample_guide_scale", "--guide_scale", dest="sample_guide_scale", type=float, default=None)

    p.add_argument("--t5_fsdp", action="store_true")
    p.add_argument("--dit_fsdp", action="store_true")
    p.add_argument("--ulysses_size", type=int, default=1)

    p.add_argument("--t5_cpu", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--convert_model_dtype", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--offload_model", type=str, default=None, choices=("true", "false"))

    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only_series", nargs="*", default=None)
    p.add_argument("--save_frames", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def str_to_bool(value):
    if value is None:
        return None
    return value.lower() == "true"


def main():
    global torch

    setup_logging()
    args = parse_args()

    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be a positive integer")

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if rank != 0:
        logging.getLogger().setLevel(logging.ERROR)

    code_dir = args.code_dir if (args.code_dir / "wan").exists() else args.model_dir
    sys.path.insert(0, str(code_dir.resolve()))

    import torch as torch_mod
    import torch.distributed as dist
    from PIL import Image

    torch = torch_mod

    import wan
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    from wan.distributed.util import init_distributed_group
    from wan.utils.utils import save_video

    cases = collect_cases(args.dataset_dir, args.only_series, args.limit)
    manifest_path = args.output_dir / "manifest.json"

    rng = random.Random(args.seed)

    if rank == 0:
        logging.info("Found %d cases", len(cases))
        logging.info(
            "Seed strategy: %s, base seed: %s, random_seed: %s",
            args.seed_strategy,
            args.seed,
            args.random_seed,
        )

    if args.dry_run:
        if rank == 0:
            for idx, case in enumerate(cases, start=1):
                seed = choose_seed(args, case, idx, rng)
                logging.info(
                    "%s seed=%s -> %s",
                    case.case_id,
                    seed,
                    case_video_path(args.output_dir, case),
                )
        return

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        device_id = local_rank
    else:
        if args.t5_fsdp or args.dit_fsdp or args.ulysses_size > 1:
            raise ValueError("FSDP/Ulysses require torchrun with WORLD_SIZE > 1")
        device_id = args.device_id

    if args.ulysses_size > 1:
        if args.ulysses_size != world_size:
            raise ValueError("--ulysses_size must equal WORLD_SIZE")
        init_distributed_group()

    cfg = WAN_CONFIGS["ti2v-5B"]

    sample_steps = args.sample_steps if args.sample_steps is not None else cfg.sample_steps
    sample_shift = args.sample_shift if args.sample_shift is not None else cfg.sample_shift
    sample_guide_scale = (
        args.sample_guide_scale
        if args.sample_guide_scale is not None
        else cfg.sample_guide_scale
    )

    native_fps = cfg.sample_fps
    save_fps = args.fps if args.fps is not None else native_fps

    offload_model = str_to_bool(args.offload_model)
    if offload_model is None:
        offload_model = False if world_size > 1 else True

    max_area = MAX_AREA_CONFIGS[args.size]

    if rank == 0:
        logging.info("Creating WanTI2V once for batch inference")
        logging.info("Prompt extension: disabled")
        logging.info(
            "Config: size=%s frame_num=%s sample_steps=%s native_fps=%s save_fps=%s",
            args.size,
            args.frame_num,
            sample_steps,
            native_fps,
            save_fps,
        )
        logging.info(
            "Duration: encoded_duration=%.4fs frame_interval_duration=%.4fs",
            args.frame_num / save_fps,
            (args.frame_num - 1) / save_fps if args.frame_num > 0 else 0.0,
        )
        logging.info(
            "Distributed: world_size=%s ulysses_size=%s t5_fsdp=%s dit_fsdp=%s",
            world_size,
            args.ulysses_size,
            args.t5_fsdp,
            args.dit_fsdp,
        )
        logging.info(
            "Memory options: t5_cpu=%s convert_model_dtype=%s offload_model=%s",
            args.t5_cpu,
            args.convert_model_dtype,
            offload_model,
        )

    model = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=str(args.model_dir.resolve()),
        device_id=device_id,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    run_record = {
        **base_metadata(
            MODEL_NAME,
            args.model_dir,
            args.dataset_dir,
            args.output_dir,
            args,
            f"official WanTI2V batch runner from {code_dir}",
        ),
        "started_at_utc": utc_now(),
        "seed_policy": {
            "base_seed": args.seed,
            "seed_strategy": args.seed_strategy,
            "random_seed": args.random_seed,
        },
        "generation_config": {
            "size": args.size,
            "frame_num": args.frame_num,
            "native_fps": native_fps,
            "save_fps": save_fps,
            "encoded_duration_seconds": args.frame_num / save_fps,
            "frame_interval_duration_seconds": (
                (args.frame_num - 1) / save_fps if args.frame_num > 0 else 0.0
            ),
            "sample_steps": sample_steps,
            "sample_solver": args.sample_solver,
            "sample_shift": sample_shift,
            "sample_guide_scale": sample_guide_scale,
        },
        "cases": [],
    }

    for idx, case in enumerate(cases, start=1):
        video_path = case_video_path(args.output_dir, case)
        frames_dir = case_frames_dir(args.output_dir, case) if args.save_frames else None
        seed = choose_seed(args, case, idx, rng)

        if rank == 0 and video_path.exists() and not args.overwrite:
            logging.info(
                "[%d/%d] skip existing %s seed=%s",
                idx,
                len(cases),
                case.case_id,
                seed,
            )
            run_record["cases"].append(
                {
                    "case_id": case.case_id,
                    "series": case.series,
                    "scene": case.scene,
                    "status": "skipped_existing",
                    "output_video": str(video_path.resolve()),
                    "seed": seed,
                    "base_seed": args.seed,
                    "seed_strategy": "random_seed" if args.random_seed else args.seed_strategy,
                    "native_fps": native_fps,
                    "save_fps": save_fps,
                    "frame_num": args.frame_num,
                    "encoded_duration_seconds": args.frame_num / save_fps,
                    "frame_interval_duration_seconds": (
                        (args.frame_num - 1) / save_fps if args.frame_num > 0 else 0.0
                    ),
                }
            )
            continue

        if world_size > 1:
            skip_flag = [video_path.exists() and not args.overwrite] if rank == 0 else [None]
            dist.broadcast_object_list(skip_flag, src=0)
            if skip_flag[0]:
                continue

        try:
            prompt = read_prompt(case)
            image = Image.open(case.image_path).convert("RGB")
            meta = case_metadata(case, prompt, video_path, frames_dir)

            if rank == 0:
                logging.info(
                    "[%d/%d] generating %s seed=%s",
                    idx,
                    len(cases),
                    case.case_id,
                    seed,
                )

            video = model.generate(
                input_prompt=prompt,
                img=image,
                max_area=max_area,
                frame_num=args.frame_num,
                shift=sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=sample_steps,
                guide_scale=sample_guide_scale,
                seed=seed,
                offload_model=offload_model,
            )

            if rank == 0:
                save_video(
                    tensor=video[None],
                    save_file=str(video_path),
                    fps=save_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )

                if frames_dir:
                    meta["exported_frame_count"] = export_frames_from_video(
                        video_path,
                        frames_dir,
                        save_fps,
                    )

                meta.update(
                    {
                        "index": idx,
                        "status": "ok",
                        "seed": seed,
                        "base_seed": args.seed,
                        "seed_strategy": "random_seed" if args.random_seed else args.seed_strategy,
                        "native_fps": native_fps,
                        "save_fps": save_fps,
                        "frame_num": args.frame_num,
                        "encoded_duration_seconds": args.frame_num / save_fps,
                        "frame_interval_duration_seconds": (
                            (args.frame_num - 1) / save_fps if args.frame_num > 0 else 0.0
                        ),
                        "completed_at_utc": utc_now(),
                    }
                )
                run_record["cases"].append(meta)

            del video
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as exc:
            if rank == 0:
                logging.exception("Failed case %s", case.case_id)
                run_record["cases"].append(
                    {
                        "case_id": case.case_id,
                        "series": case.series,
                        "scene": case.scene,
                        "status": "failed",
                        "seed": seed,
                        "base_seed": args.seed,
                        "seed_strategy": "random_seed" if args.random_seed else args.seed_strategy,
                        "native_fps": native_fps,
                        "save_fps": save_fps,
                        "frame_num": args.frame_num,
                        "encoded_duration_seconds": args.frame_num / save_fps,
                        "frame_interval_duration_seconds": (
                            (args.frame_num - 1) / save_fps if args.frame_num > 0 else 0.0
                        ),
                        **record_failure(exc),
                        "completed_at_utc": utc_now(),
                    }
                )

            if world_size > 1:
                raise

    if rank == 0:
        run_record["completed_at_utc"] = utc_now()
        append_manifest(manifest_path, run_record)
        logging.info("Batch complete: %s", args.output_dir)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
