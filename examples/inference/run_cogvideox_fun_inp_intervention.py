import argparse
import json
import logging
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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


MODEL_NAME = "CogVideoX-Fun-V1.5-5b-InP"


def parse_args():
    root = default_project_root()
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_dir", type=Path, default=root / "data" / "inputs" / "L1")

    p.add_argument("--model_dir", type=Path, default=root / MODEL_NAME)

    p.add_argument("--code_dir", type=Path, default=root / "ModelCode" / "VideoX-Fun")
    p.add_argument("--output_dir", type=Path, default=root / MODEL_NAME / "output")

    p.add_argument("--frame_num", type=int, default=85)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=1360)
    p.add_argument("--fps", type=int, default=8)

    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "The video is not of a high quality, it has a low resolution. "
            "Watermark present in each frame. Strange body and strange trajectory. Distortion."
        ),
    )
    p.add_argument(
        "--sampler_name",
        choices=("Euler", "Euler A", "DPM++", "PNDM", "DDIM_Cog", "DDIM_Origin"),
        default="DDIM_Origin",
    )
    p.add_argument(
        "--gpu_memory_mode",
        choices=(
            "model_full_load",
            "model_full_load_and_qfloat8",
            "model_cpu_offload",
            "model_cpu_offload_and_qfloat8",
            "sequential_cpu_offload",
        ),
        default="model_cpu_offload_and_qfloat8",
    )
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seed_strategy",
        type=str,
        default="case_hash",
        choices=("case_hash", "sequential", "same"),
        help=(
            "case_hash: deterministic seed from base seed and case_id; recommended. "
            "sequential: seed = base_seed + case_index - 1. "
            "same: all cases use the same base seed."
        ),
    )
    p.add_argument("--random_seed", action="store_true")

    p.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated physical GPU ids, e.g. 0,1,2,3.",
    )
    p.add_argument(
        "--parallel_cases",
        type=int,
        default=None,
        help="Number of persistent worker processes. Defaults to len(--gpus).",
    )

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only_series", nargs="*", default=None)
    p.add_argument("--save_frames", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker_id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--worker_count", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--worker_gpu", type=str, default="0", help=argparse.SUPPRESS)
    p.add_argument("--worker_manifest", type=Path, default=None, help=argparse.SUPPRESS)

    return p.parse_args()


def build_worker_cmd(args, worker_id: int, worker_count: int, worker_gpu: str, worker_manifest: Path) -> List[str]:
    script = Path(__file__).resolve()

    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--worker",
        "--worker_id",
        str(worker_id),
        "--worker_count",
        str(worker_count),
        "--worker_gpu",
        str(worker_gpu),
        "--worker_manifest",
        str(worker_manifest),

        "--dataset_dir",
        str(args.dataset_dir),
        "--model_dir",
        str(args.model_dir),
        "--code_dir",
        str(args.code_dir),
        "--output_dir",
        str(args.output_dir),

        "--frame_num",
        str(args.frame_num),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--fps",
        str(args.fps),

        "--num_inference_steps",
        str(args.num_inference_steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--negative_prompt",
        args.negative_prompt,
        "--sampler_name",
        args.sampler_name,
        "--gpu_memory_mode",
        args.gpu_memory_mode,
        "--dtype",
        args.dtype,

        "--seed",
        str(args.seed),
        "--seed_strategy",
        args.seed_strategy,

        "--gpus",
        str(worker_gpu),
        "--parallel_cases",
        "1",
    ]

    if args.random_seed:
        cmd.append("--random_seed")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.only_series:
        cmd += ["--only_series"] + list(args.only_series)
    if args.save_frames:
        cmd.append("--save_frames")

    return cmd


def make_worker_env(gpu: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    return env


def validate_args(args):
    if args.frame_num <= 0:
        raise ValueError("--frame_num must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive")

    gpu_ids = parse_gpu_list(args.gpus)
    if not args.worker:
        if not gpu_ids:
            raise ValueError("--gpus must contain at least one GPU id, e.g. --gpus 0 or --gpus 0,1,2,3")
        workers = args.parallel_cases or len(gpu_ids)
        if workers < 1:
            raise ValueError("--parallel_cases must be >= 1")
        if workers > len(gpu_ids):
            raise ValueError("--parallel_cases cannot exceed number of entries in --gpus")


def load_runtime(args):
    if not (args.code_dir / "videox_fun").exists():
        raise FileNotFoundError(f"VideoX-Fun code_dir does not contain videox_fun package: {args.code_dir}")

    sys.path.insert(0, str(args.code_dir.resolve()))

    import torch
    from diffusers import (
        CogVideoXDDIMScheduler,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        PNDMScheduler,
    )

    from videox_fun.dist import set_multi_gpus_devices
    from videox_fun.models import (
        AutoencoderKLCogVideoX,
        CogVideoXTransformer3DModel,
        T5EncoderModel,
        T5Tokenizer,
    )
    from videox_fun.pipeline import CogVideoXFunInpaintPipeline, CogVideoXFunPipeline
    from videox_fun.utils.fp8_optimization import (
        convert_model_weight_to_float8,
        convert_weight_dtype_wrapper,
    )
    from videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid

    weight_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # Each worker sees one GPU through CUDA_VISIBLE_DEVICES.
    device = set_multi_gpus_devices(1, 1)

    logging.info("Loading VideoX-Fun CogVideoX-Fun model from %s", args.model_dir)
    logging.info("Visible CUDA devices: %s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    logging.info("Runtime device: %s", device)

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        str(args.model_dir.resolve()),
        subfolder="transformer",
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).to(weight_dtype)

    vae = AutoencoderKLCogVideoX.from_pretrained(
        str(args.model_dir.resolve()),
        subfolder="vae",
    ).to(weight_dtype)

    tokenizer = T5Tokenizer.from_pretrained(
        str(args.model_dir.resolve()),
        subfolder="tokenizer",
    )

    text_encoder = T5EncoderModel.from_pretrained(
        str(args.model_dir.resolve()),
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )

    scheduler_cls = {
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler,
        "PNDM": PNDMScheduler,
        "DDIM_Cog": CogVideoXDDIMScheduler,
        "DDIM_Origin": DDIMScheduler,
    }[args.sampler_name]

    scheduler = scheduler_cls.from_pretrained(
        str(args.model_dir.resolve()),
        subfolder="scheduler",
    )

    if transformer.config.in_channels != vae.config.latent_channels:
        pipeline = CogVideoXFunInpaintPipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
    else:
        pipeline = CogVideoXFunPipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )

    if args.gpu_memory_mode == "sequential_cpu_offload":
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=[], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=[], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    return {
        "torch": torch,
        "pipeline": pipeline,
        "vae": vae,
        "transformer": transformer,
        "device": device,
        "get_image_to_video_latent": get_image_to_video_latent,
        "save_videos_grid": save_videos_grid,
    }


def adjust_video_length(args, vae, transformer) -> int:
    video_length = args.frame_num

    if video_length != 1:
        ratio = vae.config.temporal_compression_ratio
        video_length = int((video_length - 1) // ratio * ratio) + 1

    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    if (
        video_length != 1
        and transformer.config.patch_size_t is not None
        and latent_frames % transformer.config.patch_size_t != 0
    ):
        video_length += (
            transformer.config.patch_size_t - latent_frames % transformer.config.patch_size_t
        ) * vae.config.temporal_compression_ratio

    return video_length


def process_one_case(args, runtime, case, idx: int, total: int, seed: int) -> Dict:
    torch = runtime["torch"]
    pipeline = runtime["pipeline"]
    vae = runtime["vae"]
    transformer = runtime["transformer"]
    device = runtime["device"]
    get_image_to_video_latent = runtime["get_image_to_video_latent"]
    save_videos_grid = runtime["save_videos_grid"]

    video_path = case_video_path(args.output_dir, case)
    frames_dir = case_frames_dir(args.output_dir, case) if args.save_frames else None
    prompt = read_prompt(case)

    meta = case_metadata(case, prompt, video_path, frames_dir)
    meta.update(
        {
            "index": idx,
            "seed": seed,
            "base_seed": args.seed,
            "seed_strategy": "random_seed" if args.random_seed else args.seed_strategy,
            "worker_id": args.worker_id if args.worker else None,
            "worker_gpu": args.worker_gpu if args.worker else None,
            "height": args.height,
            "width": args.width,
            "requested_frame_num": args.frame_num,
            "fps": args.fps,
            "requested_duration_seconds": args.frame_num / args.fps if args.fps else None,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "sampler_name": args.sampler_name,
            "gpu_memory_mode": args.gpu_memory_mode,
            "dtype": args.dtype,
        }
    )

    if video_path.exists() and not args.overwrite:
        meta["status"] = "skipped_existing"
        logging.info(
            "[%d/%d] skip existing %s seed=%s -> %s",
            idx,
            total,
            case.case_id,
            seed,
            video_path,
        )
        return meta

    try:
        generator = torch.Generator(device=device).manual_seed(seed)

        video_length = adjust_video_length(args, vae, transformer)
        sample_size = [args.height, args.width]

        input_video, input_video_mask, _ = get_image_to_video_latent(
            str(case.image_path.resolve()),
            None,
            video_length=video_length,
            sample_size=sample_size,
        )

        logging.info(
            "[%d/%d] generating %s seed=%s requested_frames=%s actual_frames=%s size=%sx%s fps=%s",
            idx,
            total,
            case.case_id,
            seed,
            args.frame_num,
            video_length,
            args.width,
            args.height,
            args.fps,
        )

        start = time.time()

        with torch.no_grad():
            sample = pipeline(
                prompt,
                num_frames=video_length,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                generator=generator,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                video=input_video,
                mask_video=input_video_mask,
            ).videos

        save_videos_grid(sample, str(video_path), fps=args.fps)

        if frames_dir:
            meta["exported_frame_count"] = export_frames_from_video(
                video_path,
                frames_dir,
                args.fps,
            )

        elapsed = time.time() - start

        meta.update(
            {
                "status": "ok",
                "actual_frame_num": video_length,
                "actual_duration_seconds": video_length / args.fps if args.fps else None,
                "elapsed_seconds": elapsed,
                "completed_at_utc": utc_now(),
            }
        )

        logging.info(
            "[%d/%d] completed %s seed=%s elapsed=%.1fs -> %s",
            idx,
            total,
            case.case_id,
            seed,
            elapsed,
            video_path,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as exc:
        meta.update(
            {
                "status": "failed",
                **record_failure(exc),
                "completed_at_utc": utc_now(),
            }
        )
        logging.exception("Failed case %s", case.case_id)

    return meta


def run_worker(args):
    cases = collect_cases(args.dataset_dir, args.only_series, args.limit)
    assigned = [
        (idx, case)
        for idx, case in enumerate(cases, start=1)
        if (idx - 1) % args.worker_count == args.worker_id
    ]

    logging.info(
        "Worker %s/%s on gpu=%s assigned %d/%d cases",
        args.worker_id,
        args.worker_count,
        args.worker_gpu,
        len(assigned),
        len(cases),
    )

    worker_record = {
        "worker_id": args.worker_id,
        "worker_count": args.worker_count,
        "worker_gpu": args.worker_gpu,
        "started_at_utc": utc_now(),
        "cases": [],
    }

    try:
        runtime = load_runtime(args)
        rng = random.Random(args.seed + args.worker_id * 100003)

        for idx, case in assigned:
            seed = choose_seed(args, case, idx, rng)
            meta = process_one_case(args, runtime, case, idx, len(cases), seed)
            worker_record["cases"].append(meta)

    except Exception as exc:
        worker_record["worker_error"] = record_failure(exc)
        logging.exception("Worker %s crashed", args.worker_id)

    worker_record["completed_at_utc"] = utc_now()

    if args.worker_manifest is not None:
        args.worker_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.worker_manifest.write_text(
            json.dumps(worker_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if worker_record.get("worker_error"):
        raise RuntimeError(f"Worker {args.worker_id} failed before/during processing")


def run_parallel_parent(args, cases, manifest_path: Path):
    gpu_ids = parse_gpu_list(args.gpus)
    workers = args.parallel_cases or len(gpu_ids)
    workers = min(workers, len(cases)) if cases else workers

    tmp_dir = args.output_dir / "_worker_manifests" / f"run_{int(time.time())}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Launching %d persistent workers on GPUs: %s", workers, gpu_ids[:workers])

    procs = []
    for worker_id in range(workers):
        gpu = gpu_ids[worker_id]
        worker_manifest = tmp_dir / f"worker_{worker_id}.json"
        cmd = build_worker_cmd(args, worker_id, workers, gpu, worker_manifest)
        env = make_worker_env(gpu)

        logging.info(
            "[parent] launch worker=%s gpu=%s cmd=%s",
            worker_id,
            gpu,
            " ".join(shlex.quote(str(x)) for x in cmd),
        )

        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(Path.cwd()),
        )
        procs.append((worker_id, gpu, worker_manifest, proc))

    worker_status = []
    for worker_id, gpu, worker_manifest, proc in procs:
        returncode = proc.wait()
        worker_status.append(
            {
                "worker_id": worker_id,
                "gpu": gpu,
                "returncode": returncode,
                "manifest": str(worker_manifest),
            }
        )
        if returncode != 0:
            logging.error("[parent] worker=%s gpu=%s failed returncode=%s", worker_id, gpu, returncode)
        else:
            logging.info("[parent] worker=%s gpu=%s finished ok", worker_id, gpu)

    all_case_records = []
    worker_records = []

    for worker_id, gpu, worker_manifest, proc in procs:
        if worker_manifest.exists():
            data = json.loads(worker_manifest.read_text(encoding="utf-8"))
            worker_records.append(data)
            all_case_records.extend(data.get("cases", []))
        else:
            worker_records.append(
                {
                    "worker_id": worker_id,
                    "worker_gpu": gpu,
                    "worker_error": {
                        "exception_type": "MissingWorkerManifest",
                        "exception_message": f"Worker manifest not found: {worker_manifest}",
                    },
                    "cases": [],
                }
            )

    all_case_records.sort(key=lambda x: x.get("index", 10**9))

    run_record = {
        **base_metadata(
            MODEL_NAME,
            args.model_dir,
            args.dataset_dir,
            args.output_dir,
            args,
            "VideoX-Fun CogVideoXFunInpaintPipeline, persistent case-parallel workers",
        ),
        "started_at_utc": utc_now(),
        "seed_policy": {
            "base_seed": args.seed,
            "seed_strategy": args.seed_strategy,
            "random_seed": args.random_seed,
        },
        "generation_config": {
            "height": args.height,
            "width": args.width,
            "frame_num": args.frame_num,
            "fps": args.fps,
            "requested_duration_seconds": args.frame_num / args.fps if args.fps else None,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "negative_prompt": args.negative_prompt,
            "sampler_name": args.sampler_name,
            "gpu_memory_mode": args.gpu_memory_mode,
            "dtype": args.dtype,
        },
        "multi_gpu": {
            "enabled": workers > 1,
            "mode": "case_parallel_persistent_workers",
            "gpus": gpu_ids[:workers],
            "parallel_cases": workers,
        },
        "worker_status": worker_status,
        "worker_records": worker_records,
        "cases": all_case_records,
        "completed_at_utc": utc_now(),
    }

    append_manifest(manifest_path, run_record)
    logging.info("Batch complete: %s", args.output_dir)


def run_single_process(args, cases, manifest_path: Path):
    runtime = load_runtime(args)
    rng = random.Random(args.seed)

    run_record = {
        **base_metadata(
            MODEL_NAME,
            args.model_dir,
            args.dataset_dir,
            args.output_dir,
            args,
            "official VideoX-Fun CogVideoXFunInpaintPipeline",
        ),
        "started_at_utc": utc_now(),
        "seed_policy": {
            "base_seed": args.seed,
            "seed_strategy": args.seed_strategy,
            "random_seed": args.random_seed,
        },
        "generation_config": {
            "height": args.height,
            "width": args.width,
            "frame_num": args.frame_num,
            "fps": args.fps,
            "requested_duration_seconds": args.frame_num / args.fps if args.fps else None,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "negative_prompt": args.negative_prompt,
            "sampler_name": args.sampler_name,
            "gpu_memory_mode": args.gpu_memory_mode,
            "dtype": args.dtype,
        },
        "cases": [],
    }

    for idx, case in enumerate(cases, start=1):
        seed = choose_seed(args, case, idx, rng)
        meta = process_one_case(args, runtime, case, idx, len(cases), seed)
        run_record["cases"].append(meta)

    run_record["completed_at_utc"] = utc_now()
    append_manifest(manifest_path, run_record)


def main():
    setup_logging()
    args = parse_args()
    validate_args(args)

    if args.worker:
        run_worker(args)
        return

    cases = collect_cases(args.dataset_dir, args.only_series, args.limit)
    manifest_path = args.output_dir / "manifest.json"

    logging.info("Found %d cases", len(cases))
    logging.info(
        "Seed strategy: %s, base seed: %s, random_seed: %s",
        args.seed_strategy,
        args.seed,
        args.random_seed,
    )
    logging.info(
        "Config: size=%sx%s frame_num=%s fps=%s duration=%.4fs steps=%s guidance=%s sampler=%s memory=%s",
        args.width,
        args.height,
        args.frame_num,
        args.fps,
        args.frame_num / args.fps,
        args.num_inference_steps,
        args.guidance_scale,
        args.sampler_name,
        args.gpu_memory_mode,
    )

    gpu_ids = parse_gpu_list(args.gpus)
    workers = args.parallel_cases or len(gpu_ids)

    if args.dry_run:
        rng = random.Random(args.seed)
        for idx, case in enumerate(cases, start=1):
            seed = choose_seed(args, case, idx, rng)
            gpu = gpu_ids[(idx - 1) % len(gpu_ids)] if gpu_ids else "N/A"
            logging.info(
                "%s seed=%s gpu=%s -> %s",
                case.case_id,
                seed,
                gpu,
                case_video_path(args.output_dir, case),
            )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if workers > 1:
        run_parallel_parent(args, cases, manifest_path)
    else:
        run_single_process(args, cases, manifest_path)


if __name__ == "__main__":
    main()
