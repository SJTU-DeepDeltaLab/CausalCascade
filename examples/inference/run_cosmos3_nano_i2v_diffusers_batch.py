#!/usr/bin/env python3
"""Batch Cosmos3-Nano I2V inference with the Diffusers checkpoint layout.

Expected dataset layout:
  Dataset/<series>/<scene>/first_frame.png
  Dataset/<series>/<scene>/prompt.txt

Example:
  CUDA_VISIBLE_DEVICES=0 python -u run_cosmos3_nano_i2v_diffusers_batch.py \
    --dataset_dir /path/to/Dataset \
    --model_dir /path/to/Cosmos3-Nano \
    --output_dir /path/to/output \
    --height 480 --width 832 --num_frames 189 --fps 24 \
    --seed 42 --seed_strategy case_hash --limit 1 --overwrite
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    safe_name,
    stable_case_seed,
    setup_logging,
    utc_now,
)


MODEL_NAME = "Cosmos3-Nano"
BACKEND = "diffusers.Cosmos3OmniPipeline"
DEFAULT_MODEL_DIR = default_project_root() / "Cosmos3-Nano"
DEFAULT_DATASET_DIR = default_project_root() / "data" / "inputs" / "L1"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR / "output_diffusers_480p_189f_24fps_casehash_seed42"


DEFAULT_NUM_FRAMES = 189
DEFAULT_FPS = 24
DEFAULT_NUM_STEPS = 35
DEFAULT_GUIDANCE = 6.0
DEFAULT_SHIFT = 10.0
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832


_PIPE: Any = None
_PIPE_MODEL_ID: Optional[str] = None


def maybe_prepend_vendor_dir(vendor_dir: Optional[Path]) -> None:
    """Use an optional dependency directory for this inference process."""

    if not vendor_dir:
        return
    vendor_dir = vendor_dir.expanduser().resolve()
    if vendor_dir.exists() and str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))


def parse_boolish(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def dtype_from_string(dtype_name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}; choose one of {sorted(mapping)}")
    return mapping[key]


def resolve_seed(args: argparse.Namespace, case_id: str, case_index: int) -> int:
    if args.seed_strategy == "case_hash":
        return stable_case_seed(args.seed, case_id, args.seed_hash_mod)
    if args.seed_strategy == "same":
        return int(args.seed)
    if args.seed_strategy == "sequential":
        return int(args.seed) + int(case_index)
    if args.seed_strategy == "random":
        return random.SystemRandom().randint(0, int(args.seed_hash_mod) - 1)
    raise ValueError(f"Unknown seed_strategy: {args.seed_strategy}")


def compact_json_file(path: Path) -> str:
    return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=True, separators=(",", ":"))


def load_negative_prompt(args: argparse.Namespace, model_dir: Path) -> str:
    if args.negative_prompt is not None:
        return args.negative_prompt

    if args.negative_prompt_file is None:
        return ""

    if args.negative_prompt_file == "auto":
        candidates = [
            model_dir / "assets" / "negative_prompt.json",
            model_dir / "assets" / "negative_prompts" / "image2video" / "neg_prompt.json",
            model_dir / "negative_prompt.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                try:
                    return compact_json_file(candidate)
                except Exception:
                    return candidate.read_text(encoding="utf-8").strip()
        logging.warning("No auto negative prompt file found under %s; using empty negative prompt", model_dir)
        return ""

    path = Path(args.negative_prompt_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"negative_prompt_file does not exist: {path}")
    try:
        return compact_json_file(path)
    except Exception:
        return path.read_text(encoding="utf-8").strip()


def maybe_resize_image(image: Any, width: int, height: int, mode: str) -> Any:
    if mode == "none":
        return image
    from PIL import ImageOps

    if mode == "resize":
        return image.resize((width, height))
    if mode == "fit":
        return ImageOps.fit(image, (width, height), method=0)
    if mode == "contain":
        # Pads to target size without cropping, preserving aspect ratio.
        return ImageOps.pad(image, (width, height), method=0)
    raise ValueError(f"Unsupported image_resize_mode: {mode}")


def import_diffusers_objects() -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Import diffusers lazily so --vendor_dir can take effect first."""

    import torch
    from diffusers import Cosmos3OmniPipeline
    from diffusers import logging as diffusers_logging
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import encode_video, export_to_video, load_image

    return torch, Cosmos3OmniPipeline, diffusers_logging, UniPCMultistepScheduler, encode_video, export_to_video, load_image


def print_runtime_versions() -> None:
    try:
        import diffusers
        import torch
        import transformers

        logging.info("torch: %s cuda=%s", torch.__version__, getattr(torch.version, "cuda", None))
        logging.info("diffusers: %s from %s", diffusers.__version__, Path(diffusers.__file__).resolve())
        logging.info("transformers: %s from %s", transformers.__version__, Path(transformers.__file__).resolve())
    except Exception as exc:
        logging.warning("Could not print runtime versions: %s", exc)


def pipeline_from_pretrained_kwargs(
    pipeline_cls: Any,
    model_dir: Path,
    torch_dtype: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": str(model_dir),
        "torch_dtype": torch_dtype,
        "local_files_only": not args.allow_remote,
    }

    optional = {
        "safety_checker": None,
        "enable_safety_checker": bool(args.enable_safety_checker),
        "low_cpu_mem_usage": bool(args.low_cpu_mem_usage),
        "device_map": args.device_map if args.device_map else None,
        "token": os.environ.get("HF_TOKEN") or None,
    }

    try:
        sig = inspect.signature(pipeline_cls.from_pretrained)
        accepted = set(sig.parameters)
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        for key, value in optional.items():
            if value is not None and (has_var_kwargs or key in accepted):
                kwargs[key] = value
    except Exception:
        for key, value in optional.items():
            if value is not None:
                kwargs[key] = value

    return kwargs


def get_pipe(model_dir: Path, args: argparse.Namespace):
    global _PIPE, _PIPE_MODEL_ID

    torch, Cosmos3OmniPipeline, diffusers_logging, UniPCMultistepScheduler, *_ = import_diffusers_objects()

    model_id = str(model_dir.resolve())
    if _PIPE is not None and _PIPE_MODEL_ID == model_id:
        return _PIPE

    if _PIPE is not None:
        del _PIPE
        _PIPE = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    diffusers_logging.set_verbosity_info()
    torch_dtype = dtype_from_string(args.dtype)

    logging.info("Loading Cosmos3OmniPipeline from %s", model_id)
    t0 = time.time()
    kwargs = pipeline_from_pretrained_kwargs(Cosmos3OmniPipeline, model_dir, torch_dtype, args)

    try:
        pipe = Cosmos3OmniPipeline.from_pretrained(**kwargs)
    except TypeError as exc:
        logging.warning("from_pretrained kwargs rejected (%s); retrying with minimal kwargs", exc)
        pipe = Cosmos3OmniPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            local_files_only=not args.allow_remote,
        )

    try:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=float(args.shift))
    except Exception as exc:
        logging.warning("Could not set UniPCMultistepScheduler(flow_shift=%s): %s", args.shift, exc)

    if args.cpu_offload:
        logging.info("Enabling model CPU offload")
        pipe.enable_model_cpu_offload()
    elif args.sequential_cpu_offload:
        logging.info("Enabling sequential CPU offload")
        pipe.enable_sequential_cpu_offload()
    elif not args.device_map:
        logging.info("Moving pipeline to %s", args.device)
        pipe.to(args.device)

    _PIPE = pipe
    _PIPE_MODEL_ID = model_id
    logging.info("Loaded pipeline in %.1fs", time.time() - t0)
    return _PIPE


def normalize_video_frames(result: Any) -> Any:
    """Extract frames from several possible diffusers output shapes."""

    for attr in ("video", "frames", "videos", "images"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if value is not None:
                # Some pipelines return [frames], others return frames directly.
                if isinstance(value, list) and value and isinstance(value[0], list):
                    return value[0]
                return value
    raise AttributeError("Could not find video/frames/videos/images in pipeline output")


def call_pipe_i2v(pipe: Any, image: Any, prompt: str, negative_prompt: str, seed: int, args: argparse.Namespace) -> Any:
    import torch

    # Generator device should match the active CUDA device when CUDA is used.
    gen_device = "cuda" if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(int(seed))

    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": image,
        "num_frames": int(args.num_frames),
        "height": int(args.height),
        "width": int(args.width),
        "fps": int(args.fps),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "enable_sound": bool(args.enable_sound),
        "add_resolution_template": bool(args.add_resolution_template),
        "add_duration_template": bool(args.add_duration_template),
        "generator": generator,
    }

    try:
        sig = inspect.signature(pipe.__call__)
        accepted = set(sig.parameters)
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if not has_var_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    except Exception:
        pass

    return pipe(**kwargs)


def save_result_video(result: Any, pipe: Any, output_path: Path, fps: int, enable_sound: bool) -> None:
    _, _, _, _, encode_video, export_to_video, _ = import_diffusers_objects()

    video = normalize_video_frames(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sound = getattr(result, "sound", None)
    if enable_sound and sound is not None:
        sample_rate = None
        try:
            sample_rate = pipe.sound_tokenizer.config.sampling_rate
        except Exception:
            sample_rate = 48000
        encode_video(
            video,
            fps=int(fps),
            output_path=str(output_path),
            audio=sound,
            audio_sample_rate=sample_rate,
        )
    else:
        export_to_video(video, str(output_path), fps=int(fps), macro_block_size=1)


def run_one_case(
    case: Any,
    case_index: int,
    pipe: Any,
    negative_prompt: str,
    model_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    _, _, _, _, _, _, load_image = import_diffusers_objects()

    prompt = read_prompt(case)
    seed = resolve_seed(args, case.case_id, case_index)
    video_path = case_video_path(output_dir, case)
    frames_dir = case_frames_dir(output_dir, case) if args.save_frames else None

    if video_path.exists() and not args.overwrite:
        logging.info("Skipping existing output: %s", video_path)
        run_record = {
            **case_metadata(case, prompt, video_path, frames_dir),
            "status": "skipped_existing",
            "seed": seed,
            "created_at_utc": utc_now(),
        }
        append_manifest(manifest_path, run_record)
        return run_record

    logging.info("[%04d] %s seed=%s -> %s", case_index, case.case_id, seed, video_path)

    if args.dry_run:
        run_record = {
            **case_metadata(case, prompt, video_path, frames_dir),
            "status": "dry_run",
            "seed": seed,
            "created_at_utc": utc_now(),
        }
        append_manifest(manifest_path, run_record)
        return run_record

    t0 = time.time()
    image = load_image(str(case.image_path))
    image = maybe_resize_image(image, args.width, args.height, args.image_resize_mode)

    result = call_pipe_i2v(pipe, image, prompt, negative_prompt, seed, args)
    save_result_video(result, pipe, video_path, args.fps, args.enable_sound)

    frame_count = None
    if frames_dir:
        frame_count = export_frames_from_video(video_path, frames_dir, fps=args.fps)

    elapsed = time.time() - t0
    run_record = {
        **case_metadata(case, prompt, video_path, frames_dir),
        "status": "success",
        "seed": seed,
        "seed_strategy": args.seed_strategy,
        "num_frames": int(args.num_frames),
        "fps": int(args.fps),
        "height": int(args.height),
        "width": int(args.width),
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "shift": float(args.shift),
        "enable_sound": bool(args.enable_sound),
        "negative_prompt_sha256": hashlib.sha256(negative_prompt.encode("utf-8")).hexdigest(),
        "elapsed_seconds": elapsed,
        "exported_frame_count": frame_count,
        "created_at_utc": utc_now(),
    }
    append_manifest(manifest_path, run_record)
    logging.info("Finished %s in %.1fs", case.case_id, elapsed)
    return run_record


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch I2V runner for Cosmos3-Nano Diffusers")

    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--vendor_dir",
        type=Path,
        default=None,
        help="Optional directory installed with pip --target containing newer diffusers/transformers.",
    )

    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids for case-parallel execution, e.g. 0,1,2,3. Used only when --parallel_cases > 1.",
    )
    parser.add_argument(
        "--parallel_cases",
        type=int,
        default=1,
        help="Number of independent worker processes. Each worker loads one pipeline on one GPU and processes a disjoint subset of cases.",
    )
    parser.add_argument("--worker_id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num_workers", type=int, default=1, help=argparse.SUPPRESS)

    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE)
    parser.add_argument("--shift", type=float, default=DEFAULT_SHIFT)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed_strategy",
        choices=["case_hash", "same", "sequential", "random"],
        default="case_hash",
    )
    parser.add_argument("--seed_hash_mod", type=int, default=1_000_000_000)

    parser.add_argument("--only_series", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None, help="Optional diffusers device_map, e.g. balanced. Usually leave empty.")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--sequential_cpu_offload", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--allow_remote", action="store_true", help="Allow HF remote lookup instead of local_files_only.")

    parser.add_argument("--enable_sound", action="store_true", help="Generate audio if supported. Default off for I2V benchmark.")
    parser.add_argument("--enable_safety_checker", action="store_true", help="Enable safety checker if this diffusers build supports it.")
    parser.add_argument("--add_resolution_template", action="store_true")
    parser.add_argument("--add_duration_template", action="store_true")

    parser.add_argument(
        "--negative_prompt",
        default=None,
        help="Negative prompt string. If omitted, --negative_prompt_file is used. Pass '' for empty.",
    )
    parser.add_argument(
        "--negative_prompt_file",
        default="auto",
        help="Path to negative prompt json/txt, 'auto' to search model assets, or omit with --negative_prompt ''.",
    )
    parser.add_argument(
        "--image_resize_mode",
        choices=["none", "resize", "fit", "contain"],
        default="none",
        help="Whether to resize input first_frame before passing to pipeline. Default: none.",
    )

    return parser.parse_args(argv)


def split_gpu_list(gpus: Optional[str]) -> List[str]:
    if not gpus:
        return []
    return [x.strip() for x in gpus.split(",") if x.strip()]


def build_worker_command(args: argparse.Namespace, worker_id: int, num_workers: int) -> List[str]:
    """Reconstruct command line for a child worker."""

    cmd = [sys.executable, "-u", str(Path(__file__).resolve())]

    value_args = [
        "dataset_dir", "model_dir", "output_dir", "vendor_dir",
        "height", "width", "num_frames", "fps", "num_inference_steps",
        "guidance_scale", "shift", "seed", "seed_strategy", "seed_hash_mod",
        "limit", "dtype", "device", "device_map", "negative_prompt",
        "negative_prompt_file", "image_resize_mode",
    ]
    list_args = ["only_series"]
    bool_args = [
        "overwrite", "save_frames", "dry_run", "cpu_offload", "sequential_cpu_offload",
        "low_cpu_mem_usage", "allow_remote", "enable_sound", "enable_safety_checker",
        "add_resolution_template", "add_duration_template",
    ]

    for name in value_args:
        value = getattr(args, name)
        if value is None:
            continue
        cmd += [f"--{name}", str(value)]

    for name in list_args:
        value = getattr(args, name)
        if value:
            cmd.append(f"--{name}")
            cmd.extend(str(x) for x in value)

    for name in bool_args:
        if getattr(args, name):
            cmd.append(f"--{name}")

    cmd += ["--parallel_cases", "1", "--worker_id", str(worker_id), "--num_workers", str(num_workers)]
    return cmd


def run_parallel_parent(args: argparse.Namespace) -> int:
    """Run independent case-parallel workers, one pipeline per GPU."""

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    gpu_ids = split_gpu_list(args.gpus)
    if not gpu_ids:
        gpu_ids = [str(i) for i in range(args.parallel_cases)]
    if args.parallel_cases > len(gpu_ids):
        raise ValueError(f"--parallel_cases={args.parallel_cases} but only {len(gpu_ids)} GPU ids were provided: {gpu_ids}")

    num_workers = int(args.parallel_cases)
    logging.info("Starting case-parallel run: %d workers on GPUs %s", num_workers, gpu_ids[:num_workers])
    logging.info("Each worker loads its own Cosmos3OmniPipeline; this is case-parallel, not tensor/model parallel.")

    append_manifest(
        manifest_path,
        {
            "status": "parallel_run_started",
            "created_at_utc": utc_now(),
            "parallel_cases": num_workers,
            "gpus": gpu_ids[:num_workers],
            "command": sys.argv,
        },
    )

    procs = []
    for worker_id in range(num_workers):
        gpu = gpu_ids[worker_id]
        worker_log = output_dir / f"worker_{worker_id}_gpu{gpu}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        cmd = build_worker_command(args, worker_id, num_workers)
        logging.info("Launching worker %d on GPU %s; log=%s", worker_id, gpu, worker_log)
        logging.info("Worker command: %s", " ".join(cmd))
        f = worker_log.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(Path.cwd()))
        procs.append((worker_id, gpu, proc, f, worker_log))

    failed = 0
    for worker_id, gpu, proc, f, worker_log in procs:
        rc = proc.wait()
        f.close()
        if rc != 0:
            failed += 1
            logging.error("Worker %d on GPU %s failed with returncode=%s; see %s", worker_id, gpu, rc, worker_log)
        else:
            logging.info("Worker %d on GPU %s finished successfully; log=%s", worker_id, gpu, worker_log)

    merged_runs: List[Dict[str, Any]] = []
    for worker_id in range(num_workers):
        worker_manifest = output_dir / f"manifest_worker{worker_id}.json"
        if not worker_manifest.exists():
            continue
        try:
            payload = json.loads(worker_manifest.read_text(encoding="utf-8"))
            for record in payload.get("runs", []):
                record.setdefault("worker_id", worker_id)
                merged_runs.append(record)
        except Exception as exc:
            logging.warning("Could not merge worker manifest %s: %s", worker_manifest, exc)

    append_manifest(
        manifest_path,
        {
            "status": "parallel_run_finished",
            "created_at_utc": utc_now(),
            "parallel_cases": num_workers,
            "failed_workers": failed,
            "worker_logs": [str(p) for p in logs],
            "merged_worker_records": len(merged_runs),
        },
    )

    if merged_runs:
        merged_path = output_dir / "manifest_workers_merged.json"
        merged_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "created_at_utc": utc_now(),
                    "source": "merged from manifest_worker*.json",
                    "runs": merged_runs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logging.info("Merged worker manifest written to %s", merged_path)

    logging.info("Parallel run finished. failed_workers=%d", failed)
    return 0 if failed == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging()

    maybe_prepend_vendor_dir(args.vendor_dir)

    if int(args.parallel_cases) > 1 and args.worker_id is None:
        return run_parallel_parent(args)

    model_dir = args.model_dir.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.worker_id is None:
        manifest_path = output_dir / "manifest.json"
    else:
        manifest_path = output_dir / f"manifest_worker{args.worker_id}.json"
        logging.info("Worker mode: worker_id=%s num_workers=%s manifest=%s", args.worker_id, args.num_workers, manifest_path)

    logging.info("Model dir: %s", model_dir)
    logging.info("Dataset dir: %s", dataset_dir)
    logging.info("Output dir: %s", output_dir)
    logging.info("Default generation length: %d frames at %d fps", args.num_frames, args.fps)

    if not (model_dir / "model_index.json").exists():
        raise FileNotFoundError(
            f"This Diffusers runner expects model_index.json under model_dir, but not found: {model_dir / 'model_index.json'}"
        )

    print_runtime_versions()

    all_cases = collect_cases(dataset_dir, only_series=args.only_series, limit=args.limit)
    indexed_cases = list(enumerate(all_cases, start=1))
    if args.worker_id is not None:
        indexed_cases = [(i, c) for i, c in indexed_cases if (i - 1) % int(args.num_workers) == int(args.worker_id)]

    logging.info("Collected %d cases for this process", len(indexed_cases))
    if not indexed_cases:
        logging.warning("No valid cases assigned to this process")
        return 0

    negative_prompt = load_negative_prompt(args, model_dir)
    logging.info("Negative prompt length: %d chars", len(negative_prompt))

    append_manifest(
        manifest_path,
        {
            "status": "run_started",
            "created_at_utc": utc_now(),
            "worker_id": args.worker_id,
            "num_workers": args.num_workers,
            **base_metadata(MODEL_NAME, model_dir, dataset_dir, output_dir, args, BACKEND),
        },
    )

    pipe = None
    success = 0
    failed = 0

    if not args.dry_run:
        pipe = get_pipe(model_dir, args)

    for i, case in indexed_cases:
        try:
            run_record = run_one_case(
                case,
                i,
                pipe,
                negative_prompt,
                model_dir,
                dataset_dir,
                output_dir,
                manifest_path,
                args,
            )
            run_record["worker_id"] = args.worker_id
            if run_record.get("status") in {"success", "skipped_existing", "dry_run"}:
                success += 1
        except Exception as exc:
            failed += 1
            logging.exception("Failed case %s", case.case_id)
            prompt = ""
            try:
                prompt = read_prompt(case)
            except Exception:
                pass
            video_path = case_video_path(output_dir, case)
            frames_dir = case_frames_dir(output_dir, case) if args.save_frames else None
            append_manifest(
                manifest_path,
                {
                    **case_metadata(case, prompt, video_path, frames_dir),
                    "status": "failed",
                    "worker_id": args.worker_id,
                    "created_at_utc": utc_now(),
                    **record_failure(exc),
                },
            )

    append_manifest(
        manifest_path,
        {
            "status": "run_finished",
            "created_at_utc": utc_now(),
            "worker_id": args.worker_id,
            "num_workers": args.num_workers,
            "total_cases_this_process": len(indexed_cases),
            "success_or_skipped": success,
            "failed": failed,
        },
    )

    logging.info("Done. success/skipped=%d failed=%d manifest=%s", success, failed, manifest_path)
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
