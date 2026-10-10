import argparse
import logging
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    safe_name,
    read_prompt,
    record_failure,
    choose_seed,
    setup_logging,
    utc_now,
)


MODEL_NAME = "LTX-Video-13B"


def parse_args():
    root = default_project_root()

    p = argparse.ArgumentParser()

    p.add_argument("--dataset_dir", type=Path, default=root / "data" / "inputs" / "L1")
    p.add_argument("--model_dir", type=Path, default=root / MODEL_NAME)

    p.add_argument(
        "--code_dir",
        type=Path,
        default=root / "ModelCode" / "LTX-Video",
        help="Official LTX-Video repository.",
    )

    p.add_argument("--backend", choices=("official",), default="official")
    p.add_argument("--pipeline_config", type=Path, default=None)

    p.add_argument(
        "--text_encoder_model",
        type=Path,
        default=None,
        help=(
            "Local PixArt model root containing text_encoder/ and tokenizer/. "
            "Example: /path/to/PixArt-XL-2-1024-MS"
        ),
    )

    p.add_argument("--output_dir", type=Path, default=root / MODEL_NAME / "output")

    p.add_argument("--frame_num", type=int, default=151)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--fps", type=int, default=30)

    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument(
        "--negative_prompt",
        type=str,
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
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

    p.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated physical GPU ids, e.g. 0,1,2,3. Repeated ids are allowed.",
    )
    p.add_argument(
        "--parallel_cases",
        type=int,
        default=None,
        help="Number of concurrent official LTX subprocesses. Defaults to number of entries in --gpus.",
    )

    p.add_argument(
        "--stream_subprocess",
        action="store_true",
        help="Stream official LTX subprocess output in real time. Useful for debugging, noisy for 4-way parallel.",
    )

    p.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 in subprocesses.",
    )

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only_series", nargs="*", default=None)
    p.add_argument("--save_frames", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def newest_mp4(root: Path, before: set) -> Path:
    candidates = [p for p in root.rglob("*.mp4") if p not in before]
    if not candidates:
        raise FileNotFoundError("LTX official inference finished but no new mp4 was found")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def local_checkpoint(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("*.safetensors"))
    candidates = [p for p in candidates if "upscaler" not in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(f"No LTX safetensors checkpoint found in {model_dir}")
    return candidates[0]


def local_upscaler(model_dir: Path):
    candidates = sorted(model_dir.glob("*upscaler*.safetensors"))
    return candidates[0] if candidates else None


def resolve_text_encoder_model(args) -> Optional[Path]:
    """
    Return the local PixArt root directory if available.

    The official LTX code expects the root path and then loads:
    - <root>/text_encoder/config.json
    - <root>/tokenizer/...
    Therefore do NOT pass the text_encoder subdirectory itself.
    """
    if args.text_encoder_model is not None:
        root = args.text_encoder_model.resolve()
        if not (root / "text_encoder" / "config.json").exists():
            raise FileNotFoundError(
                f"--text_encoder_model should point to the PixArt root containing "
                f"text_encoder/config.json, but not found: {root / 'text_encoder' / 'config.json'}"
            )
        if not (root / "tokenizer").exists():
            raise FileNotFoundError(
                f"--text_encoder_model should point to the PixArt root containing tokenizer/, "
                f"but not found: {root / 'tokenizer'}"
            )
        return root

    local_pixart = default_project_root() / "PixArt-XL-2-1024-MS"
    if (local_pixart / "text_encoder" / "config.json").exists() and (local_pixart / "tokenizer").exists():
        return local_pixart.resolve()

    return None


def make_ltx_config(args: argparse.Namespace) -> Path:
    source = args.pipeline_config or (args.code_dir / "configs" / "ltxv-13b-0.9.8-dev.yaml")
    if not source.exists():
        raise FileNotFoundError(f"LTX pipeline config not found: {source}")

    import yaml

    data = yaml.safe_load(source.read_text(encoding="utf-8"))

    # Use local LTX-13B safetensors checkpoint.
    data["checkpoint_path"] = str(local_checkpoint(args.model_dir).resolve())

    # Use local PixArt text encoder/tokenizer if available.
    text_encoder_root = resolve_text_encoder_model(args)
    if text_encoder_root is not None:
        data["text_encoder_model_name_or_path"] = str(text_encoder_root)
        logging.info("Using local PixArt text encoder root: %s", text_encoder_root)
    else:
        logging.warning(
            "No local PixArt text encoder found. LTX may try to download %s from Hugging Face.",
            data.get("text_encoder_model_name_or_path"),
        )

    upscaler = local_upscaler(args.model_dir)
    if upscaler and "spatial_upscaler_model_path" in data:
        data["spatial_upscaler_model_path"] = str(upscaler.resolve())

    out = args.output_dir / "_runtime_ltx_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    logging.info("Runtime LTX config written to: %s", out)
    logging.info("checkpoint_path: %s", data.get("checkpoint_path"))
    logging.info("text_encoder_model_name_or_path: %s", data.get("text_encoder_model_name_or_path"))
    if "spatial_upscaler_model_path" in data:
        logging.info("spatial_upscaler_model_path: %s", data.get("spatial_upscaler_model_path"))

    return out


def validate_args(args):
    if args.height % 32 != 0 or args.width % 32 != 0:
        logging.warning(
            "LTX usually expects height and width divisible by 32, got height=%s width=%s",
            args.height,
            args.width,
        )

    if (args.frame_num - 1) % 8 != 0:
        logging.warning(
            "LTX usually expects frame_num = 8n + 1, got frame_num=%s.",
            args.frame_num,
        )

    if args.fps <= 0:
        raise ValueError("--fps must be positive")


def make_subprocess_env(args, gpu: str) -> Dict[str, str]:
    env = os.environ.copy()

    # Each subprocess sees its assigned GPU as cuda:0.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"

    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    return env


def run_subprocess_with_env(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    prefix: str,
    stream: bool,
) -> subprocess.CompletedProcess:
    logging.info("%s Running command: %s", prefix, " ".join(shlex.quote(str(x)) for x in cmd))

    if not stream:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    output_lines = []
    if proc.stdout is not None:
        for line in proc.stdout:
            output_lines.append(line)
            logging.info("%s %s", prefix, line.rstrip())

    returncode = proc.wait()
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout="".join(output_lines),
        stderr=None,
    )


def run_one_case_official(
    args,
    case,
    idx: int,
    total: int,
    seed: int,
    gpu: str,
    runtime_config: Path,
    entry: Path,
) -> Dict:
    start_time = time.time()

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
            "gpu": gpu,
            "height": args.height,
            "width": args.width,
            "frame_num": args.frame_num,
            "fps": args.fps,
            "encoded_duration_seconds": args.frame_num / args.fps if args.fps else None,
            "frame_interval_duration_seconds": (
                (args.frame_num - 1) / args.fps if args.fps and args.frame_num > 0 else None
            ),
        }
    )

    if video_path.exists() and not args.overwrite:
        elapsed = time.time() - start_time
        logging.info(
            "[%d/%d] skip existing %s seed=%s gpu=%s elapsed=%.1fs",
            idx,
            total,
            case.case_id,
            seed,
            gpu,
            elapsed,
        )
        meta["status"] = "skipped_existing"
        meta["elapsed_seconds"] = elapsed
        return meta

    try:
        case_safe = safe_name(case.case_id)

        # Isolate each case's output before selecting its generated MP4.
        tmp_out = args.output_dir / "_ltx_tmp_outputs" / f"{idx:04d}_{case_safe}_seed{seed}"
        if tmp_out.exists():
            shutil.rmtree(tmp_out)
        tmp_out.mkdir(parents=True, exist_ok=True)

        before = set(tmp_out.rglob("*.mp4"))

        cmd = [
            sys.executable,
            "-u",
            str(entry.resolve()),
            "--prompt",
            prompt,
            "--conditioning_media_paths",
            str(case.image_path.resolve()),
            "--conditioning_start_frames",
            "0",
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--num_frames",
            str(args.frame_num),
            "--frame_rate",
            str(args.fps),
            "--seed",
            str(seed),
            "--pipeline_config",
            str(runtime_config.resolve()),
            "--output_path",
            str(tmp_out.resolve()),
            "--negative_prompt",
            args.negative_prompt,
        ]

        prefix = f"[LTX case={idx}/{total} gpu={gpu}]"
        logging.info(
            "[%d/%d] generating %s seed=%s gpu=%s size=%sx%s frames=%s fps=%s",
            idx,
            total,
            case.case_id,
            seed,
            gpu,
            args.width,
            args.height,
            args.frame_num,
            args.fps,
        )

        env = make_subprocess_env(args, gpu)
        proc = run_subprocess_with_env(
            cmd,
            cwd=Path.cwd(),
            env=env,
            prefix=prefix,
            stream=args.stream_subprocess,
        )

        meta["subprocess_returncode"] = proc.returncode
        meta["subprocess_output_tail"] = proc.stdout[-8000:] if proc.stdout else ""

        if proc.returncode != 0:
            logging.error(
                "LTX subprocess failed for %s. Output tail:\n%s",
                case.case_id,
                proc.stdout[-8000:] if proc.stdout else "",
            )
            raise RuntimeError(f"LTX subprocess failed with returncode={proc.returncode}")

        produced = newest_mp4(tmp_out, before)

        video_path.parent.mkdir(parents=True, exist_ok=True)
        if video_path.exists() and args.overwrite:
            video_path.unlink()

        shutil.move(str(produced), str(video_path))

        if frames_dir:
            meta["exported_frame_count"] = export_frames_from_video(
                video_path,
                frames_dir,
                args.fps,
            )

        elapsed = time.time() - start_time
        meta.update(
            {
                "status": "ok",
                "elapsed_seconds": elapsed,
                "completed_at_utc": utc_now(),
            }
        )

        logging.info(
            "[%d/%d] completed %s seed=%s gpu=%s elapsed=%.1fs -> %s",
            idx,
            total,
            case.case_id,
            seed,
            gpu,
            elapsed,
            video_path,
        )

    except Exception as exc:
        elapsed = time.time() - start_time
        meta.update(
            {
                "status": "failed",
                "elapsed_seconds": elapsed,
                **record_failure(exc),
                "completed_at_utc": utc_now(),
            }
        )
        logging.exception(
            "[%d/%d] failed %s seed=%s gpu=%s elapsed=%.1fs",
            idx,
            total,
            case.case_id,
            seed,
            gpu,
            elapsed,
        )

    return meta


def run_official_backend(args, cases, manifest_path: Path):
    entry = args.code_dir / "inference.py"
    if not entry.exists():
        raise FileNotFoundError(f"Official LTX inference.py not found: {entry}")

    runtime_config = make_ltx_config(args)

    gpu_ids = parse_gpu_list(args.gpus)
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id, e.g. --gpus 0 or --gpus 0,1,2,3")

    workers = args.parallel_cases or len(gpu_ids)
    if workers < 1:
        raise ValueError("--parallel_cases must be >= 1")
    if workers > len(gpu_ids):
        raise ValueError("--parallel_cases cannot exceed number of entries in --gpus")

    rng = random.Random(args.seed)

    run_record = {
        **base_metadata(
            MODEL_NAME,
            args.model_dir,
            args.dataset_dir,
            args.output_dir,
            args,
            f"official LTX inference.py from {args.code_dir}; case-parallel subprocess runner",
        ),
        "started_at_utc": utc_now(),
        "seed_policy": {
            "base_seed": args.seed,
            "seed_strategy": args.seed_strategy,
            "random_seed": args.random_seed,
        },
        "generation_config": {
            "backend": args.backend,
            "height": args.height,
            "width": args.width,
            "frame_num": args.frame_num,
            "fps": args.fps,
            "encoded_duration_seconds": args.frame_num / args.fps if args.fps else None,
            "frame_interval_duration_seconds": (
                (args.frame_num - 1) / args.fps if args.fps and args.frame_num > 0 else None
            ),
            "num_inference_steps": args.num_inference_steps,
            "negative_prompt": args.negative_prompt,
            "pipeline_config": str(runtime_config.resolve()),
            "text_encoder_model": (
                str(resolve_text_encoder_model(args)) if resolve_text_encoder_model(args) is not None else None
            ),
        },
        "multi_gpu": {
            "mode": "case_parallel_subprocess",
            "gpus": gpu_ids,
            "parallel_cases": workers,
        },
        "cases": [],
    }

    indexed_cases = []
    for idx, case in enumerate(cases, start=1):
        seed = choose_seed(args, case, idx, rng)
        indexed_cases.append((idx, case, seed))

    batch_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_case = {}

        for submit_i, (idx, case, seed) in enumerate(indexed_cases):
            gpu = gpu_ids[submit_i % len(gpu_ids)]
            future = executor.submit(
                run_one_case_official,
                args,
                case,
                idx,
                len(cases),
                seed,
                gpu,
                runtime_config,
                entry,
            )
            future_to_case[future] = case.case_id

        finished = 0
        for future in as_completed(future_to_case):
            meta = future.result()
            run_record["cases"].append(meta)

            finished += 1
            logging.info(
                "Progress: %d/%d cases finished; latest=%s status=%s elapsed_total=%.1fs",
                finished,
                len(cases),
                meta.get("case_id"),
                meta.get("status"),
                time.time() - batch_start,
            )

    run_record["cases"].sort(key=lambda x: x.get("index", 10**9))
    run_record["completed_at_utc"] = utc_now()
    run_record["elapsed_seconds"] = time.time() - batch_start
    append_manifest(manifest_path, run_record)

    logging.info("Batch complete: %s elapsed=%.1fs", args.output_dir, run_record["elapsed_seconds"])


def main():
    setup_logging()
    args = parse_args()
    validate_args(args)

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
        "Config: backend=%s size=%sx%s frame_num=%s fps=%s encoded_duration=%.4fs interval_duration=%.4fs",
        args.backend,
        args.width,
        args.height,
        args.frame_num,
        args.fps,
        args.frame_num / args.fps,
        (args.frame_num - 1) / args.fps if args.frame_num > 0 else 0.0,
    )

    gpu_ids = parse_gpu_list(args.gpus)
    workers = args.parallel_cases or len(gpu_ids) or 1

    logging.info("Case-parallel settings: gpus=%s parallel_cases=%s", gpu_ids, workers)

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

    run_official_backend(args, cases, manifest_path)


if __name__ == "__main__":
    main()
