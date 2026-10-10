import argparse
import logging
import os
import random
import shlex
import shutil
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


MODEL_NAME = "SkyReels-V2-I2V-14B-720P"


def safe_name(value: str) -> str:
    bad = '<>:"/\\|?*\n\r\t'
    return "".join("_" if c in bad else c for c in value)[:180]


def parse_args():
    root = default_project_root()
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_dir", type=Path, default=root / "data" / "inputs" / "L1")

    p.add_argument("--model_dir", type=Path, default=root / MODEL_NAME)
    p.add_argument(
        "--code_dir",
        type=Path,
        default=root / "ModelCode" / "SkyReels-V2",
        help="Clone of https://github.com/SkyworkAI/SkyReels-V2",
    )
    p.add_argument("--output_dir", type=Path, default=root / MODEL_NAME / "output")

    p.add_argument("--resolution", type=str, default="720P", choices=("540P", "720P"))
    p.add_argument("--num_frames", type=int, default=121)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=3.0)

    p.add_argument(
        "--prompt_prefix",
        type=str,
        default="",
        help="Optional text prepended to every case prompt. Empty by default.",
    )
    p.add_argument(
        "--prompt_suffix",
        type=str,
        default="",
        help="Optional text appended to every case prompt. Empty by default.",
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

    # Official performance options.
    p.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--teacache", action="store_true")
    p.add_argument("--teacache_thresh", type=float, default=0.2)
    p.add_argument("--use_ret_steps", action="store_true")

    # Official multi-GPU deployment through xDiT USP.
    p.add_argument(
        "--use_usp",
        action="store_true",
        help="Use official xDiT USP multi-GPU inference. This wrapper launches torch.distributed.run per case.",
    )
    p.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated physical GPU ids. For 4-card deployment, use --gpus 0,1,2,3.",
    )
    p.add_argument(
        "--nproc_per_node",
        type=int,
        default=None,
        help="Number of torchrun processes for --use_usp. Defaults to len(--gpus).",
    )

    p.add_argument(
        "--stream_subprocess",
        action="store_true",
        help="Stream official subprocess output in real time. Useful for debugging, noisy for full runs.",
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


def build_case_prompt(args, case) -> str:
    raw_prompt = read_prompt(case).strip()
    parts = []
    if args.prompt_prefix and args.prompt_prefix.strip():
        parts.append(args.prompt_prefix.strip())
    parts.append(raw_prompt)
    if args.prompt_suffix and args.prompt_suffix.strip():
        parts.append(args.prompt_suffix.strip())
    return "\n\n".join(parts)


def newest_mp4(root: Path, before: set) -> Path:
    candidates = [p for p in root.rglob("*.mp4") if p not in before]
    if not candidates:
        raise FileNotFoundError(f"Official SkyReels-V2 script finished but no new mp4 was found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def make_subprocess_env(args) -> Dict[str, str]:
    env = os.environ.copy()

    gpu_ids = parse_gpu_list(args.gpus)
    if gpu_ids:
        if args.use_usp:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        else:
            # Single-process official inference uses only cuda:0, so expose only the first GPU.
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]

    env["PYTHONUNBUFFERED"] = "1"

    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    return env


def run_subprocess_with_env(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    prefix: str,
    stream: bool,
) -> subprocess.CompletedProcess:
    logging.info("%s Running command: %s", prefix, " ".join(shlex.quote(str(x)) for x in cmd))
    logging.info("%s CUDA_VISIBLE_DEVICES=%s", prefix, env.get("CUDA_VISIBLE_DEVICES"))

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


def build_command(args, entry: Path, case, prompt: str, seed: int, official_outdir: str) -> List[str]:
    base_cmd = [
        str(entry),
        "--model_id",
        str(args.model_dir.resolve()),
        "--resolution",
        args.resolution,
        "--num_frames",
        str(args.num_frames),
        "--guidance_scale",
        str(args.guidance_scale),
        "--shift",
        str(args.shift),
        "--inference_steps",
        str(args.inference_steps),
        "--fps",
        str(args.fps),
        "--image",
        str(case.image_path.resolve()),
        "--prompt",
        prompt,
        "--seed",
        str(seed),
        "--outdir",
        official_outdir,
    ]

    if args.offload:
        base_cmd.append("--offload")

    if args.teacache:
        base_cmd.append("--teacache")
        base_cmd += ["--teacache_thresh", str(args.teacache_thresh)]
        if args.use_ret_steps:
            base_cmd.append("--use_ret_steps")

    if args.use_usp:
        base_cmd.append("--use_usp")

        nproc = args.nproc_per_node
        if nproc is None:
            nproc = len(parse_gpu_list(args.gpus))

        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(nproc),
        ] + base_cmd

    return [sys.executable] + base_cmd


def run_one_case(args, case, idx: int, total: int, seed: int, entry: Path, run_tag: str) -> Dict:
    start_time = time.time()

    video_path = case_video_path(args.output_dir, case)
    frames_dir = case_frames_dir(args.output_dir, case) if args.save_frames else None
    prompt = build_case_prompt(args, case)

    meta = case_metadata(case, prompt, video_path, frames_dir)
    meta.update(
        {
            "index": idx,
            "seed": seed,
            "base_seed": args.seed,
            "seed_strategy": "random_seed" if args.random_seed else args.seed_strategy,
            "resolution": args.resolution,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "expected_duration_seconds_by_frame_interval": (args.num_frames - 1) / args.fps if args.fps else None,
            "encoded_duration_seconds_approx": args.num_frames / args.fps if args.fps else None,
            "inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "shift": args.shift,
            "offload": args.offload,
            "teacache": args.teacache,
            "use_usp": args.use_usp,
            "gpus": args.gpus,
            "nproc_per_node": args.nproc_per_node if args.nproc_per_node is not None else len(parse_gpu_list(args.gpus)),
            "prompt_prefix": args.prompt_prefix,
            "prompt_suffix": args.prompt_suffix,
        }
    )

    if video_path.exists() and not args.overwrite:
        elapsed = time.time() - start_time
        logging.info(
            "[%d/%d] skip existing %s seed=%s elapsed=%.1fs",
            idx,
            total,
            case.case_id,
            seed,
            elapsed,
        )
        meta["status"] = "skipped_existing"
        meta["elapsed_seconds"] = elapsed
        return meta

    try:
        # Official script writes into code_dir/result/{outdir}.
        official_outdir = f"_benchmark_skyreels_v2/{run_tag}/{idx:05d}_{safe_name(case.case_id)}_seed{seed}"
        result_dir = args.code_dir / "result" / official_outdir
        before = set(result_dir.rglob("*.mp4")) if result_dir.exists() else set()

        cmd = build_command(args, entry, case, prompt, seed, official_outdir)
        env = make_subprocess_env(args)
        prefix = f"[SkyReels-V2 case={idx}/{total}]"

        logging.info(
            "[%d/%d] generating %s seed=%s frames=%s fps=%s resolution=%s steps=%s guidance=%s shift=%s use_usp=%s gpus=%s",
            idx,
            total,
            case.case_id,
            seed,
            args.num_frames,
            args.fps,
            args.resolution,
            args.inference_steps,
            args.guidance_scale,
            args.shift,
            args.use_usp,
            args.gpus,
        )

        proc = run_subprocess_with_env(
            cmd,
            cwd=args.code_dir,
            env=env,
            prefix=prefix,
            stream=args.stream_subprocess,
        )

        meta["subprocess_returncode"] = proc.returncode
        meta["subprocess_output_tail"] = proc.stdout[-8000:] if proc.stdout else ""
        meta["official_outdir"] = str(result_dir.resolve())

        if proc.returncode != 0:
            logging.error(
                "SkyReels-V2 subprocess failed for %s. Output tail:\n%s",
                case.case_id,
                proc.stdout[-8000:] if proc.stdout else "",
            )
            raise RuntimeError(f"SkyReels-V2 subprocess failed with returncode={proc.returncode}")

        produced = newest_mp4(result_dir, before)

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
                "output_video": str(video_path.resolve()),
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
            "[%d/%d] failed %s seed=%s elapsed=%.1fs",
            idx,
            total,
            case.case_id,
            seed,
            elapsed,
        )

    return meta


def validate_args(args):
    if args.num_frames <= 0:
        raise ValueError("--num_frames must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    if args.resolution == "720P" and args.num_frames != 121:
        logging.warning("Official recommended 720P setting is --num_frames 121. You set %s.", args.num_frames)
    if args.resolution == "540P" and args.num_frames != 97:
        logging.warning("Official recommended 540P setting is --num_frames 97. You set %s.", args.num_frames)

    if args.resolution == "720P" and "720P" not in args.model_dir.name:
        logging.warning("You set --resolution 720P but model_dir name does not contain 720P: %s", args.model_dir)
    if args.resolution == "540P" and "540P" not in args.model_dir.name:
        logging.warning("You set --resolution 540P but model_dir name does not contain 540P: %s", args.model_dir)

    gpu_ids = parse_gpu_list(args.gpus)
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id, e.g. --gpus 0 or --gpus 0,1,2,3")

    if args.use_usp:
        nproc = args.nproc_per_node if args.nproc_per_node is not None else len(gpu_ids)
        if nproc != len(gpu_ids):
            logging.warning(
                "--nproc_per_node=%s but len(--gpus)=%s. Usually these should match.",
                nproc,
                len(gpu_ids),
            )
        if args.random_seed:
            logging.warning("Official USP mode requires fixed seed. --random_seed is accepted by wrapper but reproducibility is weaker.")


def main():
    setup_logging()
    args = parse_args()
    validate_args(args)

    if not args.code_dir.exists():
        raise FileNotFoundError(f"SkyReels-V2 code_dir does not exist: {args.code_dir}")

    entry = args.code_dir / "generate_video.py"
    if not entry.exists():
        raise FileNotFoundError(f"SkyReels-V2 official generate_video.py not found: {entry}")

    if not args.model_dir.exists():
        raise FileNotFoundError(f"SkyReels-V2 model_dir does not exist: {args.model_dir}")

    cases = collect_cases(args.dataset_dir, args.only_series, args.limit)
    manifest_path = args.output_dir / "manifest.json"
    rng = random.Random(args.seed)
    run_tag = f"run_{int(time.time())}_{safe_name(args.resolution)}_{args.num_frames}f_seed{args.seed}"

    logging.info("Found %d cases", len(cases))
    logging.info(
        "Seed strategy: %s, base seed: %s, random_seed: %s",
        args.seed_strategy,
        args.seed,
        args.random_seed,
    )
    logging.info(
        "Config: resolution=%s num_frames=%s fps=%s interval_duration=%.4fs steps=%s guidance=%s shift=%s use_usp=%s gpus=%s nproc_per_node=%s",
        args.resolution,
        args.num_frames,
        args.fps,
        (args.num_frames - 1) / args.fps if args.fps else -1,
        args.inference_steps,
        args.guidance_scale,
        args.shift,
        args.use_usp,
        args.gpus,
        args.nproc_per_node if args.nproc_per_node is not None else len(parse_gpu_list(args.gpus)),
    )

    if args.dry_run:
        for idx, case in enumerate(cases, start=1):
            seed = choose_seed(args, case, idx, rng)
            logging.info(
                "%s seed=%s -> %s",
                case.case_id,
                seed,
                case_video_path(args.output_dir, case),
            )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_record = {
        **base_metadata(
            MODEL_NAME,
            args.model_dir,
            args.dataset_dir,
            args.output_dir,
            args,
            "official SkyReels-V2 generate_video.py subprocess wrapper with optional torchrun xDiT USP",
        ),
        "started_at_utc": utc_now(),
        "seed_policy": {
            "base_seed": args.seed,
            "seed_strategy": args.seed_strategy,
            "random_seed": args.random_seed,
        },
        "generation_config": {
            "task_type": "image_to_video",
            "resolution": args.resolution,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "expected_duration_seconds_by_frame_interval": (args.num_frames - 1) / args.fps if args.fps else None,
            "encoded_duration_seconds_approx": args.num_frames / args.fps if args.fps else None,
            "inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "shift": args.shift,
            "offload": args.offload,
            "teacache": args.teacache,
            "prompt_prefix": args.prompt_prefix,
            "prompt_suffix": args.prompt_suffix,
        },
        "multi_gpu": {
            "enabled": args.use_usp,
            "mode": "official_xdit_usp_model_parallel_per_case" if args.use_usp else "single_gpu_per_case",
            "gpus": parse_gpu_list(args.gpus),
            "nproc_per_node": args.nproc_per_node if args.nproc_per_node is not None else len(parse_gpu_list(args.gpus)),
        },
        "cases": [],
    }

    batch_start = time.time()

    for idx, case in enumerate(cases, start=1):
        seed = choose_seed(args, case, idx, rng)
        meta = run_one_case(args, case, idx, len(cases), seed, entry, run_tag)
        run_record["cases"].append(meta)

        logging.info(
            "Progress: %d/%d cases finished; latest=%s status=%s elapsed_total=%.1fs",
            idx,
            len(cases),
            meta.get("case_id"),
            meta.get("status"),
            time.time() - batch_start,
        )

    run_record["completed_at_utc"] = utc_now()
    run_record["elapsed_seconds"] = time.time() - batch_start
    append_manifest(manifest_path, run_record)

    logging.info("Batch complete: %s elapsed=%.1fs", args.output_dir, run_record["elapsed_seconds"])


if __name__ == "__main__":
    main()
