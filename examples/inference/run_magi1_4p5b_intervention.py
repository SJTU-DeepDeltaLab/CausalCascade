import argparse
import json
import logging
import os
import random
import shlex
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


MODEL_NAME = "MAGI-1-4.5B"


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    root = default_project_root()

    p = argparse.ArgumentParser()

    p.add_argument("--dataset_dir", type=Path, default=root / "data" / "inputs" / "L1")
    p.add_argument("--model_dir", type=Path, default=root / "MAGI-1")
    p.add_argument(
        "--code_dir",
        type=Path,
        default=root / "ModelCode" / "MAGI-1",
        help="Clone of the official MAGI-1 repository.",
    )
    p.add_argument(
        "--base_config",
        type=Path,
        default=None,
        help="Official MAGI-1 4.5B config. Defaults to code_dir/example/4.5B/4.5B_base_config.json.",
    )
    p.add_argument("--output_dir", type=Path, default=root / "MAGI-1" / "output")

    p.add_argument("--mode", type=str, default="i2v", choices=("t2v", "i2v", "v2v"))

    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--frame_num", type=int, default=96)
    p.add_argument("--fps", type=int, default=24)

    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--window_size", type=int, default=None)

    p.add_argument("--load_path", type=Path, default=None, help="Override runtime_config.load")
    p.add_argument("--t5_pretrained", type=Path, default=None, help="Override runtime_config.t5_pretrained")
    p.add_argument("--vae_pretrained", type=Path, default=None, help="Override runtime_config.vae_pretrained")

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
        help="Comma-separated GPU ids, e.g. 0,1,2,3.",
    )
    p.add_argument(
        "--parallel_cases",
        type=int,
        default=None,
        help="Number of concurrent cases. Defaults to number of GPUs.",
    )

    # MAGI official 4.5B still initializes torch.distributed even for single-GPU inference.
    p.add_argument(
        "--master_addr",
        type=str,
        default="127.0.0.1",
        help="MASTER_ADDR for per-case single-process distributed jobs.",
    )
    p.add_argument(
        "--master_port_base",
        type=int,
        default=6009,
        help=(
            "Base MASTER_PORT. Each case uses master_port_base + case_index "
            "to avoid port collisions in case-parallel mode."
        ),
    )

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only_series", nargs="*", default=None)
    p.add_argument("--save_frames", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def write_case_config(args, case, seed: int, config_out: Path) -> Dict:
    base_config = args.base_config or (args.code_dir / "example" / "4.5B" / "4.5B_base_config.json")
    if not base_config.exists():
        raise FileNotFoundError(f"MAGI-1 4.5B config not found: {base_config}")

    cfg = load_json(base_config)

    runtime = cfg.setdefault("runtime_config", {})
    runtime["seed"] = int(seed)
    runtime["num_frames"] = int(args.frame_num)
    runtime["fps"] = int(args.fps)
    runtime["video_size_h"] = int(args.height)
    runtime["video_size_w"] = int(args.width)

    if args.num_steps is not None:
        runtime["num_steps"] = int(args.num_steps)

    if args.window_size is not None:
        runtime["window_size"] = int(args.window_size)

    if args.load_path is not None:
        runtime["load"] = str(args.load_path.resolve())

    if args.t5_pretrained is not None:
        runtime["t5_pretrained"] = str(args.t5_pretrained.resolve())

    if args.vae_pretrained is not None:
        runtime["vae_pretrained"] = str(args.vae_pretrained.resolve())

    # 4.5B case-parallel: each subprocess is single-GPU.
    engine = cfg.setdefault("engine_config", {})
    engine["pp_size"] = 1
    engine["cp_size"] = 1

    write_json(config_out, cfg)
    return cfg


def build_command(args, case, prompt: str, case_config: Path, output_path: Path) -> List[str]:
    entry = args.code_dir / "inference" / "pipeline" / "entry.py"
    if not entry.exists():
        raise FileNotFoundError(f"MAGI-1 official entry.py not found: {entry}")

    cmd = [
        sys.executable,
        "-u",
        str(entry),
        "--config_file",
        str(case_config.resolve()),
        "--mode",
        args.mode,
        "--prompt",
        prompt,
        "--output_path",
        str(output_path.resolve()),
    ]

    if args.mode == "i2v":
        cmd.extend(["--image_path", str(case.image_path.resolve())])

    return cmd


def run_subprocess_with_env(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_prefix: str,
) -> subprocess.CompletedProcess:
    logging.info("Running command: %s", " ".join(shlex.quote(str(x)) for x in cmd))

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
            logging.info("%s %s", log_prefix, line.rstrip())

    returncode = proc.wait()
    stdout = "".join(output_lines)

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout=stdout,
        stderr=None,
    )


def build_env(args, seed: int, gpu: str, case_index: int) -> Dict[str, str]:
    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(args.code_dir.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    # MAGI-1 official 4.5B entry.py still initializes torch.distributed.
    # Each case subprocess is a single-process distributed job.
    env["MASTER_ADDR"] = args.master_addr
    env["MASTER_PORT"] = str(args.master_port_base + int(case_index))
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    env["LOCAL_WORLD_SIZE"] = "1"
    env["GPUS_PER_NODE"] = "1"
    env["NNODES"] = "1"

    # Same style as official 4.5B launch environment.
    env["PAD_HQ"] = env.get("PAD_HQ", "1")
    env["PAD_DURATION"] = env.get("PAD_DURATION", "1")
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    env["OFFLOAD_T5_CACHE"] = env.get("OFFLOAD_T5_CACHE", "true")
    env["OFFLOAD_VAE_CACHE"] = env.get("OFFLOAD_VAE_CACHE", "true")

    # Official MAGI should use runtime_config.seed.
    # These env vars are extra reproducibility hints for dependencies.
    env["PYTHONHASHSEED"] = str(seed)
    env["MAGI_SEED"] = str(seed)
    env["SEED"] = str(seed)

    return env


def run_one_case(args, case, idx: int, total: int, seed: int, gpu: str) -> Dict:
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
            "duration_seconds": args.frame_num / args.fps if args.fps else None,
            "master_addr": args.master_addr,
            "master_port": args.master_port_base + int(idx),
        }
    )

    if video_path.exists() and not args.overwrite:
        elapsed = time.time() - start_time
        logging.info(
            "[%d/%d] skip existing %s seed=%s gpu=%s elapsed=%.1fs output=%s",
            idx,
            total,
            case.case_id,
            seed,
            gpu,
            elapsed,
            video_path,
        )
        meta["status"] = "skipped_existing"
        meta["elapsed_seconds"] = elapsed
        return meta

    try:
        case_safe = safe_name(case.case_id)

        prompt_file = args.output_dir / "prompts" / f"{case_safe}.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(prompt, encoding="utf-8")

        case_config = args.output_dir / "configs" / f"{case_safe}.seed{seed}.json"
        write_case_config(args, case, seed, case_config)

        cmd = build_command(args, case, prompt, case_config, video_path)
        env = build_env(args, seed, gpu, idx)

        logging.info(
            "[%d/%d] generating %s seed=%s gpu=%s size=%sx%s frames=%s fps=%s master_port=%s",
            idx,
            total,
            case.case_id,
            seed,
            gpu,
            args.width,
            args.height,
            args.frame_num,
            args.fps,
            env["MASTER_PORT"],
        )

        # Important: use code_dir as cwd so official config relative paths like ./downloads/... work.
        proc = run_subprocess_with_env(
            cmd,
            cwd=args.code_dir,
            env=env,
            log_prefix=f"[MAGI case={idx}/{total} gpu={gpu}]",
        )

        meta["case_config"] = str(case_config.resolve())
        meta["prompt_file"] = str(prompt_file.resolve())
        meta["subprocess_returncode"] = proc.returncode
        meta["subprocess_output_tail"] = proc.stdout[-4000:] if proc.stdout else ""

        if proc.returncode != 0:
            logging.error(
                "MAGI-1 subprocess failed for %s. Output tail:\n%s",
                case.case_id,
                proc.stdout[-4000:] if proc.stdout else "",
            )
            raise RuntimeError(f"MAGI-1 subprocess failed with returncode={proc.returncode}")

        if not video_path.exists():
            raise FileNotFoundError(
                f"Expected output video not found: {video_path}. "
                "MAGI entry.py should honor --output_path; check subprocess_output_tail in manifest."
            )

        if frames_dir:
            meta["exported_frame_count"] = export_frames_from_video(video_path, frames_dir, args.fps)

        elapsed = time.time() - start_time

        meta.update(
            {
                "status": "ok",
                "elapsed_seconds": elapsed,
                "completed_at_utc": utc_now(),
            }
        )

        logging.info(
            "[%d/%d] completed %s seed=%s gpu=%s elapsed=%.1fs output=%s",
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


def main():
    setup_logging()
    args = parse_args()

    if not args.code_dir.exists():
        raise FileNotFoundError(f"MAGI-1 code_dir does not exist: {args.code_dir}")

    if args.master_port_base <= 0:
        raise ValueError("--master_port_base must be positive")

    gpu_ids = parse_gpu_list(args.gpus)
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id, e.g. --gpus 0 or --gpus 0,1,2,3")

    workers = args.parallel_cases or len(gpu_ids)
    if workers < 1:
        raise ValueError("--parallel_cases must be >= 1")
    if workers > len(gpu_ids):
        raise ValueError("--parallel_cases cannot exceed number of GPUs in --gpus")

    cases = collect_cases(args.dataset_dir, args.only_series, args.limit)
    manifest_path = args.output_dir / "manifest.json"
    rng = random.Random(args.seed)

    logging.info("Found %d cases", len(cases))
    logging.info(
        "Seed strategy: %s, base seed: %s, random_seed: %s",
        args.seed_strategy,
        args.seed,
        args.random_seed,
    )
    logging.info(
        "MAGI-1 4.5B case-parallel: gpus=%s parallel_cases=%s size=%sx%s frame_num=%s fps=%s",
        gpu_ids,
        workers,
        args.width,
        args.height,
        args.frame_num,
        args.fps,
    )
    logging.info(
        "Distributed env per subprocess: MASTER_ADDR=%s MASTER_PORT=master_port_base+case_index, master_port_base=%s",
        args.master_addr,
        args.master_port_base,
    )

    if args.dry_run:
        for idx, case in enumerate(cases, start=1):
            seed = choose_seed(args, case, idx, rng)
            logging.info(
                "%s seed=%s gpu=%s master_port=%s -> %s",
                case.case_id,
                seed,
                gpu_ids[(idx - 1) % len(gpu_ids)],
                args.master_port_base + idx,
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
            "official MAGI-1 4.5B wrapper with per-case configuration and case-parallel GPUs",
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
            "duration_seconds": args.frame_num / args.fps if args.fps else None,
            "mode": args.mode,
        },
        "multi_gpu": {
            "mode": "case_parallel",
            "gpus": gpu_ids,
            "parallel_cases": workers,
            "master_addr": args.master_addr,
            "master_port_base": args.master_port_base,
            "world_size_per_subprocess": 1,
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
                run_one_case,
                args,
                case,
                idx,
                len(cases),
                seed,
                gpu,
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

    logging.info(
        "Batch complete: %s elapsed=%.1fs",
        args.output_dir,
        run_record["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
