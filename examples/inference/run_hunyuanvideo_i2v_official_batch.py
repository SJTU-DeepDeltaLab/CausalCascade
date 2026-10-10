#!/usr/bin/env python3
"""Batch runner for official Tencent HunyuanVideo-I2V inference.

Expected dataset layout:
  Dataset/<series>/<scene>/first_frame.png
  Dataset/<series>/<scene>/prompt.txt

This wrapper calls the official `sample_image2video.py` once per case. For multi-GPU,
it uses xDiT/USP through torchrun, so multiple GPUs cooperate on one video at a time.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import (
    append_manifest,
    base_metadata,
    case_frames_dir,
    case_metadata,
    case_video_path,
    collect_cases,
    export_frames_from_video,
    read_prompt,
    record_failure,
    stable_case_seed,
    setup_logging,
    utc_now,
)


MODEL_NAME = "HunyuanVideo-I2V"


def select_seed(strategy: str, base_seed: int, case_id: str, idx_1based: int) -> int:
    if strategy == "case_hash":
        return stable_case_seed(base_seed, case_id)
    if strategy == "same":
        return int(base_seed)
    if strategy == "sequential":
        return int(base_seed) + idx_1based - 1
    if strategy == "random":
        return random.randint(0, 999_999_999)
    raise ValueError(f"Unknown seed strategy: {strategy}")


def parse_gpu_list(gpus: str) -> List[str]:
    out = [x.strip() for x in gpus.split(",") if x.strip()]
    if not out:
        raise ValueError("--gpus must contain at least one GPU id, e.g. 0 or 0,1,2,3")
    return out


def default_video_size(i2v_resolution: str) -> List[int]:
    r = i2v_resolution.lower()
    if r == "720p":
        return [720, 1280]
    if r == "540p":
        return [544, 960]
    if r == "360p":
        return [360, 640]
    return [720, 1280]


def find_i2v_dit_weight(model_base: Path) -> Optional[Path]:
    """Try to find official I2V DiT checkpoint under model_base."""
    candidates = [
        model_base / "hunyuan-video-i2v-720p" / "transformers" / "mp_rank_00_model_states.pt",
        model_base / "ckpts" / "hunyuan-video-i2v-720p" / "transformers" / "mp_rank_00_model_states.pt",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    patterns = [
        "**/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt",
        "**/transformers/mp_rank_00_model_states.pt",
        "**/*model_states.pt",
        "**/pytorch_model_*.pt",
    ]
    for pat in patterns:
        hits = sorted(model_base.glob(pat))
        # Prefer paths that contain i2v.
        hits = sorted(hits, key=lambda p: ("i2v" not in str(p).lower(), len(str(p))))
        for p in hits:
            if p.is_file():
                return p.resolve()
    return None


def newest_mp4(directory: Path) -> Optional[Path]:
    videos = sorted(directory.rglob("*.mp4"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return videos[0] if videos else None


def run_streaming(cmd: List[str], *, cwd: Path, env: Dict[str, str], log_path: Optional[Path]) -> int:
    logging.info("Running command: %s", " ".join(cmd))
    log_f = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = log_path.open("w", encoding="utf-8")
            log_f.write("[COMMAND] " + " ".join(cmd) + "\n")
            log_f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            if log_f is not None:
                log_f.write(line)
                log_f.flush()
        return proc.wait()
    finally:
        if log_f is not None:
            log_f.close()


def build_official_cmd(
    *,
    python_exe: str,
    code_dir: Path,
    model_base: Path,
    image_path: Path,
    prompt: str,
    save_path: Path,
    seed: int,
    args: argparse.Namespace,
) -> List[str]:
    gpu_ids = parse_gpu_list(args.gpus)
    nproc = int(args.nproc_per_node or len(gpu_ids))

    if nproc > 1:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc}",
            "sample_image2video.py",
        ]
    else:
        cmd = [python_exe, "-u", "sample_image2video.py"]

    cmd += [
        "--model", args.model,
        "--prompt", prompt,
        "--i2v-mode",
        "--i2v-image-path", str(image_path.resolve()),
        "--i2v-resolution", args.i2v_resolution,
        "--infer-steps", str(args.infer_steps),
        "--video-length", str(args.video_length),
        "--flow-shift", str(args.flow_shift),
        "--seed", str(seed),
        "--embedded-cfg-scale", str(args.embedded_cfg_scale),
        "--save-path", str(save_path.resolve()),
        "--model-base", str(model_base.resolve()),
        "--video-size", str(args.video_size[0]), str(args.video_size[1]),
    ]

    if args.flow_reverse:
        cmd.append("--flow-reverse")
    if args.i2v_stability:
        cmd.append("--i2v-stability")
    if args.use_cpu_offload:
        cmd.append("--use-cpu-offload")
    if args.reproduce:
        cmd.append("--reproduce")
    if args.use_fp8:
        cmd.append("--use-fp8")
    if args.vae_tiling:
        cmd.append("--vae-tiling")
    if args.neg_prompt is not None:
        cmd += ["--neg-prompt", args.neg_prompt]
    if args.cfg_scale is not None:
        cmd += ["--cfg-scale", str(args.cfg_scale)]
    if args.precision:
        cmd += ["--precision", args.precision]
    if args.text_encoder_precision:
        cmd += ["--text-encoder-precision", args.text_encoder_precision]
    if args.text_encoder_precision_2:
        cmd += ["--text-encoder-precision-2", args.text_encoder_precision_2]
    if args.vae_precision:
        cmd += ["--vae-precision", args.vae_precision]

    if args.i2v_dit_weight:
        cmd += ["--i2v-dit-weight", str(Path(args.i2v_dit_weight).resolve())]
    elif args.auto_i2v_dit_weight:
        found = find_i2v_dit_weight(model_base)
        if found is not None:
            cmd += ["--i2v-dit-weight", str(found)]

    # xDiT / USP. Product of ulysses and ring should equal world size.
    if nproc > 1:
        cmd += [
            "--ulysses-degree", str(args.ulysses_degree or nproc),
            "--ring-degree", str(args.ring_degree),
        ]

    if args.extra_args:
        cmd += args.extra_args

    return cmd


def run_case(idx: int, total: int, case: Any, args: argparse.Namespace, meta_base: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    code_dir = Path(args.code_dir).resolve()
    model_base = Path(args.model_dir).resolve()
    video_path = case_video_path(output_dir, case)
    frames_dir = case_frames_dir(output_dir, case) if args.save_frames else None

    prompt = read_prompt(case)
    if args.prompt_prefix:
        prompt = args.prompt_prefix.rstrip() + " " + prompt
    if args.prompt_suffix:
        prompt = prompt + " " + args.prompt_suffix.lstrip()

    seed = select_seed(args.seed_strategy, args.seed, case.case_id, idx)

    run_record: Dict[str, Any] = {
        **meta_base,
        **case_metadata(case, prompt, video_path, frames_dir),
        "started_at_utc": utc_now(),
        "case_index": idx,
        "case_total": total,
        "seed": seed,
        "seed_strategy": args.seed_strategy,
        "status": "started",
    }

    if video_path.exists() and not args.overwrite:
        logging.info("[%d/%d] Skip existing: %s", idx, total, video_path)
        run_record["status"] = "skipped_existing"
        run_record["finished_at_utc"] = utc_now()
        return run_record

    tmp_root = output_dir / "_hunyuan_tmp" / f"{idx:04d}_{case.case_id.replace('/', '_').replace('.', '_')}"
    if tmp_root.exists() and args.overwrite:
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "logs" / f"{idx:04d}_{case.case_id.replace('/', '_').replace('.', '_')}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["MODEL_BASE"] = str(model_base)
    # Official xDiT README recommends this for sequence parallel resizing.
    if len(parse_gpu_list(args.gpus)) > 1 or int(args.nproc_per_node or 1) > 1:
        env.setdefault("ALLOW_RESIZE_FOR_SP", "1")

    cmd = build_official_cmd(
        python_exe=sys.executable,
        code_dir=Path(args.code_dir),
        model_base=model_base,
        image_path=case.image_path,
        prompt=prompt,
        save_path=tmp_root,
        seed=seed,
        args=args,
    )
    run_record["official_command"] = cmd
    run_record["tmp_save_path"] = str(tmp_root)
    run_record["gpus"] = args.gpus

    logging.info("[%d/%d] %s seed=%s", idx, total, case.case_id, seed)
    rc = run_streaming(cmd, cwd=code_dir, env=env, log_path=log_path if args.log_per_case else None)
    run_record["returncode"] = rc
    run_record["log_path"] = str(log_path) if args.log_per_case else None

    if rc != 0:
        run_record["status"] = "failed"
        run_record["error"] = f"official sample_image2video.py failed with returncode={rc}"
        run_record["finished_at_utc"] = utc_now()
        return run_record

    generated = newest_mp4(tmp_root)
    if generated is None:
        run_record["status"] = "failed"
        run_record["error"] = f"No mp4 found under {tmp_root}"
        run_record["finished_at_utc"] = utc_now()
        return run_record

    output_dir.mkdir(parents=True, exist_ok=True)
    if video_path.exists() and args.overwrite:
        video_path.unlink()
    shutil.copy2(generated, video_path)
    run_record["raw_generated_video"] = str(generated.resolve())

    if frames_dir is not None:
        try:
            exported = export_frames_from_video(video_path, frames_dir, fps=args.fps)
            run_record["exported_frames"] = exported
        except Exception as exc:
            run_record["frame_export_error"] = record_failure(exc)

    run_record["status"] = "success"
    run_record["finished_at_utc"] = utc_now()
    logging.info("[%d/%d] Saved: %s", idx, total, video_path)
    return run_record


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch benchmark runner for official HunyuanVideo-I2V")
    p.add_argument("--dataset_dir", type=Path, required=True)
    p.add_argument("--model_dir", type=Path, required=True, help="HunyuanVideo-I2V ckpts/model-base directory")
    p.add_argument("--code_dir", type=Path, required=True, help="Tencent-Hunyuan/HunyuanVideo-I2V code directory")
    p.add_argument("--output_dir", type=Path, required=True)

    p.add_argument("--only_series", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_frames", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seed_strategy", choices=["case_hash", "same", "sequential", "random"], default="case_hash")

    p.add_argument("--gpus", type=str, default="0,1,2,3")
    p.add_argument("--nproc_per_node", type=int, default=None, help="Default: number of GPUs in --gpus")
    p.add_argument("--ulysses_degree", type=int, default=None, help="Default: nproc_per_node")
    p.add_argument("--ring_degree", type=int, default=1)

    p.add_argument("--model", type=str, default="HYVideo-T/2")
    p.add_argument("--i2v_resolution", choices=["720p", "540p", "360p"], default="720p")
    p.add_argument("--video_size", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"))
    p.add_argument("--video_length", type=int, default=129)
    p.add_argument("--fps", type=int, default=24, help="Official sample_image2video.py saves at 24 fps; this is recorded in metadata.")
    p.add_argument("--infer_steps", type=int, default=50)
    p.add_argument("--embedded_cfg_scale", type=float, default=6.0)
    p.add_argument("--cfg_scale", type=float, default=None)
    p.add_argument("--flow_shift", type=float, default=7.0)
    p.add_argument("--flow_reverse", action="store_true", default=True)
    p.add_argument("--no_flow_reverse", dest="flow_reverse", action="store_false")
    p.add_argument("--i2v_stability", action="store_true", default=True)
    p.add_argument("--dynamic", dest="i2v_stability", action="store_false", help="Disable --i2v-stability; consider --flow_shift 17.0")
    p.add_argument("--use_cpu_offload", action="store_true", default=False)
    p.add_argument("--no_cpu_offload", dest="use_cpu_offload", action="store_false")
    p.add_argument("--reproduce", action="store_true", default=True)
    p.add_argument("--no_reproduce", dest="reproduce", action="store_false")
    p.add_argument("--use_fp8", action="store_true")
    p.add_argument("--vae_tiling", action="store_true", default=True)
    p.add_argument("--no_vae_tiling", dest="vae_tiling", action="store_false")

    p.add_argument("--precision", type=str, default="bf16")
    p.add_argument("--text_encoder_precision", type=str, default="fp16")
    p.add_argument("--text_encoder_precision_2", type=str, default="fp16")
    p.add_argument("--vae_precision", type=str, default="fp16")
    p.add_argument("--neg_prompt", type=str, default=None)
    p.add_argument("--i2v_dit_weight", type=str, default=None)
    p.add_argument("--auto_i2v_dit_weight", action="store_true", default=True)
    p.add_argument("--no_auto_i2v_dit_weight", dest="auto_i2v_dit_weight", action="store_false")

    p.add_argument("--prompt_prefix", type=str, default="")
    p.add_argument("--prompt_suffix", type=str, default="")
    p.add_argument("--log_per_case", action="store_true", default=True)
    p.add_argument("--no_log_per_case", dest="log_per_case", action="store_false")
    p.add_argument("--extra_args", nargs=argparse.REMAINDER, default=None, help="Arguments after this are passed to sample_image2video.py")
    p.add_argument("--stop_on_failure", action="store_true")
    return p


def main() -> int:
    parser = make_argparser()
    args = parser.parse_args()
    setup_logging()

    if args.video_size is None:
        args.video_size = default_video_size(args.i2v_resolution)

    dataset_dir = Path(args.dataset_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    code_dir = Path(args.code_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (code_dir / "sample_image2video.py").exists():
        raise FileNotFoundError(f"sample_image2video.py not found under --code_dir: {code_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"--model_dir does not exist: {model_dir}")

    cases = collect_cases(dataset_dir, only_series=args.only_series, limit=args.limit)
    logging.info("Collected %d cases", len(cases))
    logging.info("Model base: %s", model_dir)
    logging.info("Code dir: %s", code_dir)
    logging.info("Output dir: %s", output_dir)
    logging.info("Generation: %s, video_size=%s, video_length=%s, official fps=%s", args.i2v_resolution, args.video_size, args.video_length, args.fps)
    logging.info("GPUs: %s; nproc_per_node=%s", args.gpus, args.nproc_per_node or len(parse_gpu_list(args.gpus)))

    found_weight = find_i2v_dit_weight(model_dir)
    if args.auto_i2v_dit_weight:
        logging.info("Auto i2v dit weight: %s", found_weight if found_weight else "not found; relying on official defaults")

    meta_base = base_metadata(
        model_name=MODEL_NAME,
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        args=args,
        backend="official_sample_image2video_xdit_usp",
    )

    manifest_path = output_dir / "manifest.json"
    failures = 0
    for idx, case in enumerate(cases, start=1):
        try:
            record = run_case(idx, len(cases), case, args, meta_base)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logging.exception("Case failed before/after subprocess: %s", case.case_id)
            prompt = read_prompt(case)
            record = {
                **meta_base,
                **case_metadata(case, prompt, case_video_path(output_dir, case), None),
                "case_index": idx,
                "case_total": len(cases),
                "status": "failed_exception",
                "failure": record_failure(exc),
                "finished_at_utc": utc_now(),
            }
        append_manifest(manifest_path, record)
        if not str(record.get("status", "")).startswith("success") and record.get("status") != "skipped_existing":
            failures += 1
            if args.stop_on_failure:
                break

    logging.info("Done. failures=%d manifest=%s", failures, manifest_path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
