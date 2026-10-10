#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

from cf_effect_metrics import MotionConfig, MotionMapCache, score_pair


SCRIPT_ROOT = Path(__file__).resolve().parent
EDIT_TYPES = ("delete_type", "mass_type", "move_type", "scale_type")
PRIMARY_METRICS = (
    "cf_spatial_effect_iou",
    "cf_weighted_effect_iou",
    "cf_preserve_spatial_iou",
    "cf_preserve_weighted_iou",
)
DIAGNOSTIC_METRICS = (
    "suppressed_spatial_iou_diagnostic",
    "induced_spatial_iou_diagnostic",
    "suppressed_weighted_iou_diagnostic",
    "induced_weighted_iou_diagnostic",
)
ALL_SCORE_METRICS = PRIMARY_METRICS + DIAGNOSTIC_METRICS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def slug(text: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in text).strip("_")


def discover_cases(gt_root: Path, gt_cf_root: Path) -> list[dict]:
    cases = []
    for edit_type in EDIT_TYPES:
        type_root = gt_cf_root / edit_type
        for diff_path in sorted(type_root.rglob("diff.json")):
            edit_dir = diff_path.parent
            relative = edit_dir.relative_to(type_root)
            if len(relative.parts) != 3:
                raise RuntimeError(f"Unexpected GT edit path: {edit_dir}")
            series, scene, edit_name = relative.parts
            gt_cf_video = edit_dir / "video.mp4"
            gt_factual_video = gt_root / series / scene / f"{scene}.mp4"
            for required in (gt_cf_video, gt_factual_video):
                if not required.is_file():
                    raise FileNotFoundError(required)
            cases.append(
                {
                    "case_id": f"{edit_type}.{series}.{scene}.{edit_name}",
                    "edit_type": edit_type,
                    "series": series,
                    "scene": scene,
                    "scene_id": f"{series}/{scene}",
                    "edit_name": edit_name,
                    "diff_json": str(diff_path),
                    "gt_factual_video": str(gt_factual_video),
                    "gt_counterfactual_video": str(gt_cf_video),
                }
            )
    expected = {"delete_type": 50, "mass_type": 75, "move_type": 50, "scale_type": 50}
    counts = {key: sum(case["edit_type"] == key for case in cases) for key in EDIT_TYPES}
    if counts != expected:
        raise RuntimeError(f"Expected {expected}, discovered {counts}")
    return cases


def resolve_models(
    cases: list[dict], requested: set[str] | None, registry_path: Path,
    factual_root: Path, cf_root: Path,
) -> tuple[list[dict], list[dict]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))["models"]
    available = []
    unavailable = []
    for model in registry:
        if requested and model["model_id"] not in requested:
            continue
        cf_dir = cf_root / model["counterfactual_dir"]
        factual_dir = factual_root / model["factual_dir"]
        missing_cf = []
        missing_factual = set()
        pairs = []
        for case in cases:
            gen_cf = cf_dir / f'{case["case_id"]}.mp4'
            gen_factual = factual_dir / f'{case["series"]}.{case["scene"]}.mp4'
            if not gen_cf.is_file():
                missing_cf.append(case["case_id"])
            if not gen_factual.is_file():
                missing_factual.add(f'{case["series"]}.{case["scene"]}')
            pairs.append({**case, "gen_factual_video": str(gen_factual), "gen_counterfactual_video": str(gen_cf)})
        status = {
            **model,
            "counterfactual_path": str(cf_dir),
            "factual_path": str(factual_dir),
            "expected_counterfactual_videos": len(cases),
            "found_counterfactual_videos": len(cases) - len(missing_cf),
            "expected_factual_scenes": len({case["scene_id"] for case in cases}),
            "found_factual_scenes": len({case["scene_id"] for case in cases}) - len(missing_factual),
            "missing_counterfactual_case_ids": missing_cf,
            "missing_factual_scene_ids": sorted(missing_factual),
        }
        if missing_cf or missing_factual:
            status["status"] = "unavailable_or_incomplete"
            unavailable.append(status)
        else:
            status["status"] = "complete"
            status["pairs"] = pairs
            available.append(status)
    return available, unavailable


def precompute_maps(videos: list[Path], cache: MotionMapCache, workers: int) -> None:
    # Several input paths can resolve to the same file (for example, symlinked
    # reference rollouts). Schedule each cache key only once.
    by_cache_path = {cache.path_for(path): path for path in videos}
    unique = sorted(by_cache_path.values(), key=lambda path: str(path).lower())
    missing = [path for path in unique if not cache.path_for(path).exists()]
    print(json.dumps({"unique_videos": len(unique), "uncached_videos": len(missing), "workers": workers}), flush=True)
    if not missing:
        return
    cv2.setNumThreads(1)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(cache.get, path): path for path in missing}
        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"Motion-map extraction failed for {path}: {exc}") from exc
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(missing):
                print(f"[motion-map] {completed}/{len(missing)}", flush=True)


def mean(values) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else math.nan


def aggregate(rows: list[dict], mode: str) -> list[dict]:
    output = []
    by_model_type = defaultdict(list)
    for row in rows:
        by_model_type[(row["model_id"], row["edit_type"])].append(row)
    for (model_id, edit_type), group in sorted(by_model_type.items()):
        record = {"model_id": model_id, "edit_type": edit_type, "aggregation": mode, "edit_count": len(group), "scene_count": len({row["scene_id"] for row in group})}
        if mode == "edit_macro":
            for metric in ALL_SCORE_METRICS:
                record[metric] = mean(row[metric] for row in group)
        elif mode == "scene_macro":
            scene_groups = defaultdict(list)
            for row in group:
                scene_groups[row["scene_id"]].append(row)
            for metric in ALL_SCORE_METRICS:
                scene_values = [mean(row[metric] for row in scene_rows) for scene_rows in scene_groups.values()]
                record[metric] = mean(scene_values)
        else:
            raise ValueError(mode)
        output.append(record)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute paired preserved/suppressed/induced counterfactual motion-map metrics.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", action="append", help="Exact model_id; repeat to select multiple models")
    parser.add_argument("--force-cache", action="store_true", help="Remove matching cached maps before extraction")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--gt-cf-root", type=Path, help="Defaults to <gt-root>/CounterfactualSubset_Experiment")
    parser.add_argument("--factual-root", type=Path, required=True)
    parser.add_argument("--counterfactual-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=SCRIPT_ROOT / "model_registry.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Defaults to <output-dir>/cache")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    config = MotionConfig()
    gt_cf_root = args.gt_cf_root or args.gt_root / "CounterfactualSubset_Experiment"
    cache_root = args.cache_dir or args.output_dir / "cache"
    cases = discover_cases(args.gt_root, gt_cf_root)
    available, unavailable = resolve_models(
        cases, set(args.model) if args.model else None, args.registry,
        args.factual_root, args.counterfactual_root,
    )
    manifest = {
        "schema_version": "cf-motion-pairing-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motion_config": asdict(config),
        "empty_union_policy": "IoU is 1 only when both reference and prediction maps/channels are empty; otherwise standard intersection over union.",
        "aggregation_policy": "Report edit-macro and scene-macro separately for delete_type, mass_type, move_type, and scale_type; scene-macro is primary.",
        "duplicate_policy": "Ignore files ending in -copy.mp4; canonical generated counterfactual filename is <case_id>.mp4.",
        "available_models": [{key: value for key, value in model.items() if key != "pairs"} for model in available],
        "unavailable_models": unavailable,
        "cases": cases,
        "code_provenance": {
            "metric_script_sha256": sha256(Path(__file__)),
            "core_script_sha256": sha256(SCRIPT_ROOT / "cf_effect_metrics.py"),
            "source_benchmark_motion_implementation_sha256": sha256(SCRIPT_ROOT.parent / "motion" / "evaluate_video_physics_metrics.py"),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "pairing_manifest.json", manifest)
    print(json.dumps({"cases": len(cases), "available_models": [m["model_id"] for m in available], "unavailable_models": [m["model_id"] for m in unavailable]}, indent=2), flush=True)
    if args.dry_run:
        return 0
    if not available:
        raise RuntimeError("No complete model is available")

    cache = MotionMapCache(cache_root, config)
    videos = []
    for model in available:
        for pair in model["pairs"]:
            videos.extend(Path(pair[key]) for key in ("gt_factual_video", "gt_counterfactual_video", "gen_factual_video", "gen_counterfactual_video"))
    if args.force_cache:
        for video in set(videos):
            cache.path_for(video).unlink(missing_ok=True)
    precompute_maps(videos, cache, args.workers)

    rows = []
    for model in available:
        model_rows = []
        for index, pair in enumerate(model["pairs"], start=1):
            maps = {key: cache.get(Path(pair[key])) for key in ("gt_factual_video", "gt_counterfactual_video", "gen_factual_video", "gen_counterfactual_video")}
            scores = score_pair(maps["gt_factual_video"], maps["gt_counterfactual_video"], maps["gen_factual_video"], maps["gen_counterfactual_video"])
            metadata = {
                "model_id": model["model_id"],
                **pair,
                **scores,
                "gt_factual_frames": maps["gt_factual_video"]["frame_count"],
                "gt_counterfactual_frames": maps["gt_counterfactual_video"]["frame_count"],
                "gen_factual_frames": maps["gen_factual_video"]["frame_count"],
                "gen_counterfactual_frames": maps["gen_counterfactual_video"]["frame_count"],
            }
            model_rows.append(metadata)
            rows.append(metadata)
            if index % 25 == 0 or index == len(model["pairs"]):
                print(f'[score] {model["model_id"]}: {index}/{len(model["pairs"])}', flush=True)
        model_dir = args.output_dir / "models" / slug(model["model_id"])
        write_csv(model_dir / "case_results.csv", model_rows)
        write_json(model_dir / "case_results.json", {"model_id": model["model_id"], "results": model_rows})

    scene_macro = aggregate(rows, "scene_macro")
    edit_macro = aggregate(rows, "edit_macro")
    write_csv(args.output_dir / "all_case_results.csv", rows)
    write_json(args.output_dir / "all_case_results.json", {"results": rows})
    write_csv(args.output_dir / "leaderboard_scene_macro.csv", scene_macro)
    write_csv(args.output_dir / "leaderboard_edit_macro.csv", edit_macro)
    write_json(
        args.output_dir / "summary.json",
        {
            "schema_version": "cf-motion-effect-results-v1",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "motion_config": asdict(config),
            "primary_metrics": list(PRIMARY_METRICS),
            "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
            "scene_macro": scene_macro,
            "edit_macro": edit_macro,
            "unavailable_models": unavailable,
        },
    )
    print(f"[done] rows={len(rows)} output={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
