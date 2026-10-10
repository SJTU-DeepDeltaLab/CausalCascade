from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from cf_effect_metrics import MotionConfig, extract_motion_maps


METRICS = (
    "cf_spatial_effect_iou",
    "cf_weighted_effect_iou",
    "cf_preserve_spatial_iou",
    "cf_preserve_weighted_iou",
    "suppressed_spatial_iou_diagnostic",
    "induced_spatial_iou_diagnostic",
    "suppressed_weighted_iou_diagnostic",
    "induced_weighted_iou_diagnostic",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate RQ2 output and motion-map extraction.")
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.results_dir
    errors = []
    with (output / "all_case_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    model_count = len({row["model_id"] for row in rows})
    if len(rows) != 225 * model_count:
        errors.append(f"Expected {225 * model_count} case rows, found {len(rows)}")
    keys = [(row["model_id"], row["case_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        errors.append("Duplicate model/case rows found")
    counts = Counter((row["model_id"], row["edit_type"]) for row in rows)
    for model in sorted({row["model_id"] for row in rows}):
        for edit_type, expected in {"delete_type": 50, "mass_type": 75, "move_type": 50, "scale_type": 50}.items():
            if counts[(model, edit_type)] != expected:
                errors.append(f"{model}: {edit_type} count is {counts[(model, edit_type)]}")
    for row in rows:
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                errors.append(f"{row['model_id']}/{row['case_id']}: invalid {metric}={value}")

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if len(summary.get("scene_macro", [])) != 4 * model_count or len(summary.get("edit_macro", [])) != 4 * model_count:
        errors.append(f"Expected {4 * model_count} rows in each aggregate table")

    import evaluate_video_physics_metrics as original
    sample = Path(rows[0]["gt_factual_video"])
    config = MotionConfig()
    streaming = extract_motion_maps(sample, config)
    video = original.load_video(sample, (config.width, config.height))
    masks = original.make_motion_masks(
        video.frames,
        threshold=config.threshold,
        bg_alpha=config.background_alpha,
        blur_kernel=config.blur_kernel,
        morph_kernel=config.morph_kernel,
    )
    extractor_equivalence = {
        "sample_video": str(sample),
        "binary_map_exact": bool(np.array_equal(streaming["spatial"], np.any(masks, axis=0))),
        "weighted_map_max_abs_error": float(np.max(np.abs(streaming["weighted"] - masks.astype(np.float32).mean(axis=0)))),
        "frame_count_matches": streaming["frame_count"] == masks.shape[0],
    }
    if not extractor_equivalence["binary_map_exact"]:
        errors.append("Streaming binary motion map differs from benchmark implementation")
    if extractor_equivalence["weighted_map_max_abs_error"] > 1e-7:
        errors.append("Streaming weighted motion map differs from benchmark implementation")
    if not extractor_equivalence["frame_count_matches"]:
        errors.append("Streaming frame count differs from benchmark implementation")

    report = {
        "schema_version": "cf-motion-metrics-validation-v1",
        "case_result_count": len(rows),
        "model_count": model_count,
        "counts": {f"{model}/{edit_type}": count for (model, edit_type), count in sorted(counts.items())},
        "extractor_equivalence": extractor_equivalence,
        "error_count": len(errors),
        "errors": errors,
    }
    (output / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
