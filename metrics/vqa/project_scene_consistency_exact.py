#!/usr/bin/env python3
"""Apply the frozen Scene Consistency rules and project them to exact options.

This script never opens human labels and records hashes of every input used
for each prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPRO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRO))
from scene_consistency_rule_evaluator import load_jsonl, predict  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project(task: dict[str, Any], decision: dict[str, Any]) -> tuple[int, str]:
    subdimension = task["diagnostic_subdimension"]
    signals = decision["signals"]
    if subdimension == "entity_persistence":
        birth = bool(
            signals.get("fixed_mask_duplication")
            or signals.get("reliable_sparse_redetection")
            or signals.get("fusion_or_split")
        )
        death = bool(signals.get("interior_disappearance"))
        option = 4 if birth and death else 3 if birth else 2 if death else 1
        rule = "neither->1, death->2, birth->3, birth+death->4"
    elif subdimension == "identity_geometry_integrity":
        option = 1 if decision["consistent"] else 5
        rule = "frozen first/last projection: consistent->1, violation->5"
    elif subdimension == "trajectory_continuity":
        option = 1 if decision["consistent"] else 3
        rule = "frozen first/last projection: consistent->1, violation->3"
    elif subdimension == "static_scene_camera_stability":
        camera = bool(signals.get("broad_background_motion"))
        anchored = bool(signals.get("anchored_region_residual"))
        option = 4 if camera and anchored else 3 if camera else 2 if anchored else 1
        rule = "stable->1, anchored residual->2, camera motion->3, both->4"
    else:
        raise KeyError(subdimension)
    return option, rule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--entity-evidence", required=True, type=Path)
    parser.add_argument("--identity-evidence", required=True, type=Path)
    parser.add_argument("--trajectory-evidence", required=True, type=Path)
    parser.add_argument("--static-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=0,
                        help="Optional expected number of predictions; 0 disables the count check.")
    args = parser.parse_args()

    tasks = load_jsonl(args.manifest)
    evidence_paths = {
        "entity_persistence": args.entity_evidence,
        "identity_geometry_integrity": args.identity_evidence,
        "trajectory_continuity": args.trajectory_evidence,
        "static_scene_camera_stability": args.static_evidence,
    }
    evidence = {
        subdimension: {row["unit_id"]: row for row in load_jsonl(path)}
        for subdimension, path in evidence_paths.items()
    }
    predictions = []
    for task in tasks:
        subdimension = task["diagnostic_subdimension"]
        row = evidence[subdimension][task["unit_id"]]
        decision = predict(
            subdimension,
            row if subdimension != "static_scene_camera_stability" else None,
            row if subdimension == "static_scene_camera_stability" else None,
        )
        option, projection = project(task, decision)
        predictions.append({
            "unit_id": task["unit_id"],
            "review_index": task.get("review_index"),
            "model_name": task.get("model_name"),
            "prompt_level": task.get("prompt_level"),
            "scene_id": task.get("scene_id"),
            "diagnostic_dimension": task["diagnostic_dimension"],
            "diagnostic_subdimension": subdimension,
            "predicted_option": option,
            "predicted_answer": task["answer_options"][option - 1],
            "predicted_consistent": decision["consistent"],
            "signals": decision["signals"],
            "option_projection": projection,
            "evidence_file": str(evidence_paths[subdimension].resolve()),
        })

    if args.expected_n and len(predictions) != args.expected_n:
        raise AssertionError(
            f"Expected {args.expected_n} Scene Consistency predictions, got {len(predictions)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "causalcascade-scene-consistency-exact-predictions-1.0",
        "labels_loaded": False,
        "evaluation_status": "automated predictions without human labels",
        "n": len(predictions),
        "source_hashes": {
            "manifest": sha256(args.manifest),
            "rule_evaluator": sha256(REPRO / "scene_consistency_rule_evaluator.py"),
            "this_projection_script": sha256(Path(__file__)),
            "evidence": {subdimension: sha256(path) for subdimension, path in evidence_paths.items()},
        },
        "predictions": predictions,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "n": len(predictions),
        "options": {str(i): sum(row["predicted_option"] == i for row in predictions) for i in range(1, 6)},
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
