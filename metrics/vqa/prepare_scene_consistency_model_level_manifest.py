#!/usr/bin/env python3
"""Build the 200-question Scene Consistency manifest for one rollout.

The canonical v2 question file contains 200 Scene Consistency definitions
(50 each for entity persistence, identity/geometry, trajectory continuity,
and static-scene/camera stability).  This script binds those definitions to a
specific model/prompt-level video directory without changing the questions or
their simulator reference answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from dataset_questions import load_questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="Dataset root containing <series>/<scene>/diagnostic_vqa.json")
    ap.add_argument("--questions", type=Path, default=None,
                    help="Optional legacy flattened question JSONL")
    ap.add_argument("--generated-root", type=Path, required=True)
    ap.add_argument("--ground-truth-root", type=Path, required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--prompt-level", required=True, choices=["L1", "L2", "L3"])
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    questions, question_source, question_digest = load_questions(
        args.questions, args.dataset_root)
    selected = [q for q in questions if q.get("diagnostic_dimension") == "scene_consistency"]
    if len({q["scene_id"] for q in questions}) != 50 or len(selected) != 4 * 50:
        raise SystemExit("expected four Scene Consistency questions for each of the 50 scenes")

    sub_order = {
        "entity_persistence": 0,
        "identity_geometry_integrity": 1,
        "trajectory_continuity": 2,
        "static_scene_camera_stability": 3,
    }
    selected.sort(key=lambda q: (q["scene_id"], sub_order[q["diagnostic_subdimension"]],
                                q["question_id"]))
    rows = []
    missing = []
    for i, q in enumerate(selected, 1):
        filename = q["scene_id"].replace("/", ".") + ".mp4"
        gen = (args.generated_root / filename).resolve()
        gt = (args.ground_truth_root / filename).resolve()
        if not gen.is_file():
            missing.append(str(gen))
        if not gt.is_file():
            missing.append(str(gt))
        rows.append({
            **q,
            "model_name": args.model_name,
            "prompt_level": args.prompt_level,
            "gen_video": str(gen),
            "gt_video": str(gt),
            "model_dir": str(args.generated_root.resolve()),
            "unit_id": f"{args.prompt_level}::{args.model_name}::{q['scene_id']}::{q['question_id']}",
            "evaluation_batch": f"{args.model_name}_{args.prompt_level}_scene_consistency_v1",
            "review_index": i,
        })
    if missing:
        raise SystemExit("missing media (first 20):\n" + "\n".join(missing[:20]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "schema": "causalcascade-200-scene-consistency-model-level-manifest-1.0",
        "question_set": question_source,
        "question_set_sha256": question_digest,
        "generated_root": str(args.generated_root.resolve()),
        "ground_truth_root": str(args.ground_truth_root.resolve()),
        "model_name": args.model_name,
        "prompt_level": args.prompt_level,
        "units": len(rows),
        "unique_scenes": len({r["scene_id"] for r in rows}),
        "subdimensions": dict(Counter(r["diagnostic_subdimension"] for r in rows)),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
