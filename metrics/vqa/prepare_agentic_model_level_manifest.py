#!/usr/bin/env python3
"""Build an agentic question manifest for one model and prompt level."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from dataset_questions import load_questions

DIM_ORDER = ["event_occurrence", "causal_pathway", "temporal_ordering", "outcome_termination"]


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
    selected = [q for q in questions if q.get("diagnostic_dimension") in DIM_ORDER]
    if len({q["scene_id"] for q in questions}) != 50 or len(selected) + sum(q.get("diagnostic_dimension") == "scene_consistency" for q in questions) != len(questions):
        raise SystemExit("expected the complete five-dimension, 50-scene question bank")
    rank = {d: i for i, d in enumerate(DIM_ORDER)}
    selected.sort(key=lambda q: (q["scene_id"], rank[q["diagnostic_dimension"]], q["question_id"]))

    rows = []
    missing = []
    for i, q in enumerate(selected, 1):
        filename = q["scene_id"].replace("/", ".") + ".mp4"
        gen = (args.generated_root / filename).resolve()
        gt = (args.ground_truth_root / filename).resolve()
        if not gen.is_file(): missing.append(str(gen))
        if not gt.is_file(): missing.append(str(gt))
        rows.append({
            **q,
            "model_name": args.model_name,
            "prompt_level": args.prompt_level,
            "gen_video": str(gen),
            "gt_video": str(gt),
            "model_dir": str(args.generated_root.resolve()),
            "unit_id": f"{args.prompt_level}::{args.model_name}::{q['scene_id']}::{q['question_id']}",
            "evaluation_batch": f"{args.model_name}_{args.prompt_level}_agentic",
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
        "schema": "causalcascade-agentic-model-level-manifest-1.0",
        "question_set": question_source,
        "question_set_sha256": question_digest,
        "generated_root": str(args.generated_root.resolve()),
        "ground_truth_root": str(args.ground_truth_root.resolve()),
        "model_name": args.model_name,
        "prompt_level": args.prompt_level,
        "units": len(rows),
        "unique_scenes": len({r["scene_id"] for r in rows}),
        "dimensions": dict(Counter(r["diagnostic_dimension"] for r in rows)),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
