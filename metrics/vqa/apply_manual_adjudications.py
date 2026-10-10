#!/usr/bin/env python3
"""Apply human choices only to three-vote cases without an agentic majority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", required=True, type=Path,
                        help="Adaptive vote merge output JSONL")
    parser.add_argument("--reviews", required=True, type=Path,
                        help="JSONL rows with unit_id and answer")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.merged)
    reviews = load_jsonl(args.reviews)
    review_map = {row["unit_id"]: row for row in reviews}
    if len(review_map) != len(reviews):
        raise ValueError("Duplicate unit_id in reviews")
    unresolved = {row["unit_id"] for row in rows
                  if row.get("adaptive_status") == "no_majority_manual_review"}
    if set(review_map) != unresolved:
        raise ValueError(f"Review IDs must exactly match unresolved IDs: "
                         f"missing={len(unresolved - set(review_map))}, "
                         f"unexpected={len(set(review_map) - unresolved)}")
    result = []
    for row in rows:
        row = dict(row)
        if row["unit_id"] in review_map:
            review = review_map[row["unit_id"]]
            answer = review.get("answer")
            if answer not in row["answer_options"]:
                raise ValueError(f"Review answer outside options: {row['unit_id']}")
            row.update({
                "adaptive_final": answer,
                "predicted": answer,
                "predicted_raw": answer,
                "adaptive_status": "manual_adjudication",
                "manual_adjudicated": True,
                "manual_adjudication_note": review.get("note", ""),
                "is_correct": answer == row["ground_truth_answer"],
            })
        result.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                   for row in result), encoding="utf-8")
    print(json.dumps({"units": len(result), "manual_adjudications": len(review_map),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
