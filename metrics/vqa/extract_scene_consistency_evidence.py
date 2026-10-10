#!/usr/bin/env python3
"""Extract deterministic Scene-consistency evidence without loading labels.

This is deliberately a thin wrapper around the frozen evidence extractor.  It
exists so a held-out manifest can be processed before its annotations are
opened, without modifying the frozen rule/evidence implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene_consistency_diagnostics import extract_evidence


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subdimension", default=None)
    parser.add_argument("--scale", type=float, default=0.20)
    args = parser.parse_args()

    rows = [
        row for row in load_jsonl(args.manifest)
        if row.get("diagnostic_dimension") == "scene_consistency"
        and (args.subdimension is None
             or row.get("diagnostic_subdimension") == args.subdimension)
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            evidence = extract_evidence(row["gen_video"], scale=args.scale)
            evidence.update({
                "unit_id": row["unit_id"],
                "subdimension": row["diagnostic_subdimension"],
                "model_name": row.get("model_name"),
                "prompt_level": row.get("prompt_level"),
                "scene_id": row.get("scene_id"),
            })
            handle.write(json.dumps(evidence, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index:02d}/{len(rows):02d}] "
                f"{row['diagnostic_subdimension']}: {row['unit_id']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
