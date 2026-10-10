#!/usr/bin/env python3
"""Select only pass1/pass2 disagreements for a third independent agent vote."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def by_unit(rows: list[dict], description: str) -> dict[str, dict]:
    result = {row["unit_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate unit_id in {description}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pass1", required=True, type=Path)
    parser.add_argument("--pass2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = load_jsonl(args.manifest)
    tasks = by_unit(manifest, "manifest")
    first = by_unit(load_jsonl(args.pass1), "pass1")
    second = by_unit(load_jsonl(args.pass2), "pass2")
    if set(first) != set(tasks) or set(second) != set(tasks):
        raise ValueError("Both passes must contain every manifest unit exactly once")

    disagreements = []
    for task in manifest:
        uid = task["unit_id"]
        a1 = first[uid].get("predicted")
        a2 = second[uid].get("predicted")
        if a1 not in task["answer_options"] or a2 not in task["answer_options"]:
            raise ValueError(f"Invalid vote for {uid}; repair or rerun it before pass3")
        if a1 != a2:
            disagreements.append(task)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in disagreements),
        encoding="utf-8",
    )
    print(json.dumps({"units": len(manifest), "disagreements": len(disagreements),
                      "pass3_manifest": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
