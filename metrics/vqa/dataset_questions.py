"""Load the canonical diagnostic questions from the released dataset."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def default_dataset_root() -> Path:
    configured = os.environ.get("CAUSALCASCADE_DATASET_ROOT")
    return Path(configured) if configured else Path(__file__).resolve().parents[2] / "data" / "reference"


def _digest(rows: list[dict]) -> str:
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_questions(
    questions_jsonl: Path | None = None,
    dataset_root: Path | None = None,
) -> tuple[list[dict], str, str]:
    """Return flattened questions, their source description, and a stable digest."""
    if questions_jsonl is not None:
        rows = [
            json.loads(line)
            for line in questions_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows, str(questions_jsonl.resolve()), _digest(rows)

    root = (dataset_root or default_dataset_root()).resolve()
    files = sorted(root.glob("*/*/diagnostic_vqa.json"))
    if not files:
        raise FileNotFoundError(f"No diagnostic_vqa.json files found under {root}")

    rows: list[dict] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scene_id = payload.get("graph_id") or f"{payload['series']}/{payload['scene']}"
        questions = payload.get("questions", [])
        if payload.get("summary", {}).get("question_count") != len(questions):
            raise ValueError(f"Question count mismatch: {path}")
        for question in questions:
            rows.append({**question, "scene_id": scene_id})

    rows.sort(key=lambda row: (row["scene_id"], row["question_id"]))
    return rows, str(root), _digest(rows)
