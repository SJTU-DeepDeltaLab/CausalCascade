"""Environment and response-parsing helpers for the VQA client."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _load_env_file() -> None:
    path = Path(__file__).with_name(".env")
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key, value)


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("```")[0]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None
