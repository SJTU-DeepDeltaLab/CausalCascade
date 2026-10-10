"""Summarize per-video motion results for the three prompt conditions."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

METRICS = ("spatial_iou", "strict_st_iou", "weighted_spatial_iou",
           "tempo_aligned_st_iou", "dtw_aligned_st_iou")


def summarize(root: Path) -> list[dict[str, object]]:
    output = []
    for level in ("L1", "L2", "L3"):
        files = sorted((root / level).glob("*/results.csv"))
        if not files:
            raise FileNotFoundError(f"No results.csv files under {root / level}")
        for path in files:
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if not rows:
                raise ValueError(f"Empty results: {path}")
            record: dict[str, object] = {"level": level, "model": path.parent.name,
                                         "scenes": len(rows)}
            for metric in METRICS:
                record[metric] = statistics.mean(float(row[metric]) for row in rows)
            output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path,
                        help="Directory with L1/<model>/results.csv, L2/..., L3/...")
    args = parser.parse_args()
    writer = csv.DictWriter(sys.stdout, fieldnames=("level", "model", "scenes", *METRICS))
    writer.writeheader()
    writer.writerows(summarize(args.results_root))


if __name__ == "__main__":
    main()
