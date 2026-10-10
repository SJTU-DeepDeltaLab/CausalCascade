#!/usr/bin/env python3
"""Merge the local first pass and remote adaptive votes.

The first pass is treated as vote 1 and the remote full run as vote 2.  Only
units whose normalized answers differ are sent back for vote 3.  A final answer
is accepted only when at least two of the available votes agree; all-distinct
triples remain explicitly marked for manual review.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def rows(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["unit_id"]] = row
    return out


def answer(row):
    if not row:
        return None
    value = row.get("predicted")
    if value in (None, ""):
        value = row.get("predicted_raw")
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--pass1", required=True, type=Path)
    ap.add_argument("--pass2", required=True, type=Path)
    ap.add_argument("--pass3", type=Path)
    ap.add_argument("--disagreement-json", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    manifest = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    p1, p2 = rows(args.pass1), rows(args.pass2)
    p3 = rows(args.pass3) if args.pass3 and args.pass3.exists() else {}

    disagreements = []
    merged = []
    for task in manifest:
        uid = task["unit_id"]
        r1, r2, r3 = p1.get(uid), p2.get(uid), p3.get(uid)
        a1, a2, a3 = answer(r1), answer(r2), answer(r3)
        if a1 != a2:
            disagreements.append({
                "unit_id": uid,
                "scene_id": task.get("scene_id"),
                "question_id": task.get("question_id"),
                "pass1": a1,
                "pass2": a2,
                "pass3": a3,
            })
        votes = [a for a in (a1, a2, a3) if a not in (None, "")]
        counts = Counter(votes)
        majority = [a for a, n in counts.items() if n >= 2]
        final = majority[0] if len(majority) == 1 else None
        base = dict(r2 or r1 or task)
        pass_errors = {
            "pass1": bool(r1 and r1.get("error") is True),
            "pass2": bool(r2 and r2.get("error") is True),
            "pass3": bool(r3 and r3.get("error") is True),
        }
        pass_error_messages = {
            name: row.get("error_message") if row and row.get("error") is True else None
            for name, row in (("pass1", r1), ("pass2", r2), ("pass3", r3))
        }
        any_pass_error = any(pass_errors.values())
        base.update({
            "vote1": a1,
            "vote2": a2,
            "vote3": a3,
            "votes_used": len(votes),
            "adaptive_final": final,
            "adaptive_status": (
                "two_pass_agree" if a1 == a2 and a1 is not None else
                "three_pass_majority" if final is not None else
                "no_majority_manual_review"
            ),
            "pass_errors": pass_errors,
            "pass_error_messages": pass_error_messages,
            "any_pass_error": any_pass_error,
            # ``error`` describes the merged decision, not an error inherited
            # from whichever input row happened to be used as the base.
            "error": bool(final is None and any_pass_error),
            "manual_review_recommended": final is None,
        })
        # Make the merged file safe for downstream readers: ``predicted`` is
        # the adaptive majority answer, not merely the second-pass answer.
        # An all-distinct triple is deliberately left unset for manual review.
        if final is not None:
            base["predicted"] = final
            base["predicted_raw"] = final
            base.pop("error_message", None)
            if "ground_truth_answer" in base:
                base["is_correct"] = final == base["ground_truth_answer"]
        else:
            base["predicted"] = None
            base["predicted_raw"] = None
            base["is_correct"] = None
            if not any_pass_error:
                base.pop("error_message", None)
            elif not base.get("error_message"):
                base["error_message"] = next(
                    (message for message in pass_error_messages.values() if message),
                    "At least one voting pass failed; no majority was available.",
                )
        merged.append(base)

    args.disagreement_json.parent.mkdir(parents=True, exist_ok=True)
    args.disagreement_json.write_text(json.dumps(disagreements, indent=2, ensure_ascii=False) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        for row in merged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "units": len(manifest),
        "pass1": len(p1),
        "pass2": len(p2),
        "pass3": len(p3),
        "disagreements": len(disagreements),
        "two_pass_agree": sum(r["adaptive_status"] == "two_pass_agree" for r in merged),
        "three_pass_majority": sum(r["adaptive_status"] == "three_pass_majority" for r in merged),
        "no_majority_manual_review": sum(r["adaptive_status"] == "no_majority_manual_review" for r in merged),
        "disagreement_json": str(args.disagreement_json),
        "output": str(args.output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
