#!/usr/bin/env python3
"""Evaluate fixed, interpretable Scene-consistency evidence rules.

The rules operate on semantically named evidence rather than a flexible fit.
Consequently leaving one pilot row out cannot change their parameters; the
reported leave-one-out stability score is the per-row prediction of this fixed
pipeline, not a claim that 20 examples constitute an independent test set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


CONSISTENT = {
    "entity_persistence": "Entity persistence is consistent.",
    "identity_geometry_integrity": "Object identity and rigid geometry are preserved.",
    "static_scene_camera_stability": "Anchored structures and the fixed camera remain stable.",
    "trajectory_continuity": "Object trajectories are spatially continuous.",
}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fusion_split_candidates(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "fusion_split_events" in row:
        return list(row["fusion_split_events"])
    redetection = row.get("sparse_redetection", {}).get("frames", [])
    if not redetection:
        return []
    unmatched = [x["center"] for x in redetection[-1].get("assignments", [])
                 if x.get("cluster") is None]
    events = []
    for track in row.get("tracks", []):
        observations = [x for x in track.get("observations", []) if x.get("visible")]
        if len(observations) < 3 or track.get("initial", {}).get("area", 0.0) < .005:
            continue
        initial_area = max(observations[0]["area"], 1e-8)
        ratios = [x["area"] / initial_area for x in observations[-3:]]
        centre = observations[-1]["center"]
        if sum(x < .82 for x in ratios) >= 2 and any(math.dist(centre, x) <= .12 for x in unmatched):
            events.append({"track_id": track["id"], "last_area_ratios": ratios})
    return events


def _dynamic_entity_lifecycle(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compute lifecycle candidates without consulting human labels.

    Propagated masks are initialized on the first frame, so their count cannot
    reveal a later object entering the scene.  Sparse re-detection supplies
    candidate additions; we retain repeated additions, a multi-instance
    terminal addition with several unmatched detections, or a strong terminal
    unmatched signal.  A track absent only in an interior gap is treated as an
    occlusion/reassociation case, not as a death; only a terminal interior
    absence is retained here.
    """
    redetection = row.get("sparse_redetection") or {}
    frames = redetection.get("frames") or []
    positive_records = []
    for frame in frames:
        deltas = [int(x) for x in (frame.get("cluster_deltas") or [])]
        positive = sum(max(0, x) for x in deltas)
        if positive:
            positive_records.append({
                "sample": int(frame.get("sample_frame", 0)),
                "positive": positive,
                "positive_clusters": [i for i, x in enumerate(deltas) if x > 0],
                "deltas": deltas,
                "unmatched": int(frame.get("unmatched_instances", 0) or 0),
            })
    cluster_hits: Dict[int, int] = {}
    cluster_strong_hits: Dict[int, int] = {}
    for record in positive_records:
        for cluster in record["positive_clusters"]:
            cluster_hits[cluster] = cluster_hits.get(cluster, 0) + 1
            # Keep the actual per-cluster delta when multiple clusters are
            # positive in one frame.  Require two delta>=2 observations so a
            # one-frame segmentation split is not promoted to a birth.
            if int(record["deltas"][cluster]) >= 2:
                cluster_strong_hits[cluster] = cluster_strong_hits.get(cluster, 0) + 1
    last_sample = max((int(x.get("sample_frame", 0)) for x in frames), default=0)
    births = []
    for record in positive_records:
        stable = any(cluster_hits.get(cluster, 0) >= 2 and
                     cluster_strong_hits.get(cluster, 0) >= 2
                     for cluster in record["positive_clusters"])
        terminal_multi = (record["sample"] == last_sample and
                          len(record["positive_clusters"]) >= 2 and
                          record["unmatched"] >= 7)
        terminal_unmatched = (record["sample"] == last_sample and
                              record["positive"] >= 1 and
                              record["unmatched"] >= 10)
        if stable or terminal_multi or terminal_unmatched:
            births.append(record)

    deaths = []
    for track in row.get("tracks", []):
        observations = track.get("observations") or []
        if len(observations) < 2:
            continue
        if (not bool(observations[-1].get("visible")) and
                bool(observations[-2].get("visible")) and
                not bool(track.get("boundary_exit"))):
            deaths.append({"track_id": track.get("id"),
                           "sample_frame": observations[-1].get("sample_frame")})
    return {"births": births, "deaths": deaths}


def predict(subdimension: str, sam: Dict[str, Any] | None,
            deterministic: Dict[str, Any] | None) -> Dict[str, Any]:
    summary = (sam or {}).get("summary", {})
    if subdimension == "entity_persistence":
        dynamic = _dynamic_entity_lifecycle(sam or {})
        fixed_duplication = int(summary.get("entity_duplication_frames", 0)) >= 2
        reliable_redetection = bool(dynamic["births"])
        fusion_split = bool(_fusion_split_candidates(sam or {}))
        disappearance = bool(dynamic["deaths"])
        # Fall back to the legacy death summary only for evidence rows that do
        # not contain per-track observations; rows with observations use the
        # boundary-aware terminal rule above.
        if not (sam or {}).get("tracks"):
            disappearance = int(summary.get("entity_death_candidates", 0)) > 0
        violation = fixed_duplication or reliable_redetection or fusion_split or disappearance
        return {"consistent": not violation,
                "signals": {"fixed_mask_duplication": fixed_duplication,
                            "reliable_sparse_redetection": reliable_redetection,
                            "computed_birth_candidates": len(dynamic["births"]),
                            "fusion_or_split": fusion_split,
                            "computed_terminal_death_candidates": len(dynamic["deaths"]),
                            "interior_disappearance": disappearance}}
    if subdimension == "identity_geometry_integrity":
        mask_split = int(summary.get("entity_duplication_frames", 0)) >= 2
        identity_loss = int(summary.get("entity_death_candidates", 0)) > 0
        violation = mask_split or identity_loss
        return {"consistent": not violation,
                "signals": {"persistent_mask_split": mask_split,
                            "interior_identity_loss": identity_loss}}
    if subdimension == "trajectory_continuity":
        replay = int(summary.get("trajectory_state_revisit_candidates", 0)) > 0
        multi_path = int(summary.get("entity_duplication_frames", 0)) >= 2
        violation = replay or multi_path
        return {"consistent": not violation,
                "signals": {"state_revisit": replay,
                            "simultaneous_multi_path_mask": multi_path}}
    if subdimension == "static_scene_camera_stability":
        features = (deterministic or {}).get("features", {})
        broad_motion = float(features.get("flow_median", 0.0)) > .10
        anchored_residual = float(features.get("centre_diff_q95", 0.0)) > 2.0
        violation = broad_motion or anchored_residual
        return {"consistent": not violation,
                "signals": {"broad_background_motion": broad_motion,
                            "anchored_region_residual": anchored_residual}}
    raise KeyError(subdimension)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--sam-evidence", action="append", default=[])
    parser.add_argument("--deterministic-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = {x["unit_id"]: x for x in load_jsonl(args.manifest)}
    labels = {x["unit_id"]: x["manual_answer"] for x in load_jsonl(args.labels)}
    sam: Dict[str, Dict[str, Any]] = {}
    for path in args.sam_evidence:
        sam.update({x["unit_id"]: x for x in load_jsonl(path)})
    deterministic = {x["unit_id"]: x for x in load_jsonl(args.deterministic_evidence)}

    rows = []
    for unit_id, task in manifest.items():
        subdimension = task.get("diagnostic_subdimension")
        if subdimension not in CONSISTENT or unit_id not in labels:
            continue
        decision = predict(subdimension, sam.get(unit_id), deterministic.get(unit_id))
        gold_consistent = labels[unit_id] == CONSISTENT[subdimension]
        rows.append({"unit_id": unit_id, "subdimension": subdimension,
                     "gold_answer": labels[unit_id],
                     "gold_consistent": gold_consistent,
                     "predicted_consistent": decision["consistent"],
                     "correct": decision["consistent"] == gold_consistent,
                     "signals": decision["signals"]})
    metrics: Dict[str, Any] = {}
    for subdimension in CONSISTENT:
        subset = [x for x in rows if x["subdimension"] == subdimension]
        correct = sum(x["correct"] for x in subset)
        metrics[subdimension] = {"n": len(subset), "correct": correct,
                                 "accuracy": correct / max(len(subset), 1)}
    correct = sum(x["correct"] for x in rows)
    report = {
        "method": "fixed_interpretable_rules",
        "pilot_cases": len(rows), "correct": correct,
        "accuracy": correct / max(len(rows), 1),
        "leave_one_out_stability_correct": correct,
        "leave_one_out_stability_note": (
            "The rules contain no pilot-fitted coefficients, so removing one row does not change "
            "its prediction. This is a stability check on the calibration set, not held-out accuracy."),
        "by_subdimension": metrics, "predictions": rows,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["pilot_cases", "correct", "accuracy",
                                             "by_subdimension"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
