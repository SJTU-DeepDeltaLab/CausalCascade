#!/usr/bin/env python3
"""Deterministic diagnostics for the four Scene-consistency subdimensions.

This module is deliberately separate from the VLM judge.  It extracts visual
evidence from a rollout and applies a small, serialised calibration tree.  A
generic blob/flow tracker is used only to propose object lifecycles; camera and
appearance evidence are computed independently so that tracker artefacts do
not become a verdict by themselves.

The pilot calibrator is intended for the current 20 manually adjudicated
cases.  Its in-sample score is a calibration score.  A new held-out set must
be run with the frozen JSON calibration and labelled only afterwards.  The
object-level fallback is RGB-only because this benchmark copy contains no
simulator instance masks; its lifecycle candidates are therefore evidence for
audit, not semantic labels by themselves.

Examples
--------
  python scene_consistency_diagnostics.py fit \
      --manifest pilot_100/pilot_manifest.jsonl \
      --labels pilot_100/manual_reviews_adjudicated_v7.jsonl \
      --out results/scene_consistency_pilot_calibration

  python scene_consistency_diagnostics.py predict \
      --manifest pilot_100/pilot_manifest.jsonl \
      --calibration results/scene_consistency_pilot_calibration/calibration.json \
      --out results/scene_consistency_pilot_calibration/predictions.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
except Exception as exc:  # pragma: no cover - a clear error is better than import failure
    DecisionTreeClassifier = None
    _SKLEARN_ERROR = exc

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover
    pearsonr = spearmanr = None


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUBDIMS = {
    "entity_persistence": {
        "labels": [
            "Entity persistence is consistent.",
            "Unexplained disappearance occurs.",
            "Unexplained duplication occurs.",
            "Both unexplained disappearance and duplication occur.",
        ],
        "answer_type": "entity_persistence_status",
    },
    "identity_geometry_integrity": {
        "labels": [
            "Object identity and rigid geometry are preserved.",
            "An identity or persistent appearance switch occurs.",
            "A rigid body visibly deforms.",
            "Unexplained fragmentation or fusion occurs.",
            "Multiple identity or geometry violations occur.",
        ],
        "answer_type": "identity_geometry_status",
    },
    "static_scene_camera_stability": {
        "labels": [
            "Anchored structures and the fixed camera remain stable.",
            "An anchored structure moves, deforms, appears, or disappears.",
            "The camera viewpoint changes.",
            "Both anchored-structure and camera changes occur.",
        ],
        "answer_type": "static_scene_status",
    },
    "trajectory_continuity": {
        "labels": [
            "Object trajectories are spatially continuous.",
            "Unexplained teleportation occurs.",
            "Multiple trajectory discontinuities occur.",
        ],
        "answer_type": "trajectory_continuity_status",
    },
}


# Keep this list short enough to inspect in a report.  All features are
# dimensionless or normalised by frame width/height where appropriate.
FEATURE_NAMES = [
    "n_frames",
    "duration_s",
    "flow_median",
    "flow_q90",
    "flow_q99",
    "flow_spatial_spread",
    "frame_diff_q90",
    "frame_diff_q95",
    "frame_diff_max",
    "hist_change_q95",
    "edge_density_std",
    "affine_residual_median",
    "affine_residual_q90",
    "affine_transform_median",
    "affine_inlier_median",
    "affine_valid_fraction",
    "border_diff_q95",
    "centre_diff_q95",
    "corner_diff_q95",
    "track_count",
    "track_jumps",
    "track_fragments",
    "track_spawns",
    "track_sync_events",
    "track_max_life",
    "track_median_life",
    "track_life_coverage",
    "semantic_births",
    "semantic_deaths",
    "semantic_reidentifications",
    "semantic_duplication_candidates",
    "semantic_jump_candidates",
    "semantic_raw_jump_candidates",
    "semantic_appearance_switches",
    "semantic_geometry_jumps",
    "semantic_count_range",
    "semantic_count_variance",
]

# Video length/encoding is useful for auditing a file, but it is not evidence
# of scene consistency.  In particular, different generators use different
# frame rates and lengths; allowing those fields into calibration would leak
# model identity into the pilot labels.
POLY_CANDIDATE_FEATURES = [k for k in FEATURE_NAMES if k not in {"n_frames", "duration_s"}]


def _q(values: Sequence[float], q: float, default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        x = float(x)
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _read_video(path: str, scale: float = 0.20) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"Video contains no frames: {path}")
    return frames, float(fps)


def _affine_stats(gray: Sequence[np.ndarray]) -> Dict[str, float]:
    """Estimate frame-to-frame/global affine motion and residuals.

    ORB correspondences are intentionally used instead of dense flow for this
    quantity.  A high-consensus affine transform is evidence for camera motion;
    a high residual after the transform is evidence that the scene is not
    explained by one rigid camera transform (e.g. an anchored structure moved).
    """
    if len(gray) < 3:
        return {"resid_med": 0.0, "resid_q90": 0.0, "trans_med": 0.0,
                "inlier_med": 0.0, "valid_fraction": 0.0}
    H, W = gray[0].shape[:2]
    orb = cv2.ORB_create(nfeatures=600)
    ref_indices = sorted(set([0, len(gray) // 4, len(gray) // 2,
                              (3 * len(gray)) // 4, len(gray) - 1]))
    residuals: List[float] = []
    transforms: List[float] = []
    inliers: List[float] = []
    for i in ref_indices[1:]:
        kp0, des0 = orb.detectAndCompute(gray[0], None)
        kp1, des1 = orb.detectAndCompute(gray[i], None)
        if des0 is None or des1 is None or len(kp0) < 8 or len(kp1) < 8:
            continue
        try:
            matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des0, des1, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        except Exception:
            good = []
        if len(good) < 8:
            continue
        a = np.float32([kp0[m.queryIdx].pt for m in good])
        b = np.float32([kp1[m.trainIdx].pt for m in good])
        M, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                               ransacReprojThreshold=3.0)
        if M is None or mask is None:
            continue
        pred = cv2.transform(a[None, :, :], M)[0]
        rr = np.sqrt(((pred - b) ** 2).sum(axis=1))
        inl = mask.ravel().astype(bool)
        if not np.any(inl):
            continue
        residuals.append(float(np.median(rr[inl])))
        # ``M`` is estimated in pixel coordinates. Include normalized
        # translation as well as rotation/scale; the former implementation
        # silently ignored pure camera pans.
        rotation_scale = float(np.linalg.norm(M[:, :2] - np.eye(2)))
        translation = math.hypot(float(M[0, 2]) / max(W, 1),
                                 float(M[1, 2]) / max(H, 1))
        transforms.append(float(math.hypot(rotation_scale, translation)))
        inliers.append(float(np.mean(inl)))
    if not residuals:
        return {"resid_med": 0.0, "resid_q90": 0.0, "trans_med": 0.0,
                "inlier_med": 0.0, "valid_fraction": 0.0}
    return {
        "resid_med": float(np.median(residuals)),
        "resid_q90": float(np.quantile(residuals, 0.90)),
        "trans_med": float(np.median(transforms)),
        "inlier_med": float(np.median(inliers)),
        "valid_fraction": float(len(residuals) / max(len(ref_indices) - 1, 1)),
    }


def _blob_layer(gray: Sequence[np.ndarray]) -> Tuple[List[List[Tuple[float, float, float, int, int, int, int]]], int]:
    """Return motion blobs, with conservative adaptive-background filtering."""
    first = gray[0].astype(np.float32)
    bg = first.copy()
    H, W = gray[0].shape[:2]
    min_area = max(15, int(0.0015 * H * W))
    max_area = int(0.32 * H * W)
    layers: List[List[Tuple[float, float, float, int, int, int, int]]] = []
    for g in gray:
        d = cv2.GaussianBlur(np.abs(g.astype(np.float32) - bg), (5, 5), 0)
        mask = (d >= 7.0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        num, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        blobs = []
        for k in range(1, num):
            x, y, w, h, a = [int(v) for v in stats[k]]
            if min_area <= a <= max_area:
                blobs.append((float(cents[k][0]), float(cents[k][1]), float(a), x, y, w, h))
        layers.append(blobs)
        # Slow adaptation suppresses persistent moving objects only after they
        # have been observed for a while; it does not make a single jump vanish.
        bg = 0.97 * bg + 0.03 * g.astype(np.float32)
    return layers, H * W


def _legacy_semantic_track_stats(gray: Sequence[np.ndarray], fps: float) -> Dict[str, float]:
    """Track motion blobs with centroid, size and appearance-aware re-ID.

    This is not an object detector.  Its role is to expose lifecycle evidence
    (birth/death/re-identification/geometry changes) and to separate a likely
    tracker failure from a stable semantic track.  A human/VLM still remains
    appropriate when objects are heavily occluded or the confidence is low.
    """
    layers, image_area = _blob_layer(gray)
    H, W = gray[0].shape[:2]
    diag = math.hypot(W, H)

    def desc(g: np.ndarray, b: Tuple[float, float, float, int, int, int, int]) -> Tuple[np.ndarray, float, float]:
        _, _, _, x, y, w, h = b
        crop = g[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
        if crop.size == 0:
            hist = np.zeros(16, dtype=np.float32)
        else:
            hist = cv2.calcHist([crop], [0], None, [16], [0, 256]).ravel().astype(np.float32)
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm
        return hist, float(w / max(h, 1)), float(b[2] / image_area)

    # Track records keep last frame, center, appearance, aspect ratio and area.
    active: Dict[int, Dict[str, Any]] = {}
    next_id = 0
    all_tracks: List[Dict[str, Any]] = []
    births = deaths = reids = appearance_switches = geometry_jumps = 0
    frame_counts: List[int] = []
    pending: List[Dict[str, Any]] = []
    for fi, blobs in enumerate(layers):
        frame_counts.append(len(blobs))
        cur_desc = [desc(gray[fi], b) for b in blobs]
        candidates: List[Tuple[float, int, int]] = []
        for tid, tr in active.items():
            for bi, b in enumerate(blobs):
                cx, cy = b[0] / W, b[1] / H
                dist = math.hypot(cx - tr["cx"], cy - tr["cy"])
                if dist > 0.12:
                    continue
                hist, aspect, area = cur_desc[bi]
                app = 1.0 - float(np.dot(hist, tr["hist"]))
                ar = abs(math.log((aspect + 1e-4) / (tr["aspect"] + 1e-4)))
                size = abs(math.log((area + 1e-5) / (tr["area"] + 1e-5)))
                cost = dist / 0.12 + 0.35 * app + 0.18 * min(ar, 3.0) + 0.12 * min(size, 3.0)
                candidates.append((cost, tid, bi))
        candidates.sort(key=lambda x: x[0])
        used_t, used_b = set(), set()
        new_active: Dict[int, Dict[str, Any]] = {}
        for cost, tid, bi in candidates:
            if tid in used_t or bi in used_b:
                continue
            used_t.add(tid); used_b.add(bi)
            b = blobs[bi]; hist, aspect, area = cur_desc[bi]
            tr = active[tid]
            # A large residual change is recorded as evidence, but only if it
            # persists for two consecutive observations (avoids blur spikes).
            if 1.9 < aspect / max(tr["aspect"], 1e-5) or aspect / max(tr["aspect"], 1e-5) < 1 / 1.9:
                tr["geometry_events"] += 1
            if float(np.dot(hist, tr["hist"])) < 0.40:
                tr["appearance_events"] += 1
            tr["last_f"] = fi; tr["cx"] = b[0] / W; tr["cy"] = b[1] / H
            tr["hist"] = 0.75 * tr["hist"] + 0.25 * hist
            tr["aspect"] = 0.75 * tr["aspect"] + 0.25 * aspect
            tr["area"] = 0.75 * tr["area"] + 0.25 * area
            tr["n"] += 1
            new_active[tid] = tr
        # Unmatched tracks get one missing frame for re-ID.  A nearby blob of
        # compatible size in the next two frames counts as re-identification.
        for tid, tr in active.items():
            if tid in used_t:
                continue
            tr["missing"] += 1
            if tr["missing"] <= 2:
                pending.append(tr)
            else:
                deaths += 1
                tr["closed"] = True
                all_tracks.append(tr)
        # Unmatched blobs are births unless they re-identify a pending track.
        for bi, b in enumerate(blobs):
            if bi in used_b:
                continue
            hist, aspect, area = cur_desc[bi]
            best = None
            for tr in pending:
                if tr.get("closed"):
                    continue
                dist = math.hypot(b[0] / W - tr["cx"], b[1] / H - tr["cy"])
                if dist > 0.18:
                    continue
                app = float(np.dot(hist, tr["hist"]))
                if 0.35 <= app and 0.45 <= area / max(tr["area"], 1e-5) <= 2.2:
                    score = dist - 0.04 * app
                    if best is None or score < best[0]:
                        best = (score, tr)
            if best is not None:
                tr = best[1]; tr["missing"] = 0; tr["last_f"] = fi
                tr["cx"] = b[0] / W; tr["cy"] = b[1] / H
                tr["hist"] = hist; tr["aspect"] = aspect; tr["area"] = area; tr["n"] += 1
                new_active[tr["id"]] = tr; reids += 1
            else:
                births += 1
                tr = {"id": next_id, "first_f": fi, "last_f": fi,
                      "cx": b[0] / W, "cy": b[1] / H, "hist": hist,
                      "aspect": aspect, "area": area, "n": 1, "missing": 0,
                      "appearance_events": 0, "geometry_events": 0, "closed": False}
                new_active[next_id] = tr; next_id += 1
        active = new_active
        pending = [tr for tr in pending if not tr.get("closed") and tr.get("missing", 0) <= 2]
    for tr in active.values():
        all_tracks.append(tr)
    for tr in all_tracks:
        appearance_switches += min(int(tr.get("appearance_events", 0)), 3)
        geometry_jumps += min(int(tr.get("geometry_events", 0)), 3)
    counts = np.asarray(frame_counts, dtype=np.float32)
    life = [max(0.0, (tr["last_f"] - tr["first_f"]) / max(fps, 1e-6)) for tr in all_tracks]
    coverage = sum(max(0, tr["last_f"] - tr["first_f"] + 1) for tr in all_tracks) / max(len(gray) * 1.0, 1.0)
    return {
        "births": float(births), "deaths": float(deaths), "reids": float(reids),
        "appearance_switches": float(appearance_switches), "geometry_jumps": float(geometry_jumps),
        "count_range": float(counts.max() - counts.min()) if len(counts) else 0.0,
        "count_variance": float(np.var(counts)) if len(counts) else 0.0,
        "track_count": float(len([tr for tr in all_tracks if tr["n"] >= 3])),
        "track_max_life": float(max(life, default=0.0)),
        "track_median_life": float(np.median(life)) if life else 0.0,
        "track_life_coverage": float(coverage),
    }


def _semantic_track_stats(gray: Sequence[np.ndarray], fps: float) -> Dict[str, float]:
    """Legacy direct-call fallback; normal extraction uses RGB tracking below."""
    return _legacy_semantic_track_stats(gray, fps)


def extract_evidence(video_path: str, scale: float = 0.20) -> Dict[str, Any]:
    """Extract auditable numeric evidence from one generated rollout."""
    frames, fps = _read_video(video_path, scale=scale)
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    H, W = gray[0].shape[:2]
    flow_med: List[float] = []
    flow_q90: List[float] = []
    flow_q99: List[float] = []
    spatial_spread: List[float] = []
    diffs: List[float] = []
    border_diffs: List[float] = []
    centre_diffs: List[float] = []
    corner_diffs: List[float] = []
    hist_changes: List[float] = []
    edges: List[float] = []
    prev = gray[0]
    hists: List[np.ndarray] = []
    border = np.zeros((H, W), dtype=np.uint8); border[:max(2, H // 10), :] = 1
    border[-max(2, H // 10):, :] = 1; border[:, :max(2, W // 10)] = 1; border[:, -max(2, W // 10):] = 1
    centre = np.zeros((H, W), dtype=np.uint8); centre[H // 5:4 * H // 5, W // 5:4 * W // 5] = 1
    corner = np.zeros((H, W), dtype=np.uint8)
    corner[:H // 3, :W // 3] = 1; corner[:H // 3, 2 * W // 3:] = 1
    corner[2 * H // 3:, :W // 3] = 1; corner[2 * H // 3:, 2 * W // 3:] = 1
    for f in frames:
        hh = cv2.calcHist([f], [0, 1], None, [16, 16], [0, 256, 0, 256])
        cv2.normalize(hh, hh); hists.append(hh.ravel().astype(np.float32))
        edges.append(float(cv2.Canny(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 50, 150).mean()))
    for i in range(1, len(gray)):
        fl = cv2.calcOpticalFlowFarneback(prev, gray[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt((fl ** 2).sum(axis=2))
        flow_med.append(float(np.median(mag))); flow_q90.append(float(np.quantile(mag, .90)))
        flow_q99.append(float(np.quantile(mag, .99)))
        blocks = [np.median(mag[y * H // 3:(y + 1) * H // 3,
                                   x * W // 3:(x + 1) * W // 3]) for y in range(3) for x in range(3)]
        spatial_spread.append(float(max(blocks) - min(blocks)))
        ad = cv2.absdiff(prev, gray[i]).astype(np.float32)
        diffs.append(float(np.mean(ad)))
        border_diffs.append(float(np.mean(ad[border.astype(bool)])))
        centre_diffs.append(float(np.mean(ad[centre.astype(bool)])))
        corner_diffs.append(float(np.mean(ad[corner.astype(bool)])))
        hist_changes.append(float(cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA)))
        prev = gray[i]
    aff = _affine_stats(gray)
    object_tracker_result: Optional[Dict[str, Any]] = None
    try:
        from scene_consistency_object_tracker import track_video as object_track_video, summary_from_tracks
        # The detailed tracker samples every other source frame by default. It
        # remains dense enough for lifecycle events while keeping pilot
        # extraction practical on the available CPU.
        object_tracker_result = object_track_video(video_path, scale=scale, frame_stride=2)
        obj = summary_from_tracks(object_tracker_result)
        sem = {
            "track_count": obj["track_count"], "track_max_life": obj["track_max_life"],
            "track_median_life": obj["track_median_life"], "track_life_coverage": obj["track_life_coverage"],
            "births": obj["semantic_births"], "deaths": obj["semantic_deaths"],
            "reids": obj["semantic_reidentifications"], "appearance_switches": obj["semantic_appearance_switches"],
            "duplications": obj["semantic_duplication_candidates"],
            "jumps": obj["semantic_jump_candidates"],
            "raw_jumps": obj["semantic_raw_jump_candidates"],
            "geometry_jumps": obj["semantic_geometry_jumps"], "count_range": obj["semantic_count_range"],
            "count_variance": obj["semantic_count_variance"],
        }
    except Exception:
        sem = _legacy_semantic_track_stats(gray, fps)
        sem.setdefault("duplications", 0.0)
        sem.setdefault("jumps", 0.0)
        sem.setdefault("raw_jumps", 0.0)
    # The old tracker remains useful as a cross-check, but is not imported at
    # module import time: this file can be used independently of tools_core.
    old = {"jumps": 0.0, "frags": 0.0, "spawns": 0.0, "sync": 0.0}
    try:
        from tools_core import track_video
        raw = track_video(video_path, scale=scale)
        old["jumps"] = float(sum(t.get("jumps", 0) for t in raw.get("tracks", [])))
        old["frags"] = float(raw.get("fragment_count", 0))
        old["spawns"] = float(raw.get("spawn_count", 0))
        old["sync"] = float(len(raw.get("frame_sync_events", [])))
    except Exception:
        pass
    feat = {
        "n_frames": float(len(frames)), "duration_s": float(len(frames) / max(fps, 1e-6)),
        "flow_median": _q(flow_med, .50), "flow_q90": _q(flow_q90, .90),
        "flow_q99": _q(flow_q99, .90), "flow_spatial_spread": _q(spatial_spread, .50),
        "frame_diff_q90": _q(diffs, .90), "frame_diff_q95": _q(diffs, .95),
        "frame_diff_max": max(diffs, default=0.0), "hist_change_q95": _q(hist_changes, .95),
        "edge_density_std": float(np.std(edges)) if edges else 0.0,
        "affine_residual_median": aff["resid_med"], "affine_residual_q90": aff["resid_q90"],
        "affine_transform_median": aff["trans_med"], "affine_inlier_median": aff["inlier_med"],
        "affine_valid_fraction": aff["valid_fraction"],
        "border_diff_q95": _q(border_diffs, .95), "centre_diff_q95": _q(centre_diffs, .95),
        "corner_diff_q95": _q(corner_diffs, .95), "track_count": sem["track_count"],
        "track_jumps": old["jumps"], "track_fragments": old["frags"],
        "track_spawns": old["spawns"], "track_sync_events": old["sync"],
        "track_max_life": sem["track_max_life"], "track_median_life": sem["track_median_life"],
        "track_life_coverage": sem["track_life_coverage"], "semantic_births": sem["births"],
        "semantic_deaths": sem["deaths"], "semantic_reidentifications": sem["reids"],
        "semantic_duplication_candidates": sem["duplications"],
        "semantic_jump_candidates": sem["jumps"],
        "semantic_raw_jump_candidates": sem["raw_jumps"],
        "semantic_appearance_switches": sem["appearance_switches"],
        "semantic_geometry_jumps": sem["geometry_jumps"], "semantic_count_range": sem["count_range"],
        "semantic_count_variance": sem["count_variance"],
    }
    return {
        "video": os.path.abspath(video_path), "fps": fps, "width": W, "height": H,
        "features": {k: _safe_float(feat.get(k)) for k in FEATURE_NAMES},
        "evidence": {
            "camera_motion_proxy": aff["trans_med"],
            "unexplained_residual_proxy": aff["resid_med"],
            "border_vs_centre_change": {
                "border_q95": feat["border_diff_q95"], "centre_q95": feat["centre_diff_q95"],
            },
            "semantic_lifecycle": {k: sem[k] for k in sem},
            "object_level_tracker": (object_tracker_result.get("summary", {})
                                      if object_tracker_result is not None else None),
            "object_mask_source": (object_tracker_result.get("mask_source")
                                    if object_tracker_result is not None else "legacy_fallback"),
            "legacy_tracker_crosscheck": old,
            "limitations": [
                "The tracker uses RGB colour/shape and conservative motion proposals rather than simulator instance masks or category-level detection.",
                "A visual verdict is gated for manual review when calibration confidence is low.",
                "Appearance changes can break pixel tracking; they are not treated as identity changes without persistent evidence.",
            ],
        },
    }


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _manual_label_map(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in _load_jsonl(path):
        if row.get("unit_id") and row.get("manual_answer"):
            out[row["unit_id"]] = row["manual_answer"]
    return out


def _scene_rows(manifest: str) -> List[Dict[str, Any]]:
    return [r for r in _load_jsonl(manifest) if r.get("diagnostic_dimension") == "scene_consistency"]


def _prepare_heldout(manifest: str, questions: str, workspace: str, out_path: str,
                     seed: int = 20260911) -> Dict[str, Any]:
    """Create an unlabeled, disjoint 20-case held-out manifest.

    The pilot is excluded by ``(scene_id, question_id)`` rather than by model,
    so the held-out set tests the frozen visual rules on new rollouts.  We only
    write question/video metadata here; no answer is inferred or stored.
    """
    import random
    from collections import Counter

    pilot = _scene_rows(manifest)
    pilot_keys = {(r.get("scene_id"), r.get("question_id")) for r in pilot}
    qs = [r for r in _load_jsonl(questions)
          if r.get("diagnostic_dimension") == "scene_consistency"
          and (r.get("scene_id"), r.get("question_id")) not in pilot_keys]
    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for sub in SUBDIMS:
        pool = [q for q in qs if q.get("diagnostic_subdimension") == sub]
        rng.shuffle(pool)
        selected.extend(pool[:5])
    if len(selected) != 20:
        raise ValueError(f"expected 20 held-out questions, got {len(selected)}")

    # Discover the same 27 model/level directories used by the benchmark.  The
    # import is kept here so extraction/prediction still works in a minimal
    # environment without the benchmark preparation helper.
    try:
        from prepare_vqa_v2_pilot import discover_model_dirs
        combos = sorted(discover_model_dirs(workspace), key=lambda r: (r["model_name"], r["prompt_level"]))
    except Exception as exc:
        raise RuntimeError(f"cannot discover model-level rollouts: {exc}")
    gt_dir = os.path.join(workspace, "GroundTruth")
    tasks: List[Dict[str, Any]] = []
    # Rotate combinations across the selected questions.  This keeps model and
    # level variation without using labels or the model's prior performance.
    for i, q in enumerate(selected):
        combo = combos[i % len(combos)]
        video_file = q["scene_id"].replace("/", ".") + ".mp4"
        gen_video = os.path.join(combo["model_dir"], video_file)
        gt_video = os.path.join(gt_dir, video_file)
        if not os.path.isfile(gen_video) or not os.path.isfile(gt_video):
            raise FileNotFoundError(f"missing held-out rollout: gen={gen_video}, gt={gt_video}")
        # Do not carry the benchmark answer into the held-out artifact.  The
        # selector may read the canonical file to obtain question wording, but
        # the resulting manifest must be safe to hand to an annotator.
        question_fields = {k: v for k, v in q.items()
                           if k not in {"scene_id", "ground_truth_answer", "ground_truth_answer_code"}}
        tasks.append({
            "unit_id": f"{combo['prompt_level']}::{combo['model_name']}::{q['scene_id']}::{q['question_id']}",
            "scene_id": q["scene_id"], "prompt_level": combo["prompt_level"],
            "model_name": combo["model_name"], "model_dir": combo["model_dir"],
            "gen_video": os.path.realpath(gen_video), "gt_video": os.path.realpath(gt_video),
            **question_fields,
        })
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in tasks:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = {
        "schema_version": 1, "type": "scene_consistency_heldout_manifest",
        "seed": seed, "pilot_manifest": os.path.abspath(manifest),
        "question_source": os.path.abspath(questions), "n": len(tasks),
        "labels_intentionally_omitted": True,
        "subdimensions": dict(Counter(t["diagnostic_subdimension"] for t in tasks)),
        "models": dict(Counter(t["model_name"] for t in tasks)),
        "prompt_levels": dict(Counter(t["prompt_level"] for t in tasks)),
        "pilot_exclusion": "scene_id + question_id",
    }
    with open(str(out) + ".metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, ensure_ascii=False))
    return metadata


def _vector(evidence: Dict[str, Any]) -> List[float]:
    f = evidence["features"]
    return [_safe_float(f.get(k)) for k in FEATURE_NAMES]


def _is_consistent_answer(sub: str, answer: str) -> int:
    """Binary target used only for the deterministic consistency analysis.

    The first option in each canonical question is the positive/consistent
    state (consistent, preserved, stable, or continuous).  The full
    multi-class answer remains the output of the categorical calibration tree.
    """
    return int(answer == SUBDIMS[sub]["labels"][0])


def _corr(x: Sequence[float], y: Sequence[float]) -> Dict[str, Optional[float]]:
    if len(x) < 3 or len(set(float(v) for v in x)) < 2 or len(set(float(v) for v in y)) < 2:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    if pearsonr is None or spearmanr is None:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr, pp = pearsonr(x, y); sr, sp = spearmanr(x, y)
    return {"pearson_r": round(float(pr), 5), "pearson_p": round(float(pp), 5),
            "spearman_r": round(float(sr), 5), "spearman_p": round(float(sp), 5)}


def _fit_polynomial_models(evidence_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fit an auditable degree-2 logistic model for binary consistency.

    Feature selection is deliberately restricted to the two strongest
    monotonic associations on the pilot.  With only five cases per
    subdimension, fitting all 33 features (or a high-degree polynomial) would
    be numerically impressive but scientifically meaningless.  The selected
    features, correlations, coefficients, and leave-one-out score are retained
    in the calibration artifact.
    """
    if PolynomialFeatures is None or pearsonr is None:
        raise RuntimeError(f"scikit-learn/scipy are required for polynomial analysis: {_SKLEARN_ERROR}")
    models: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    c_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    for sub in SUBDIMS:
        ds = [d for d in evidence_rows if d["subdimension"] == sub]
        if not ds:
            continue
        X = np.asarray([_vector(d) for d in ds], dtype=np.float64)
        y = np.asarray([_is_consistent_answer(sub, d["manual_answer"]) for d in ds], dtype=np.int64)
        correlations = {}
        ranks = []
        for name in POLY_CANDIDATE_FEATURES:
            j = FEATURE_NAMES.index(name)
            c = _corr(X[:, j].tolist(), y.tolist())
            correlations[name] = c
            ranks.append((abs(c["spearman_r"] or 0.0), abs(c["pearson_r"] or 0.0), name, j))
        ranks.sort(key=lambda z: (-z[0], -z[1], z[2]))
        selected = [z[3] for z in ranks[:2]]
        Xs = X[:, selected]
        best = None
        for C in c_grid:
            model = Pipeline([
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("lr", LogisticRegression(C=C, solver="liblinear", max_iter=5000,
                                            random_state=0)),
            ]).fit(Xs, y)
            pred = model.predict(Xs).astype(int)
            fit_correct = int(np.sum(pred == y))
            candidate = (fit_correct, -C, model)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        fit_correct, _, model = best
        # Leave-one-out is reported for transparency; it is not used to tune C
        # or to claim held-out performance.
        loo_correct = 0
        for i in range(len(y)):
            loo = Pipeline([
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("lr", LogisticRegression(C=-best[1], solver="liblinear", max_iter=5000,
                                            random_state=0)),
            ]).fit(np.delete(Xs, i, axis=0), np.delete(y, i))
            loo_correct += int(loo.predict(Xs[i:i + 1])[0] == y[i])
        scaler = model.named_steps["scale"]
        poly = model.named_steps["poly"]
        lr = model.named_steps["lr"]
        # ``powers_`` makes the polynomial representation independent of
        # sklearn's feature-name formatting and easy to re-evaluate elsewhere.
        models[sub] = {
            "selected_features": [FEATURE_NAMES[j] for j in selected],
            "selected_indices": selected,
            "degree": 2,
            "C": float(-best[1]),
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "powers": poly.powers_.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
            "correlations": correlations,
        }
        metrics[sub] = {
            "n": int(len(y)), "consistent_n": int(np.sum(y)),
            "selected_features": [FEATURE_NAMES[j] for j in selected],
            "pilot_fit_correct": fit_correct, "pilot_fit_accuracy": fit_correct / max(len(y), 1),
            "leave_one_out_correct": int(loo_correct), "leave_one_out_accuracy": loo_correct / max(len(y), 1),
            "note": "degree-2 logistic fit on two selected features; pilot fit is calibration, LOO is diagnostic",
        }
    return {"models": models, "metrics": metrics}


def _poly_probability(spec: Dict[str, Any], features: Dict[str, float]) -> float:
    raw = np.asarray([_safe_float(features.get(name)) for name in spec["selected_features"]], dtype=np.float64)
    scale = np.asarray(spec["scale"], dtype=np.float64)
    mean = np.asarray(spec["mean"], dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    z = (raw - mean) / scale
    powers = np.asarray(spec["powers"], dtype=np.int64)
    terms = np.prod(np.power(z[None, :], powers), axis=1)
    logit = float(np.dot(np.asarray(spec["coef"], dtype=np.float64), terms) + spec["intercept"])
    return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit)))))


def _tree_to_json(clf: Any, feature_names: Sequence[str]) -> Dict[str, Any]:
    t = clf.tree_
    def rec(node: int) -> Dict[str, Any]:
        left, right = int(t.children_left[node]), int(t.children_right[node])
        # sklearn stores only the classes observed by this subdimension.  Keep
        # their original integer ids, otherwise a leaf containing class 4
        # would be mistaken for class 1 when the tree is reloaded.
        values = t.value[node][0].tolist()
        if left == right:
            return {"leaf": True, "value": values,
                    "class_indices": [int(x) for x in clf.classes_],
                    "samples": int(t.n_node_samples[node])}
        return {"leaf": False, "feature": feature_names[int(t.feature[node])],
                "threshold": float(t.threshold[node]), "samples": int(t.n_node_samples[node]),
                "left": rec(left), "right": rec(right)}
    return rec(0)


def _tree_predict(node: Dict[str, Any], features: Dict[str, float]) -> Tuple[int, float, int]:
    while not node.get("leaf"):
        val = _safe_float(features.get(node["feature"]))
        node = node["left"] if val <= node["threshold"] else node["right"]
    values = np.asarray(node.get("value", []), dtype=np.float64)
    if len(values) == 0 or values.sum() <= 0:
        return 0, 0.0, int(node.get("samples", 0))
    local_idx = int(values.argmax())
    class_indices = node.get("class_indices") or list(range(len(values)))
    idx = int(class_indices[local_idx])
    return idx, float(values[local_idx] / values.sum()), int(node.get("samples", 0))


def _fit(manifest: str, labels_path: str, out_dir: str, scale: float) -> Dict[str, Any]:
    if DecisionTreeClassifier is None:
        raise RuntimeError(f"scikit-learn is required for calibration: {_SKLEARN_ERROR}")
    rows = _scene_rows(manifest)
    labels = _manual_label_map(labels_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    evidence_rows: List[Dict[str, Any]] = []
    X_by: Dict[str, List[List[float]]] = defaultdict(list)
    y_by: Dict[str, List[str]] = defaultdict(list)
    for i, row in enumerate(rows, 1):
        uid = row["unit_id"]
        if uid not in labels:
            raise ValueError(f"No manual label for {uid}")
        ev = extract_evidence(row["gen_video"], scale=scale)
        ev.update({"unit_id": uid, "subdimension": row["diagnostic_subdimension"],
                   "model_name": row.get("model_name"), "prompt_level": row.get("prompt_level"),
                   "scene_id": row.get("scene_id"), "manual_answer": labels[uid]})
        evidence_rows.append(ev)
        sub = row["diagnostic_subdimension"]
        X_by[sub].append(_vector(ev)); y_by[sub].append(labels[uid])
        print(f"[{i:02d}/{len(rows):02d}] extracted {sub}: {row['scene_id']}", flush=True)
    trees: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    for sub, cfg in SUBDIMS.items():
        if not X_by[sub]:
            continue
        classes = cfg["labels"]
        y_idx = [classes.index(y) for y in y_by[sub]]
        tree_indices = [FEATURE_NAMES.index(name) for name in POLY_CANDIDATE_FEATURES]
        X_tree = np.asarray([[row[j] for j in tree_indices] for row in X_by[sub]], dtype=np.float64)
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, random_state=0,
                                     class_weight=None).fit(X_tree, y_idx)
        pred = clf.predict(X_tree).astype(int).tolist()
        correct = sum(int(a == b) for a, b in zip(y_idx, pred))
        # The tree shown in calibration.json is fit on all pilot cases, but
        # report an explicit leave-one-out diagnostic for the primary model as
        # well.  This prevents the 20/20 calibration fit from being mistaken
        # for generalization.
        tree_loo_correct = 0
        for holdout in range(len(y_idx)):
            loo_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1,
                                              random_state=0,
                                              class_weight=None).fit(
                                                  np.delete(X_tree, holdout, axis=0),
                                                  np.delete(y_idx, holdout))
            tree_loo_correct += int(loo_tree.predict(X_tree[holdout:holdout + 1])[0] == y_idx[holdout])
        trees[sub] = {"classes": classes, "features": POLY_CANDIDATE_FEATURES,
                      "tree": _tree_to_json(clf, POLY_CANDIDATE_FEATURES), "n": len(y_idx)}
        metrics[sub] = {"n": len(y_idx), "correct": correct,
                        "accuracy": correct / max(len(y_idx), 1),
                        "leave_one_out_correct": tree_loo_correct,
                        "leave_one_out_accuracy": tree_loo_correct / max(len(y_idx), 1),
                        "calibration_note": "in-sample pilot fit; not a held-out estimate"}
    polynomial = _fit_polynomial_models(evidence_rows)
    calibration = {
        "schema_version": 1, "type": "scene_consistency_calibration",
        "created_from": {"manifest": os.path.abspath(manifest), "labels": os.path.abspath(labels_path),
                          "scale": scale, "n_cases": len(rows)},
        "feature_names": FEATURE_NAMES, "subdimensions": trees, "metrics": metrics,
        "polynomial_consistency": polynomial,
        "decision_policy": {
            "manual_review_confidence_below": 0.75,
            "manual_review_reason": "numeric evidence is ambiguous or leaf has mixed pilot labels",
            "fixed_before_heldout": True,
        },
    }
    with open(out / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2, ensure_ascii=False)
    with open(out / "evidence.jsonl", "w", encoding="utf-8") as f:
        for row in evidence_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out / "polynomial_analysis.json", "w", encoding="utf-8") as f:
        json.dump(polynomial, f, indent=2, ensure_ascii=False)
    print(json.dumps({"categorical_metrics": metrics,
                      "polynomial_metrics": polynomial["metrics"],
                      "total_correct": sum(x["correct"] for x in metrics.values()),
                      "total_cases": sum(x["n"] for x in metrics.values())}, ensure_ascii=False))
    return calibration


def _predict(manifest: str, calibration_path: str, out_path: str, scale: float,
             labels_path: Optional[str] = None) -> Dict[str, Any]:
    with open(calibration_path, encoding="utf-8") as f:
        calibration = json.load(f)
    rows = _scene_rows(manifest)
    labels = _manual_label_map(labels_path) if labels_path else {}
    predictions: List[Dict[str, Any]] = []
    correct = 0
    evaluated = 0
    for i, row in enumerate(rows, 1):
        ev = extract_evidence(row["gen_video"], scale=scale)
        sub = row["diagnostic_subdimension"]
        spec = calibration["subdimensions"][sub]
        idx, confidence, leaf_n = _tree_predict(spec["tree"], ev["features"])
        answer = spec["classes"][idx]
        poly_spec = calibration.get("polynomial_consistency", {}).get("models", {}).get(sub)
        poly_prob = _poly_probability(poly_spec, ev["features"]) if poly_spec else None
        uid = row["unit_id"]
        gold = labels.get(uid)
        match = gold == answer if gold else None
        if match is not None:
            evaluated += 1; correct += int(match)
        predictions.append({
            "unit_id": uid, "model_name": row.get("model_name"), "prompt_level": row.get("prompt_level"),
            "scene_id": row.get("scene_id"), "subdimension": sub,
            "prediction": answer, "confidence": round(confidence, 4), "leaf_samples": leaf_n,
            "needs_manual_review": confidence < calibration["decision_policy"]["manual_review_confidence_below"],
            "polynomial_consistency_probability": round(poly_prob, 4) if poly_prob is not None else None,
            "polynomial_consistent": bool(poly_prob >= 0.5) if poly_prob is not None else None,
            "gold": gold, "match": match, "evidence": ev,
        })
        print(f"[{i:02d}/{len(rows):02d}] {answer} conf={confidence:.2f} {row['scene_id']}", flush=True)
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"n": len(predictions), "evaluated": evaluated,
               "correct": correct if evaluated else None,
               "accuracy": correct / evaluated if evaluated else None,
               "manual_review": sum(p["needs_manual_review"] for p in predictions),
               "calibration": os.path.abspath(calibration_path), "predictions": os.path.abspath(out_path)}
    with open(str(out) + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="calibrate on manually adjudicated pilot cases")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--labels", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--scale", type=float, default=0.20)
    pred = sub.add_parser("predict", help="run a frozen calibration on a manifest")
    pred.add_argument("--manifest", required=True)
    pred.add_argument("--calibration", required=True)
    pred.add_argument("--out", required=True)
    pred.add_argument("--labels", default=None, help="optional labels for evaluation only")
    pred.add_argument("--scale", type=float, default=0.20)
    held = sub.add_parser("prepare-heldout", help="prepare an unlabeled 20-case held-out manifest")
    held.add_argument("--manifest", required=True, help="current pilot manifest used for exclusion")
    held.add_argument("--questions", required=True, help="canonical diagnostic question JSONL")
    held.add_argument("--workspace", required=True, help="benchmark workspace containing model dirs and GroundTruth")
    held.add_argument("--out", required=True)
    held.add_argument("--seed", type=int, default=20260911)
    args = p.parse_args()
    if args.command == "fit":
        _fit(args.manifest, args.labels, args.out, args.scale)
    elif args.command == "predict":
        _predict(args.manifest, args.calibration, args.out, args.scale, args.labels)
    else:
        _prepare_heldout(args.manifest, args.questions, args.workspace, args.out, args.seed)


if __name__ == "__main__":
    _main()
