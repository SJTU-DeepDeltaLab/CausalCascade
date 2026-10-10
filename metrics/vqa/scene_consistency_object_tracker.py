#!/usr/bin/env python3
"""Object-level (RGB-only) tracking used by Scene-consistency diagnostics.

The benchmark videos do not ship simulator instance masks.  This module
therefore provides a deterministic fallback: motion-region masks are proposed
from an adaptive background, then associated with centroid, area, aspect-ratio
and colour descriptors.  A proposal is not called a semantic duplication or
disappearance until its lifecycle and boundary/occlusion evidence satisfy the
rules below.  If simulator masks or SAM2 become available later, they can
replace only ``_detect_objects``; the lifecycle and audit schema stay the same.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def read_video(path: str, scale: float = 0.25) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
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
        raise ValueError(f"empty video: {path}")
    return frames, float(fps)


def _describe_component(cmask: np.ndarray, contour: np.ndarray, cents: np.ndarray,
                        frame: np.ndarray, H: int, W: int, area: int,
                        x: int, y: int, w: int, h: int) -> Dict[str, Any]:
    """Create one object proposal with appearance and shape evidence."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    crop = hsv[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
    cmask_crop = cmask[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
    hist = cv2.calcHist([crop], [0, 1], cmask_crop, [8, 8], [0, 180, 0, 256]).ravel()
    hist = hist.astype(np.float32)
    norm = float(np.linalg.norm(hist))
    if norm > 1e-8:
        hist /= norm
    hull = cv2.convexHull(contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    contour_area = max(float(cv2.contourArea(contour)), 1.0)
    cx, cy = float(cents[0] / W), float(cents[1] / H)
    return {
        "center": [cx, cy],
        "bbox": [float(x / W), float(y / H), float(w / W), float(h / H)],
        "area": float(area / (W * H)), "aspect": float(w / max(h, 1)),
        "solidity": float(contour_area / hull_area), "hist": hist,
        "contour": [[int(p[0][0]), int(p[0][1])] for p in contour[::max(1, len(contour) // 40)]],
        "boundary": bool(cx < .035 or cx > .965 or cy < .035 or cy > .965),
    }


def _components(mask: np.ndarray, frame: np.ndarray, min_area: int,
                max_area: int) -> List[Dict[str, Any]]:
    H, W = mask.shape[:2]
    num, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    objects: List[Dict[str, Any]] = []
    for k in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[k]]
        if area < min_area or area > max_area:
            continue
        cmask = (lab == k).astype(np.uint8)
        contours, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        objects.append(_describe_component(cmask, contour, cents[k], frame, H, W,
                                            area, x, y, w, h))
    return objects


def _instance_mask_layers(video_path: str, source_indices: Sequence[int],
                          frames: Sequence[np.ndarray]) -> Optional[List[List[Dict[str, Any]]]]:
    """Load simulator instance masks when a conventional sidecar is present.

    The released benchmark copy currently has no such sidecars, so this
    returns ``None`` and the RGB fallback is used. Supporting both an NPZ
    (`masks` or `labels`, shape T/H/W) and PNG masks in a sibling directory
    keeps the proposal interface replaceable without changing association or
    lifecycle rules.
    """
    video = Path(video_path)
    npz_candidates = [video.with_suffix(".masks.npz"),
                      video.with_name(video.stem + "_masks.npz")]
    masks: Optional[np.ndarray] = None
    for candidate in npz_candidates:
        if not candidate.is_file():
            continue
        try:
            data = np.load(candidate, allow_pickle=False)
            key = "masks" if "masks" in data else ("labels" if "labels" in data else None)
            if key is not None:
                masks = np.asarray(data[key])
                break
        except Exception:
            masks = None
    mask_dir = video.with_name(video.stem + "_masks")
    if masks is None and mask_dir.is_dir():
        pngs = sorted(mask_dir.glob("*.png"))
        if len(pngs) >= len(source_indices):
            loaded = []
            for idx in source_indices:
                if idx >= len(pngs):
                    return None
                im = cv2.imread(str(pngs[idx]), cv2.IMREAD_UNCHANGED)
                if im is None:
                    return None
                loaded.append(im)
            masks = np.stack(loaded, axis=0)
    if masks is None or masks.ndim < 3:
        return None
    if masks.shape[0] <= max(source_indices, default=0):
        return None
    selected = masks[list(source_indices)]
    H, W = frames[0].shape[:2]
    layers: List[List[Dict[str, Any]]] = []
    for i, label_img in enumerate(selected):
        if label_img.shape[:2] != (H, W):
            label_img = cv2.resize(label_img, (W, H), interpolation=cv2.INTER_NEAREST)
        labels = np.unique(label_img)
        objects: List[Dict[str, Any]] = []
        for label in labels:
            if int(label) == 0:
                continue
            mask = (label_img == label).astype(np.uint8)
            num, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
            for k in range(1, num):
                x, y, w, h, comp_area = [int(v) for v in stats[k]]
                if comp_area < max(12, int(.00025 * H * W)) or comp_area > int(.5 * H * W):
                    continue
                cmask = (lab == k).astype(np.uint8)
                contours, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    objects.append(_describe_component(cmask, max(contours, key=cv2.contourArea), cents[k],
                                                        frames[i], H, W, comp_area, x, y, w, h))
        layers.append(objects)
    return layers


def _mask_proposals(frames: Sequence[np.ndarray]) -> List[List[Dict[str, Any]]]:
    """Propose object regions from colour/shape first, motion second.

    Pure frame differencing fragments an object whenever its appearance changes
    or it moves faster than the sampling interval.  Saturated connected regions
    are therefore the primary proposals (the benchmark scenes use coloured
    balls, blocks, dominoes and targets); conservative motion regions are only
    added when they do not overlap a colour proposal.  This is still an RGB
    fallback, not a semantic segmentation model, and all proposals remain
    visible in the audit report.
    """
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    H, W = gray[0].shape[:2]
    bg = gray[0].astype(np.float32)
    min_area = max(12, int(0.0007 * H * W))
    max_area = int(0.22 * H * W)
    layers: List[List[Dict[str, Any]]] = []
    kernel = np.ones((3, 3), np.uint8)
    colour_kernel = np.ones((5, 5), np.uint8)
    for i, g in enumerate(gray):
        diff = cv2.GaussianBlur(np.abs(g.astype(np.float32) - bg), (5, 5), 0)
        motion_mask = (diff >= 10.0).astype(np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, colour_kernel)
        motion_objects = _components(motion_mask, frames[i], max(min_area, int(.0025 * H * W)),
                                      max_area)

        hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
        colour_mask = ((hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 40)).astype(np.uint8)
        colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_OPEN, kernel)
        colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, colour_kernel)
        colour_objects = _components(colour_mask, frames[i], min_area, max_area)

        # Motion proposals that are not already represented by a coloured
        # object can still capture low-saturation wooden/metal pieces.  Reject
        # overlap instead of blindly taking the union, which is what caused
        # one object to become several IDs in the former tracker.
        objects = list(colour_objects)
        for mo in motion_objects:
            mx, my, mw, mh = mo["bbox"]
            mbox = (mx, my, mx + mw, my + mh)
            overlap = False
            for co in colour_objects:
                cx, cy, cw, ch = co["bbox"]
                cbox = (cx, cy, cx + cw, cy + ch)
                ix0, iy0 = max(mbox[0], cbox[0]), max(mbox[1], cbox[1])
                ix1, iy1 = min(mbox[2], cbox[2]), min(mbox[3], cbox[3])
                inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                union = mw * mh + cw * ch - inter
                if union > 0 and inter / union >= .25:
                    overlap = True
                    break
            if not overlap:
                objects.append(mo)
        layers.append(objects)
        bg = 0.97 * bg + 0.03 * g.astype(np.float32)
    return layers


def _pair_affine(prev: np.ndarray, cur: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Return an affine prev->cur, residual and inlier fraction."""
    orb = cv2.ORB_create(nfeatures=450)
    kp0, des0 = orb.detectAndCompute(prev, None)
    kp1, des1 = orb.detectAndCompute(cur, None)
    eye = np.eye(3, dtype=np.float64)
    if des0 is None or des1 is None or len(kp0) < 8 or len(kp1) < 8:
        return eye, 9.0, 0.0
    try:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des0, des1, k=2)
        good = [m for m, n in matches if m.distance < .75 * n.distance]
    except Exception:
        good = []
    if len(good) < 8:
        return eye, 9.0, 0.0
    a = np.float32([kp0[m.queryIdx].pt for m in good])
    b = np.float32([kp1[m.trainIdx].pt for m in good])
    M, inlier_mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                  ransacReprojThreshold=3.0)
    if M is None or inlier_mask is None:
        return eye, 9.0, 0.0
    pred = cv2.transform(a[None], M)[0]
    rr = np.sqrt(((pred - b) ** 2).sum(axis=1))
    inl = inlier_mask.ravel().astype(bool)
    if not np.any(inl):
        return eye, 9.0, 0.0
    out = eye.copy(); out[:2] = M
    return out, float(np.median(rr[inl])), float(np.mean(inl))


def _apply_homography(M: np.ndarray, xy: Sequence[float]) -> List[float]:
    p = np.array([float(xy[0]), float(xy[1]), 1.0], dtype=np.float64)
    q = M @ p
    if abs(q[2]) < 1e-9:
        return [float(xy[0]), float(xy[1])]
    return [float(q[0] / q[2]), float(q[1] / q[2])]


def _compensated_normalized_center(M_pixel: np.ndarray, xy_normalized: Sequence[float],
                                   width: int, height: int) -> List[float]:
    """Apply a pixel-coordinate camera transform to a normalized centre.

    ``estimateAffinePartial2D`` returns translations in pixels.  Applying that
    matrix directly to a [0, 1] centre (as the earlier implementation did)
    mixes coordinate systems and creates artificial trajectory jumps.
    """
    pixel = [float(xy_normalized[0]) * width, float(xy_normalized[1]) * height]
    corrected = _apply_homography(M_pixel, pixel)
    return [float(corrected[0] / max(width, 1)),
            float(corrected[1] / max(height, 1))]


def _close_track(track: Dict[str, Any], tracks: List[Dict[str, Any]], events: Dict[str, List[Any]], W: int, H: int) -> None:
    if track.get("closed"):
        return
    track["closed"] = True
    last = track["observations"][-1] if track["observations"] else None
    if last is None:
        return
    # Boundary exit is expected disappearance and is never coded as an
    # unexplained disappearance.  Interior death is retained as a candidate;
    # the ambiguity is surfaced in the HTML rather than hidden.
    if not last.get("boundary", False):
        events["deaths"].append({"track_id": track["id"], "frame": last["frame"],
                                  "reason": "interior_track_end"})


def track_video(path: str, scale: float = 0.25, frame_stride: int = 1,
                max_missing: int = 3) -> Dict[str, Any]:
    """Track motion-region objects and return full frame-level audit data."""
    source_frames, fps = read_video(path, scale=scale)
    indices = list(range(0, len(source_frames), max(1, int(frame_stride))))
    if indices[-1] != len(source_frames) - 1:
        indices.append(len(source_frames) - 1)
    frames = [source_frames[i] for i in indices]
    H, W = frames[0].shape[:2]
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    layers = _instance_mask_layers(path, indices, frames)
    mask_source = "simulator_instance_sidecar" if layers is not None else "rgb_colour_shape_motion_fallback"
    if layers is None:
        layers = _mask_proposals(frames)

    # Cumulative image->reference transform and per-frame camera diagnostics.
    cumulative = np.eye(3, dtype=np.float64)
    cumulative_inv = [np.eye(3, dtype=np.float64)]
    camera = [{"frame": indices[0], "motion": 0.0, "residual": 0.0, "inlier": 1.0}]
    for i in range(1, len(frames)):
        M, resid, inlier = _pair_affine(gray[i - 1], gray[i])
        cumulative = M @ cumulative
        try:
            inv = np.linalg.inv(cumulative)
        except np.linalg.LinAlgError:
            inv = np.eye(3, dtype=np.float64)
        cumulative_inv.append(inv)
        rotation_scale = float(np.linalg.norm(M[:2, :2] - np.eye(2)))
        translation = math.hypot(float(M[0, 2]) / max(W, 1),
                                 float(M[1, 2]) / max(H, 1))
        camera.append({"frame": indices[i], "motion": float(math.hypot(rotation_scale, translation)),
                       "rotation_scale": rotation_scale, "translation": translation,
                       "residual": resid, "inlier": inlier})

    active: Dict[int, Dict[str, Any]] = {}
    closed: List[Dict[str, Any]] = []
    next_id = 0
    events: Dict[str, List[Any]] = {"births": [], "deaths": [], "reidentifications": [],
                                    "duplicates": [], "jumps": [], "appearance_switches": [],
                                    "geometry_jumps": []}
    frame_records: List[Dict[str, Any]] = []

    for local_i, objects in enumerate(layers):
        source_i = indices[local_i]
        used_t, used_o = set(), set()
        candidates: List[Tuple[float, int, int]] = []
        for tid, tr in active.items():
            if tr.get("closed"):
                continue
            last = tr["observations"][-1]
            # Constant-velocity prediction in image coordinates.
            pred = list(last["center"])
            if len(tr["observations"]) >= 2:
                a = tr["observations"][-2]["center"]; b = last["center"]
                pred = [b[0] + (b[0] - a[0]), b[1] + (b[1] - a[1])]
            for oi, ob in enumerate(objects):
                dist = math.hypot(ob["center"][0] - pred[0], ob["center"][1] - pred[1])
                if dist > .18:
                    continue
                app = 1.0 - float(np.dot(ob["hist"], last["hist"]))
                area = abs(math.log((ob["area"] + 1e-5) / (last["area"] + 1e-5)))
                aspect = abs(math.log((ob["aspect"] + 1e-4) / (last["aspect"] + 1e-4)))
                cost = dist / .18 + .30 * app + .12 * min(area, 3.0) + .12 * min(aspect, 3.0)
                candidates.append((cost, tid, oi))
        candidates.sort(key=lambda x: x[0])
        for cost, tid, oi in candidates:
            if tid in used_t or oi in used_o:
                continue
            used_t.add(tid); used_o.add(oi)
            tr = active[tid]; ob = objects[oi]
            last = tr["observations"][-1]
            gap = local_i - last["local_frame"]
            app_sim = float(np.dot(ob["hist"], last["hist"]))
            if app_sim < .40:
                events["appearance_switches"].append({"track_id": tid, "frame": source_i, "similarity": app_sim})
            if abs(math.log((ob["aspect"] + 1e-4) / (last["aspect"] + 1e-4))) > math.log(1.9):
                events["geometry_jumps"].append({"track_id": tid, "frame": source_i})
            obs = dict(ob); obs.update({"track_id": tid, "frame": source_i, "local_frame": local_i,
                                       "time": source_i / max(fps, 1e-6),
                                       "compensated_center": _compensated_normalized_center(
                                           cumulative_inv[local_i], ob["center"], W, H)})
            tr["observations"].append(obs); tr["missing"] = 0
        # Let tracks survive short occlusions; close longer interior gaps.
        for tid, tr in list(active.items()):
            if tid in used_t:
                continue
            tr["missing"] += 1
            if tr["missing"] > max_missing:
                _close_track(tr, closed, events, W, H); closed.append(tr); active.pop(tid, None)
        # Re-identify unmatched objects against recently closed tracks when
        # appearance and size agree.  A new unmatched interior object is only a
        # birth; duplication is decided after temporal coexistence is checked.
        for oi, ob in enumerate(objects):
            if oi in used_o:
                continue
            best: Optional[Tuple[float, Dict[str, Any]]] = None
            for tr in reversed(closed[-30:]):
                if tr.get("reidentified") or not tr.get("observations"):
                    continue
                last = tr["observations"][-1]
                if local_i - last["local_frame"] > 8:
                    continue
                dist = math.hypot(ob["center"][0] - last["center"][0], ob["center"][1] - last["center"][1])
                sim = float(np.dot(ob["hist"], last["hist"]))
                size_ratio = ob["area"] / max(last["area"], 1e-5)
                if dist <= .22 and sim >= .58 and .45 <= size_ratio <= 2.2:
                    score = dist - .05 * sim
                    if best is None or score < best[0]:
                        best = (score, tr)
            if best is not None:
                tr = best[1]; tr["reidentified"] = True; tr["missing"] = 0
                obs = dict(ob); obs.update({"track_id": tr["id"], "frame": source_i, "local_frame": local_i,
                                            "time": source_i / max(fps, 1e-6),
                                            "compensated_center": _compensated_normalized_center(
                                                cumulative_inv[local_i], ob["center"], W, H)})
                tr["observations"].append(obs); active[tr["id"]] = tr
                events["reidentifications"].append({"track_id": tr["id"], "frame": source_i})
                continue
            tr = {"id": next_id, "observations": [], "missing": 0, "reidentified": False,
                  "closed": False, "birth_frame": source_i}
            obs = dict(ob); obs.update({"track_id": next_id, "frame": source_i, "local_frame": local_i,
                                        "time": source_i / max(fps, 1e-6),
                                        "compensated_center": _compensated_normalized_center(
                                            cumulative_inv[local_i], ob["center"], W, H)})
            tr["observations"].append(obs); active[next_id] = tr
            events["births"].append({"track_id": next_id, "frame": source_i, "boundary": ob["boundary"]})
            next_id += 1

        # Record frame-level IDs and masks after association.
        frame_objects = []
        for tid, tr in active.items():
            if tr["observations"][-1]["local_frame"] == local_i:
                ob = tr["observations"][-1]
                frame_objects.append({k: v for k, v in ob.items() if k not in {"hist"}})
        frame_records.append({"frame": source_i, "time": source_i / max(fps, 1e-6),
                              "objects": frame_objects, "raw_count": len(objects),
                              "active_count": len(frame_objects), "camera": camera[local_i]})

    for tr in list(active.values()):
        _close_track(tr, closed, events, W, H); closed.append(tr)

    # Mark duplicate candidates only when two similar objects coexist for at
    # least three sampled frames and the second birth is interior.  This avoids
    # calling a boundary re-entry or a one-frame segmentation fragment a copy.
    for birth in events["births"]:
        if birth["boundary"]:
            continue
        new = next((t for t in closed if t["id"] == birth["track_id"]), None)
        if not new or len(new["observations"]) < 3:
            continue
        first_new = new["observations"][0]
        for old in closed:
            if old is new or not old["observations"]:
                continue
            coexist = {o["local_frame"] for o in old["observations"]} & {o["local_frame"] for o in new["observations"]}
            if len(coexist) < 3:
                continue
            old_near = min(old["observations"], key=lambda o: abs(o["local_frame"] - first_new["local_frame"]))
            sim = float(np.dot(first_new["hist"], old_near["hist"]))
            ratio = first_new["area"] / max(old_near["area"], 1e-5)
            if sim >= .60 and .45 <= ratio <= 2.2:
                events["duplicates"].append({"source_track": old["id"], "duplicate_track": new["id"],
                                              "frame": first_new["frame"], "coexist_frames": len(coexist),
                                              "appearance_similarity": sim})
                break

    # Motion jumps use camera-compensated coordinates and a robust per-track
    # baseline.  We also retain the uncorrected count so an audit can tell
    # whether a candidate is introduced by camera motion or by association.
    raw_jump_events: List[Dict[str, Any]] = []
    for tr in closed:
        obs = tr["observations"]
        if len(obs) < 3:
            continue
        comp_steps = [math.hypot(obs[i]["compensated_center"][0] - obs[i - 1]["compensated_center"][0],
                                 obs[i]["compensated_center"][1] - obs[i - 1]["compensated_center"][1])
                      for i in range(1, len(obs))]
        raw_steps = [math.hypot(obs[i]["center"][0] - obs[i - 1]["center"][0],
                                obs[i]["center"][1] - obs[i - 1]["center"][1])
                     for i in range(1, len(obs))]
        med = float(np.median(comp_steps)) if comp_steps else 0.0
        threshold = max(.08, 4.0 * med)
        raw_med = float(np.median(raw_steps)) if raw_steps else 0.0
        raw_threshold = max(.08, 4.0 * raw_med)
        for i, step in enumerate(comp_steps, 1):
            if step > threshold:
                events["jumps"].append({"track_id": tr["id"], "frame": obs[i]["frame"],
                                        "jump": step, "threshold": threshold})
            if raw_steps[i - 1] > raw_threshold:
                raw_jump_events.append({"track_id": tr["id"], "frame": obs[i]["frame"],
                                        "jump": raw_steps[i - 1], "threshold": raw_threshold})

    valid_tracks = [tr for tr in closed if len(tr["observations"]) >= 2]
    life = [(tr["observations"][-1]["frame"] - tr["observations"][0]["frame"]) / max(fps, 1e-6)
            for tr in valid_tracks]
    raw_counts = [f["raw_count"] for f in frame_records]
    active_counts = [f["active_count"] for f in frame_records]
    return {
        "video": os.path.abspath(path), "fps": fps, "width": W, "height": H,
        "scale": scale, "frame_stride": frame_stride, "frames": frame_records,
        "mask_source": mask_source,
        "tracks": [{"id": tr["id"], "birth_frame": tr["birth_frame"],
                    "observations": [{k: v for k, v in o.items() if k != "hist"} for o in tr["observations"]],
                    "life_s": (tr["observations"][-1]["frame"] - tr["observations"][0]["frame"]) / max(fps, 1e-6),
                    "boundary_start": tr["observations"][0].get("boundary", False),
                    "boundary_end": tr["observations"][-1].get("boundary", False)}
                   for tr in valid_tracks],
        "events": events,
        "raw_jump_events": raw_jump_events,
        "camera": camera,
        "counts": {"raw": raw_counts, "active": active_counts,
                   "raw_range": max(raw_counts, default=0) - min(raw_counts, default=0),
                   "active_range": max(active_counts, default=0) - min(active_counts, default=0)},
        "summary": {
            "track_count": len(valid_tracks), "births": len(events["births"]),
            "interior_deaths": len(events["deaths"]), "reidentifications": len(events["reidentifications"]),
            "duplication_candidates": len(events["duplicates"]), "jump_candidates": len(events["jumps"]),
            "raw_jump_candidates": len(raw_jump_events),
            "appearance_switches": len(events["appearance_switches"]),
            "geometry_jumps": len(events["geometry_jumps"]),
            "max_life_s": max(life, default=0.0), "median_life_s": float(np.median(life)) if life else 0.0,
        },
    }


def summary_from_tracks(result: Dict[str, Any]) -> Dict[str, float]:
    """Map detailed tracks to the feature names consumed by the classifier."""
    s = result.get("summary", {})
    c = result.get("counts", {})
    return {
        "track_count": float(s.get("track_count", 0)), "track_max_life": float(s.get("max_life_s", 0)),
        "track_median_life": float(s.get("median_life_s", 0)),
        "track_life_coverage": float(sum(t.get("life_s", 0) for t in result.get("tracks", [])) /
                                      max((result.get("frames") or [{}])[-1].get("time", 1.0), 1e-6)),
        "semantic_births": float(s.get("births", 0)), "semantic_deaths": float(s.get("interior_deaths", 0)),
        "semantic_reidentifications": float(s.get("reidentifications", 0)),
        "semantic_duplication_candidates": float(s.get("duplication_candidates", 0)),
        "semantic_jump_candidates": float(s.get("jump_candidates", 0)),
        "semantic_raw_jump_candidates": float(s.get("raw_jump_candidates", 0)),
        "semantic_appearance_switches": float(s.get("appearance_switches", 0)),
        "semantic_geometry_jumps": float(s.get("geometry_jumps", 0)),
        "semantic_count_range": float(c.get("active_range", 0)),
        "semantic_count_variance": float(np.var(c.get("active", [0]))),
    }
