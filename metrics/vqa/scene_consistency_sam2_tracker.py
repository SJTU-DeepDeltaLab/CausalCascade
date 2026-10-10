#!/usr/bin/env python3
"""SAM2-backed object evidence for Scene consistency.

This module is the semantic-mask backend that the earlier RGB proposal tracker
was missing.  It automatically segments compact first-frame instances with a
dense SAM2 point grid, propagates those masks with SAM2 Video, and computes
auditable lifecycle, trajectory, geometry, appearance, and camera evidence.

It intentionally does not contain an LLM or a learned task classifier.  The
output is deterministic visual evidence that can be inspected and then fed to
a small rule/logistic model.  Generated videos in this benchmark start from a
fully visible conditioning frame, making first-frame initialization the right
tracking prior.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


MODEL_ID = "facebook/sam2.1-hiera-tiny"


def _palette(index: int) -> Tuple[int, int, int]:
    colours = [
        (52, 211, 153), (251, 146, 60), (192, 132, 252), (56, 189, 248),
        (248, 113, 113), (250, 204, 21), (45, 212, 191), (244, 114, 182),
        (163, 230, 53), (129, 140, 248), (251, 191, 36), (34, 211, 238),
        (244, 63, 94), (167, 139, 250), (74, 222, 128), (96, 165, 250),
    ]
    return colours[index % len(colours)]


def _write_tracking_artifacts(
    frames: Sequence[np.ndarray], masks: np.ndarray, source_indices: Sequence[int],
    tracks: Sequence[Dict[str, Any]], output_dir: Path,
) -> Dict[str, str]:
    """Write masks and a compact visual audit without changing evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "masks.npz", masks=masks.astype(np.uint8),
                        source_indices=np.asarray(source_indices, dtype=np.int32))
    selected = sorted(set(np.linspace(0, len(frames) - 1, min(9, len(frames))).round().astype(int)))
    rendered: List[np.ndarray] = []
    H, W = frames[0].shape[:2]
    for fi in selected:
        canvas = frames[fi].copy()
        overlay = canvas.copy()
        for obj_id in range(masks.shape[1]):
            mask = masks[fi, obj_id]
            if not mask.any():
                continue
            colour = _palette(obj_id)
            overlay[mask] = colour
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, colour, 2, cv2.LINE_AA)
            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(canvas, f"ID {obj_id}", (max(2, cx - 18), max(14, cy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1, cv2.LINE_AA)
            trail = []
            for obs in tracks[obj_id].get("observations", [])[:fi + 1]:
                x, y = obs.get("center", [float("nan"), float("nan")])
                if math.isfinite(x) and math.isfinite(y):
                    trail.append((int(x * W), int(y * H)))
            if len(trail) >= 2:
                cv2.polylines(canvas, [np.asarray(trail, dtype=np.int32)], False,
                              colour, 1, cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, .28, canvas, .72, 0)
        cv2.rectangle(canvas, (0, 0), (W, 25), (20, 20, 20), -1)
        cv2.putText(canvas, f"sample {fi} / source frame {source_indices[fi]}",
                    (7, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        rendered.append(canvas)
    cell_w = 360
    resized = [cv2.resize(x, (cell_w, max(1, round(x.shape[0] * cell_w / x.shape[1]))))
               for x in rendered]
    cell_h = max(x.shape[0] for x in resized)
    cols = 3
    rows = math.ceil(len(resized) / cols)
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 24, dtype=np.uint8)
    for i, image in enumerate(resized):
        y, x = (i // cols) * cell_h, (i % cols) * cell_w
        sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
    cv2.imwrite(str(output_dir / "mask_tracks.jpg"), sheet,
                [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return {"mask_archive": str((output_dir / "masks.npz").resolve()),
            "mask_track_sheet": str((output_dir / "mask_tracks.jpg").resolve())}


def read_video_uniform(path: str, max_frames: int = 17) -> Tuple[List[np.ndarray], List[int], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        total = len(frames)
        indices = np.linspace(0, max(total - 1, 0), min(max_frames, total)).round().astype(int).tolist()
        return [frames[i] for i in indices], indices, fps
    indices = np.linspace(0, total - 1, min(max_frames, total)).round().astype(int).tolist()
    indices = sorted(set(indices))
    frames: List[np.ndarray] = []
    wanted = set(indices)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames.append(frame)
        i += 1
    cap.release()
    if len(frames) != len(indices):
        raise RuntimeError(f"decoded {len(frames)} of {len(indices)} requested frames: {path}")
    return frames, indices, fps


def _mask_properties(mask: np.ndarray) -> Dict[str, float]:
    H, W = mask.shape
    area_px = int(mask.sum())
    if area_px == 0:
        return {"area": 0.0, "fill": 0.0, "aspect": 0.0, "boundary_fraction": 1.0,
                "largest_component_fraction": 0.0}
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    border = np.zeros_like(mask, dtype=bool)
    margin = max(2, int(round(min(H, W) * .008)))
    border[:margin] = True; border[-margin:] = True
    border[:, :margin] = True; border[:, -margin:] = True
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if num > 1 else 0
    return {
        "area": float(area_px / (H * W)),
        "fill": float(area_px / max(bw * bh, 1)),
        "aspect": float(max(bw / max(bh, 1), bh / max(bw, 1))),
        "boundary_fraction": float((mask & border).sum() / area_px),
        "largest_component_fraction": float(largest / area_px),
        "bbox_x": float(x0 / W), "bbox_y": float(y0 / H),
        "bbox_w": float(bw / W), "bbox_h": float(bh / H),
    }


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    return float(inter / max(int(np.logical_or(a, b).sum()), 1))


def _first_frame_prompt_points(frame_bgr: np.ndarray, grid_x: int, grid_y: int,
                               max_proposals: int = 64) -> List[List[List[float]]]:
    """Combine coverage points with CV proposals for tiny synthetic objects.

    A uniform SAM grid alone can step over a 4--8 px marble.  Colour-component,
    circle, and compact-contour centres are therefore placed before the grid.
    The prompts do not assign an identity; they only ensure that SAM sees small
    candidates which can subsequently be filtered by mask quality.
    """
    H, W = frame_bgr.shape[:2]
    proposals: List[Tuple[float, float, float]] = []
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    saturated = ((hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 45)).astype(np.uint8)
    saturated = cv2.morphologyEx(saturated, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, _, stats, centres = cv2.connectedComponentsWithStats(saturated, 8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 6 <= area <= .04 * H * W and w <= .20 * W and h <= .20 * H:
            cx, cy = map(float, centres[i])
            proposals.append((3.0 + min(area / 500.0, 1.0), cx, cy))

    # Tiny dark balls are common triggers and can fall between grid points.
    # Large dark furniture/background components are rejected by size.
    dark = (gray <= min(80, int(np.percentile(gray, 22)))).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, _, stats, centres = cv2.connectedComponentsWithStats(dark, 8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 5 <= area <= .008 * H * W and w <= .10 * W and h <= .10 * H:
            cx, cy = map(float, centres[i])
            proposals.append((5.0 + min(area / 300.0, .5), cx, cy))

    # Compact edge contours recover dark/low-saturation balls and ornaments.
    edge = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 120)
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = abs(cv2.contourArea(contour)); x, y, w, h = cv2.boundingRect(contour)
        if not (5 <= area <= .02 * H * W and 3 <= w <= .12 * W and 3 <= h <= .12 * H):
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > 4.0:
            continue
        moment = cv2.moments(contour)
        if abs(moment["m00"]) > 1e-6:
            perimeter = max(cv2.arcLength(contour, True), 1e-6)
            circularity = min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))
            proposals.append((1.0 + circularity + min(area / 1000.0, .5),
                              moment["m10"] / moment["m00"],
                              moment["m01"] / moment["m00"]))

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=max(5, round(min(H, W) * .012)),
                               param1=80, param2=13,
                               minRadius=max(2, round(min(H, W) * .004)),
                               maxRadius=max(5, round(min(H, W) * .06)))
    if circles is not None:
        proposals.extend((4.0, float(x), float(y)) for x, y, _ in circles[0][:32])

    # Keep a bounded proposal set.  This avoids thousands of compression-edge
    # contours while retaining circles and coherent colour components first.
    kept: List[Tuple[float, float]] = []
    radius2 = max(3.0, min(H, W) * .006) ** 2
    for _, x, y in sorted(proposals, reverse=True):
        if x < 1 or x >= W - 1 or y < 1 or y >= H - 1:
            continue
        if any((x - ox) ** 2 + (y - oy) ** 2 < radius2 for ox, oy in kept):
            continue
        kept.append((x, y))
        if len(kept) >= max_proposals:
            break
    # The lower-resolution grid guarantees coverage when CV proposals miss an
    # object.  Proposal points remain first so prompt truncation is predictable.
    for y in np.linspace(.025, .975, grid_y):
        for x in np.linspace(.02, .98, grid_x):
            px, py = float(x * W), float(y * H)
            if any((px - ox) ** 2 + (py - oy) ** 2 < radius2 for ox, oy in kept):
                continue
            kept.append((px, py))
    return [[[x, y]] for x, y in kept]


def _foreground_rank_features(frame_bgr: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Return colourfulness and local colour contrast for candidate ranking."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    saturation = float(np.mean(hsv[:, :, 1][mask]) / 255.0) if mask.any() else 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ring = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~mask
    if ring.any() and mask.any():
        contrast = float(np.linalg.norm(lab[mask].mean(axis=0) - lab[ring].mean(axis=0)) / 255.0)
    else:
        contrast = 0.0
    return saturation, contrast


def _instance_descriptor(frame_bgr: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    props = _mask_properties(mask)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8),
                        [12, 8], [0, 180, 0, 256]).ravel().astype(np.float32)
    hist /= max(float(np.linalg.norm(hist)), 1e-8)
    ys, xs = np.where(mask)
    return {"hist": hist, "area": props["area"], "aspect": props["aspect"],
            "center": [float(xs.mean() / mask.shape[1]), float(ys.mean() / mask.shape[0])],
            "boundary_fraction": props["boundary_fraction"]}


def _descriptor_match(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    area_ratio = min(a["area"], b["area"]) / max(a["area"], b["area"], 1e-8)
    aspect_ratio = min(a["aspect"], b["aspect"]) / max(a["aspect"], b["aspect"], 1e-8)
    appearance = float(np.dot(a["hist"], b["hist"]))
    if area_ratio < .32 or aspect_ratio < .38:
        return 0.0
    return .68 * appearance + .20 * area_ratio + .12 * aspect_ratio


def _cluster_initial_instances(frame_bgr: np.ndarray,
                               initial: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[int]]:
    descriptors = [_instance_descriptor(frame_bgr, x["mask"]) for x in initial]
    clusters: List[Dict[str, Any]] = []
    assignments: List[int] = []
    for obj_id, descriptor in enumerate(descriptors):
        best = -1; best_score = 0.0
        for cluster_id, cluster in enumerate(clusters):
            score = max(_descriptor_match(descriptor, descriptors[j]) for j in cluster["members"])
            if score > best_score:
                best, best_score = cluster_id, score
        if best >= 0 and best_score >= .78:
            clusters[best]["members"].append(obj_id)
            assignments.append(best)
        else:
            assignments.append(len(clusters))
            clusters.append({"members": [obj_id]})
    for cluster_id, cluster in enumerate(clusters):
        cluster["id"] = cluster_id
        cluster["initial_count"] = len(cluster["members"])
        cluster["initial_instances"] = [
            {"object_id": j,
             "area": float(initial[j].get("area", descriptors[j]["area"])),
             "aspect": float(initial[j].get("aspect", descriptors[j]["aspect"])),
             "saturation": float(initial[j].get("saturation", 0.0)),
             "local_contrast": float(initial[j].get("local_contrast", 0.0)),
             "rank": float(initial[j].get("rank", 0.0)),
             "center": descriptors[j]["center"]}
            for j in cluster["members"]
        ]
    return clusters, assignments


def _sparse_redetection_evidence(
    backend: "Sam2ObjectEvidence", frames: Sequence[np.ndarray],
    initial_all: Sequence[Dict[str, Any]], max_objects: int = 30,
) -> Dict[str, Any]:
    """Re-detect a few frames to expose new instances missed by propagation."""
    if len(frames) <= 1:
        return {"frames": [], "persistent_duplication_clusters": [],
                "duplication_frame_count": 0, "max_positive_count_delta": 0}
    initial_descriptors = [_instance_descriptor(frames[0], x["mask"]) for x in initial_all]
    clusters, assignments = _cluster_initial_instances(frames[0], initial_all)
    # Include early frames because short-lived duplication often occurs within
    # the first two seconds, plus middle and terminal evidence.
    positions = sorted(set([min(2, len(frames) - 1), len(frames) // 2,
                            max(1, len(frames) - 2), len(frames) - 1]))
    records: List[Dict[str, Any]] = []
    duplication_by_cluster: Dict[int, int] = {}
    for fi in positions:
        current = backend.automatic_first_frame_masks(frames[fi], max_objects=max_objects)
        counts = [0 for _ in clusters]
        unmatched = 0
        assignments_now = []
        for candidate in current:
            descriptor = _instance_descriptor(frames[fi], candidate["mask"])
            best_obj = -1; best_score = 0.0
            for obj_id, initial_descriptor in enumerate(initial_descriptors):
                score = _descriptor_match(descriptor, initial_descriptor)
                if score > best_score:
                    best_obj, best_score = obj_id, score
            if best_obj >= 0 and best_score >= .72:
                cluster_id = assignments[best_obj]
                counts[cluster_id] += 1
                assignments_now.append({"cluster": cluster_id, "score": best_score,
                                        "center": descriptor["center"],
                                        "area": descriptor["area"],
                                        "aspect": descriptor["aspect"]})
            else:
                unmatched += 1
                assignments_now.append({"cluster": None, "score": best_score,
                                        "center": descriptor["center"],
                                        "area": descriptor["area"],
                                        "aspect": descriptor["aspect"]})
        deltas = [counts[i] - int(cluster["initial_count"])
                  for i, cluster in enumerate(clusters)]
        duplicate_clusters = []
        for cluster_id, delta in enumerate(deltas):
            # Repeated structures tolerate one segmentation-count fluctuation;
            # unique/paired instances require one genuinely additional mask.
            tolerance = 1 if clusters[cluster_id]["initial_count"] >= 4 else 0
            if delta > tolerance:
                duplicate_clusters.append(cluster_id)
                duplication_by_cluster[cluster_id] = duplication_by_cluster.get(cluster_id, 0) + 1
        records.append({"sample_frame": fi, "detected_instances": len(current),
                        "cluster_counts": counts, "cluster_deltas": deltas,
                        "duplicate_clusters": duplicate_clusters,
                        "unmatched_instances": unmatched,
                        "assignments": assignments_now})
    persistent = sorted(k for k, v in duplication_by_cluster.items() if v >= 2)
    duplication_frames = sum(any(k in persistent for k in x["duplicate_clusters"])
                             for x in records)
    return {
        "frames": records,
        "clusters": clusters,
        "persistent_duplication_clusters": persistent,
        "duplication_frame_count": int(duplication_frames),
        "max_positive_count_delta": int(max(
            [max(x["cluster_deltas"], default=0) for x in records], default=0)),
    }


def _first_frame_alignment(generated: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Estimate generated-first-frame to reference-first-frame affine mapping."""
    H, W = reference.shape[:2]
    resized = cv2.resize(generated, (W, H), interpolation=cv2.INTER_AREA)
    # ECC handles the small crop/aspect-ratio difference between generated and
    # simulator videos better than a similarity-only feature transform.
    scale = min(1.0, 960.0 / max(W, H))
    work_size = (max(32, round(W * scale)), max(32, round(H * scale)))
    input_gray = cv2.resize(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), work_size).astype(np.float32) / 255.0
    template_gray = cv2.resize(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), work_size).astype(np.float32) / 255.0
    warp_ref_to_gen = np.eye(2, 3, dtype=np.float32)
    try:
        score, warp_ref_to_gen = cv2.findTransformECC(
            template_gray, input_gray, warp_ref_to_gen, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-5),
            None, 5)
        if score >= .82:
            warp_ref_to_gen[:, 2] /= scale
            return cv2.invertAffineTransform(warp_ref_to_gen).astype(np.float32)
    except cv2.error:
        pass
    a = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2500)
    ka, da = orb.detectAndCompute(a, None); kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return np.eye(2, 3, dtype=np.float32)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in matches if m.distance < .72 * n.distance]
    if len(good) < 12:
        return np.eye(2, 3, dtype=np.float32)
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    M, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                      ransacReprojThreshold=2.5)
    if M is None or inliers is None or float(inliers.mean()) < .45:
        return np.eye(2, 3, dtype=np.float32)
    return M.astype(np.float32)


def _reference_activity_scores(initial: Sequence[Dict[str, Any]], generated_first: np.ndarray,
                               gt_video: str, max_frames: int) -> List[float]:
    """Estimate which first-frame instances participate in the reference motion.

    The reference is used only to establish object roles, not to decide whether
    generated motion is correct.  This prevents static coloured floor markings
    and decorations from dominating trajectory evidence.
    """
    gt_frames, _, _ = read_video_uniform(gt_video, max_frames=max_frames)
    first = cv2.cvtColor(gt_frames[0], cv2.COLOR_BGR2LAB).astype(np.float32)
    H, W = first.shape[:2]
    alignment = _first_frame_alignment(generated_first, gt_frames[0])
    scores: List[float] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for item in initial:
        mask = item["mask"]
        mask = cv2.resize(mask.astype(np.uint8), (W, H),
                          interpolation=cv2.INTER_NEAREST)
        mask = cv2.warpAffine(mask, alignment, (W, H), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT).astype(bool)
        ring = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~mask
        changes = []
        for frame in gt_frames[1:]:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            delta = np.linalg.norm(lab - first, axis=2) / 255.0
            object_change = float(np.median(delta[mask])) if mask.any() else 0.0
            local_change = float(np.median(delta[ring])) if ring.any() else 0.0
            changes.append(max(0.0, object_change - .6 * local_change))
        scores.append(float(max(changes, default=0.0)))
    return scores


class Sam2ObjectEvidence:
    def __init__(self, model_id: str = MODEL_ID, device: Optional[str] = None):
        import torch
        # The CPU nodes use torch 2.1, whereas recent Transformers expects the
        # torch 2.2 spelling.  The underlying pytree implementation is already
        # present under the legacy name, so bridge it before importing SAM2.
        try:
            from torch.utils import _pytree
            if not hasattr(_pytree, "register_pytree_node") and hasattr(
                    _pytree, "_register_pytree_node"):
                def _compat_register(typ, flatten_fn, unflatten_fn, **kwargs):
                    # ``serialized_type_name`` was added after torch 2.1;
                    # registration itself is otherwise identical.
                    kwargs.pop("serialized_type_name", None)
                    return _pytree._register_pytree_node(
                        typ, flatten_fn, unflatten_fn, **kwargs)
                _pytree.register_pytree_node = _compat_register
        except Exception:
            pass
        if not hasattr(torch, "compiler"):
            class _CompilerCompat:
                @staticmethod
                def is_compiling() -> bool:
                    return False
            torch.compiler = _CompilerCompat()
        elif not hasattr(torch.compiler, "is_compiling"):
            torch.compiler.is_compiling = lambda: False
        from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor
        # torchvision shipped with torch 2.1 lacks the v2 helper used by the
        # newer SAM2 video preprocessor.  RGB benchmark frames do not need this
        # path, but providing the equivalent singleton-channel conversion keeps
        # the processor compatible on the CPU nodes.
        try:
            from torchvision.transforms.v2 import functional as tvf
            if not hasattr(tvf, "grayscale_to_rgb"):
                def _gray_to_rgb(value):
                    if getattr(value, "shape", None) is not None and value.shape[-3] == 1:
                        return value.repeat_interleave(3, dim=-3)
                    return value
                tvf.grayscale_to_rgb = _gray_to_rgb
        except Exception:
            pass

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.image_processor = Sam2Processor.from_pretrained(model_id)
        # Image auto-mask generation is faster and more memory-stable on CPU
        # for many point prompts. Video propagation uses MPS when available.
        self.image_model = Sam2Model.from_pretrained(model_id).eval()
        self.video_processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.video_model = Sam2VideoModel.from_pretrained(model_id).to(self.device).eval()

    def automatic_first_frame_masks(
        self,
        frame_bgr: np.ndarray,
        grid_x: int = 24,
        grid_y: int = 14,
        chunk: int = 24,
        score_threshold: float = .78,
        max_objects: int = 16,
    ) -> List[Dict[str, Any]]:
        from PIL import Image

        torch = self.torch
        H, W = frame_bgr.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        base = self.image_processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            embeddings = self.image_model.get_image_embeddings(base["pixel_values"])
        points = _first_frame_prompt_points(frame_bgr, grid_x, grid_y)
        candidates: List[Dict[str, Any]] = []
        # Keep a small pool of lower-confidence proposals for scenes whose
        # first frame contains only thin, low-contrast supports or rails.  The
        # strict filter below is appropriate for ordinary objects, but it can
        # reject every valid proposal in these sparse synthetic scenes.
        relaxed_candidates: List[Dict[str, Any]] = []
        for start in range(0, len(points), chunk):
            subset = points[start:start + chunk]
            prompts = self.image_processor(
                input_points=[subset], input_labels=[[[1] for _ in subset]],
                original_sizes=[[H, W]], return_tensors="pt")
            with torch.inference_mode():
                output = self.image_model(
                    image_embeddings=embeddings,
                    input_points=prompts["input_points"],
                    input_labels=prompts["input_labels"],
                    multimask_output=True,
                )
            masks = self.image_processor.post_process_masks(
                output.pred_masks, base["original_sizes"])[0].cpu().numpy()
            scores = output.iou_scores[0].cpu().numpy()
            for i in range(masks.shape[0]):
                best = int(np.argmax(scores[i]))
                mask = masks[i, best].astype(bool)
                props = _mask_properties(mask)
                score = float(scores[i, best])
                saturation, contrast = _foreground_rank_features(frame_bgr, mask)
                broad_ok = (
                    score >= .45 and .00001 <= props["area"] <= .25 and
                    props["boundary_fraction"] <= .45 and props["aspect"] <= 15.0 and
                    props["fill"] >= .04 and props["largest_component_fraction"] >= .35
                )
                if broad_ok:
                    relaxed_candidates.append({
                        "mask": mask, "sam_iou": score,
                        "saturation": saturation, "local_contrast": contrast,
                        **props,
                    })
                    # A single point can yield several near-duplicate masks;
                    # retaining the strongest bounded pool avoids a large
                    # first-frame memory spike on CPU.
                    if len(relaxed_candidates) > 128:
                        relaxed_candidates.sort(key=lambda x: x["sam_iou"], reverse=True)
                        del relaxed_candidates[128:]
                # Compact, non-border instances retain balls, cubes, dominoes,
                # small ornaments, supports, and rigid scene elements while
                # rejecting table/background/sofa regions.
                if score < score_threshold:
                    continue
                if not (.00004 <= props["area"] <= .035):
                    continue
                if props["boundary_fraction"] > .18 or props["aspect"] > 5.0:
                    continue
                if props["fill"] < .12 or props["largest_component_fraction"] < .82:
                    continue
                compact_bonus = min(.08, .004 / max(props["area"], 1e-6) * .01)
                # Distinct, compact foreground objects should outrank repeated
                # low-contrast pegs even when SAM's IoU estimate is modest.
                salient_bonus = (.25 if saturation > .35 and contrast > .15
                                  and props["area"] > .0015 else 0.0)
                rank = (score + .08 * props["fill"] - .04 * props["aspect"]
                        + .14 * saturation + .16 * contrast + compact_bonus
                        + salient_bonus)
                candidates.append({"mask": mask, "sam_iou": score, "rank": rank,
                                   "saturation": saturation, "local_contrast": contrast,
                                   **props})
        # Suppress duplicate point prompts for the same object. A containment
        # test complements IoU because SAM may return a precise object mask and
        # a slightly larger region with modest IoU.
        kept: List[Dict[str, Any]] = []
        for cand in sorted(candidates, key=lambda x: x["rank"], reverse=True):
            duplicate = False
            for old in kept:
                inter = int(np.logical_and(cand["mask"], old["mask"]).sum())
                containment = inter / max(min(int(cand["mask"].sum()), int(old["mask"].sum())), 1)
                if _mask_iou(cand["mask"], old["mask"]) >= .68 or containment >= .88:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(cand)
            if len(kept) >= max_objects:
                break
        # Fallback for sparse structural scenes (e.g. a rail course) where
        # SAM returns valid but elongated/low-contrast masks and the strict
        # object filter removes all of them.  We still require a reasonable
        # SAM score, non-border support, and a compact connected component;
        # this is not a blind "use the whole frame" fallback.
        if not kept and relaxed_candidates:
            relaxed_candidates.sort(
                key=lambda x: (x["sam_iou"] + .12 * x["saturation"]
                               + .14 * x["local_contrast"] + .06 * x["fill"]),
                reverse=True)
            for cand in relaxed_candidates:
                duplicate = False
                for old in kept:
                    inter = int(np.logical_and(cand["mask"], old["mask"]).sum())
                    containment = inter / max(
                        min(int(cand["mask"].sum()), int(old["mask"].sum())), 1)
                    if (_mask_iou(cand["mask"], old["mask"]) >= .60
                            or containment >= .82):
                        duplicate = True
                        break
                if not duplicate:
                    cand["rank"] = (cand["sam_iou"] + .12 * cand["saturation"]
                                     + .14 * cand["local_contrast"] + .06 * cand["fill"])
                    kept.append(cand)
                if len(kept) >= max_objects:
                    break
        return kept

    def propagate(self, frames_bgr: Sequence[np.ndarray], initial: Sequence[Dict[str, Any]]) -> np.ndarray:
        import torch.nn.functional as F

        torch = self.torch
        rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
        H, W = frames_bgr[0].shape[:2]
        session = self.video_processor.init_video_session(
            video=rgb,
            inference_device=self.device,
            processing_device=self.device,
            inference_state_device=self.device,
            video_storage_device="cpu" if self.device == "mps" else self.device,
            dtype=torch.float32,
        )
        obj_ids = list(range(len(initial)))
        self.video_processor.add_inputs_to_inference_session(
            session, frame_idx=0, obj_ids=obj_ids,
            input_masks=[x["mask"] for x in initial])
        masks_by_frame: List[np.ndarray] = []
        with torch.inference_mode():
            self.video_model(session, frame_idx=0)
            for output in self.video_model.propagate_in_video_iterator(session, start_frame_idx=0):
                logits = F.interpolate(output.pred_masks.float(), size=(H, W),
                                       mode="bilinear", align_corners=False)
                masks_by_frame.append((logits[:, 0] > 0).cpu().numpy())
        result = np.stack(masks_by_frame, axis=0)
        del session
        if self.device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        return result


def _background_affines(frames: Sequence[np.ndarray], masks: np.ndarray) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
    """Estimate pairwise camera motion using only pixels outside object masks."""
    H, W = frames[0].shape[:2]
    gray = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in frames]
    cumulative = np.eye(3, dtype=np.float64)
    inverse = [np.eye(3, dtype=np.float64)]
    diagnostics = [{"motion": 0.0, "translation": 0.0, "rotation_scale": 0.0,
                    "residual": 0.0, "inlier": 1.0, "valid": 1.0}]
    orb = cv2.ORB_create(nfeatures=800)
    for i in range(1, len(frames)):
        excluded = np.logical_or(masks[i - 1].any(axis=0), masks[i].any(axis=0))
        bgmask = (~excluded).astype(np.uint8) * 255
        kp0, des0 = orb.detectAndCompute(gray[i - 1], bgmask)
        kp1, des1 = orb.detectAndCompute(gray[i], bgmask)
        M = None; residual = 0.0; inlier_fraction = 0.0
        if des0 is not None and des1 is not None and len(kp0) >= 8 and len(kp1) >= 8:
            try:
                matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des0, des1, k=2)
                good = [m for m, n in matches if m.distance < .75 * n.distance]
            except Exception:
                good = []
            if len(good) >= 8:
                a = np.float32([kp0[x.queryIdx].pt for x in good])
                b = np.float32([kp1[x.trainIdx].pt for x in good])
                M2, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                          ransacReprojThreshold=3.0)
                if M2 is not None and inliers is not None and inliers.any():
                    pred = cv2.transform(a[None], M2)[0]
                    rr = np.sqrt(((pred - b) ** 2).sum(axis=1))
                    valid = inliers.ravel().astype(bool)
                    residual = float(np.median(rr[valid]))
                    inlier_fraction = float(valid.mean())
                    M = np.eye(3, dtype=np.float64); M[:2] = M2
        if M is None:
            M = np.eye(3, dtype=np.float64)
            valid_value = 0.0
        else:
            valid_value = 1.0
        cumulative = M @ cumulative
        try:
            inverse.append(np.linalg.inv(cumulative))
        except np.linalg.LinAlgError:
            inverse.append(np.eye(3, dtype=np.float64))
        rotation = float(np.linalg.norm(M[:2, :2] - np.eye(2)))
        translation = math.hypot(float(M[0, 2]) / max(W, 1), float(M[1, 2]) / max(H, 1))
        diagnostics.append({"motion": float(math.hypot(rotation, translation)),
                            "translation": translation, "rotation_scale": rotation,
                            "residual": residual, "inlier": inlier_fraction,
                            "valid": valid_value})
    return inverse, diagnostics


def _transform_center(M: np.ndarray, center: Sequence[float], W: int, H: int) -> List[float]:
    p = np.array([center[0] * W, center[1] * H, 1.0], dtype=np.float64)
    q = M @ p
    return [float(q[0] / max(q[2], 1e-9) / W), float(q[1] / max(q[2], 1e-9) / H)]


def extract_sam2_evidence(path: str, backend: Sam2ObjectEvidence,
                          max_frames: int = 17, max_objects: int = 16,
                          artifact_dir: Optional[Path] = None,
                          sparse_redetection: bool = False,
                          gt_path: Optional[str] = None,
                          reference_activity: bool = False) -> Dict[str, Any]:
    frames, source_indices, fps = read_video_uniform(path, max_frames=max_frames)
    initial_all = backend.automatic_first_frame_masks(
        frames[0], max_objects=max(30 if sparse_redetection else max_objects, max_objects))
    activity_scores = ([0.0 for _ in initial_all] if not (reference_activity and gt_path)
                       else _reference_activity_scores(initial_all, frames[0], gt_path, max_frames))
    for item, score in zip(initial_all, activity_scores):
        item["reference_activity_score"] = score
    if reference_activity:
        active = [x for x in initial_all if x["reference_activity_score"] >= .055]
        # Activity is primary; SAM/foreground rank breaks ties.  Retain at
        # least four objects so subtle reference changes are not discarded.
        active.sort(key=lambda x: (x["reference_activity_score"], x["rank"]), reverse=True)
        if len(active) < min(4, len(initial_all)):
            seen = {id(x) for x in active}
            active.extend(x for x in initial_all if id(x) not in seen)
        initial = active[:max_objects]
    else:
        initial = initial_all[:max_objects]
    if not initial:
        raise RuntimeError(f"SAM2 found no usable first-frame instances: {path}")
    masks = backend.propagate(frames, initial)
    H, W = frames[0].shape[:2]
    inv_camera, camera = _background_affines(frames, masks)
    tracks: List[Dict[str, Any]] = []
    all_jump_count = 0; all_duplication_frames = 0; all_death_events = 0
    appearance_events = 0; geometry_events = 0
    all_direction_reversals = 0; all_axis_reversals = 0; all_state_revisits = 0
    for obj_id in range(masks.shape[1]):
        observations = []
        initial_area = max(float(masks[0, obj_id].mean()), 1e-6)
        initial_hist = None
        for fi in range(len(frames)):
            mask = masks[fi, obj_id]
            props = _mask_properties(mask)
            if mask.any():
                ys, xs = np.where(mask)
                center = [float(xs.mean() / W), float(ys.mean() / H)]
                comp = _transform_center(inv_camera[fi], center, W, H)
                hsv = cv2.cvtColor(frames[fi], cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8),
                                    [8, 8], [0, 180, 0, 256]).ravel().astype(np.float32)
                hist /= max(float(np.linalg.norm(hist)), 1e-8)
            else:
                center = [float("nan"), float("nan")]
                comp = [float("nan"), float("nan")]
                hist = np.zeros(64, dtype=np.float32)
            if initial_hist is None:
                initial_hist = hist
            num, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
            components = [int(x) for x in stats[1:, cv2.CC_STAT_AREA]
                          if int(x) >= max(12, int(.12 * mask.sum()))]
            observations.append({
                "sample_frame": fi, "source_frame": source_indices[fi],
                "time_s": float(source_indices[fi] / max(fps, 1e-6)),
                "area": props["area"], "aspect": props["aspect"],
                "fill": props["fill"], "center": center,
                "compensated_center": comp,
                "visible": bool(props["area"] >= .18 * initial_area),
                "components": len(components),
                "appearance_similarity_to_initial": float(np.dot(initial_hist, hist)),
            })
        valid_obs = [x for x in observations if x["visible"]]
        steps = []
        for a, b in zip(valid_obs, valid_obs[1:]):
            if b["sample_frame"] != a["sample_frame"] + 1:
                continue
            steps.append(math.hypot(b["compensated_center"][0] - a["compensated_center"][0],
                                    b["compensated_center"][1] - a["compensated_center"][1]))
        median_step = float(np.median(steps)) if steps else 0.0
        mad_step = float(np.median(np.abs(np.asarray(steps) - median_step))) if steps else 0.0
        object_scale = math.sqrt(initial_area)
        jump_threshold = max(.045, median_step + 4.0 * mad_step, 1.5 * object_scale)
        jumps = int(sum(step > jump_threshold for step in steps))
        xy = np.asarray([x["compensated_center"] for x in valid_obs], dtype=np.float64)
        velocity = np.diff(xy, axis=0) if len(xy) >= 2 else np.empty((0, 2))
        speed = np.linalg.norm(velocity, axis=1) if len(velocity) else np.empty(0)
        direction_reversals = 0
        for va, vb, sa, sb in zip(velocity[:-1], velocity[1:], speed[:-1], speed[1:]):
            if min(sa, sb) < max(.012, .35 * median_step):
                continue
            cosine = float(np.dot(va, vb) / max(sa * sb, 1e-9))
            direction_reversals += int(cosine < -.35)
        axis_reversals = 0
        if len(velocity):
            ranges = np.ptp(xy, axis=0)
            axis = int(np.argmax(ranges))
            threshold = max(.01, .20 * median_step)
            signs = [int(np.sign(x)) for x in velocity[:, axis] if abs(x) >= threshold]
            axis_reversals = sum(a != b for a, b in zip(signs, signs[1:]))
        state_revisit = False
        if len(xy) >= 4:
            path_length = float(speed.sum())
            net = float(np.linalg.norm(xy[-1] - xy[0]))
            farthest = float(np.max(np.linalg.norm(xy - xy[0], axis=1)))
            state_revisit = bool(path_length > max(.12, 3.0 * object_scale)
                                 and farthest > 2.5 * object_scale
                                 and net < .45 * path_length)
        duplicate_frames = int(sum(x["components"] >= 2 for x in observations))
        # Disappearance requires persistence and an interior last observation.
        visible = [x["visible"] for x in observations]
        longest_absent = 0; run = 0
        for present in visible[1:]:
            run = 0 if present else run + 1
            longest_absent = max(longest_absent, run)
        last_visible = next((x for x in reversed(observations) if x["visible"]), observations[0])
        cx, cy = last_visible["center"]
        boundary_exit = bool(math.isfinite(cx) and (cx < .04 or cx > .96 or cy < .04 or cy > .96))
        death = bool(longest_absent >= 2 and not boundary_exit)
        similarities = [x["appearance_similarity_to_initial"] for x in observations[2:]]
        appearance = bool(sum(x < .45 for x in similarities) >= 2)
        areas = [x["area"] / initial_area for x in observations[1:] if x["visible"]]
        aspects = [x["aspect"] / max(observations[0]["aspect"], 1e-6)
                   for x in observations[1:] if x["visible"]]
        geometry = bool(sum((x < .42 or x > 2.4) for x in areas) >= 2 or
                        sum((x < .5 or x > 2.0) for x in aspects) >= 2)
        all_jump_count += jumps
        all_duplication_frames += duplicate_frames
        all_death_events += int(death)
        appearance_events += int(appearance)
        geometry_events += int(geometry)
        all_direction_reversals += direction_reversals
        all_axis_reversals += axis_reversals
        all_state_revisits += int(state_revisit)
        tracks.append({"id": obj_id, "initial": {k: v for k, v in initial[obj_id].items() if k != "mask"},
                       "observations": observations, "jump_count": jumps,
                       "jump_threshold": jump_threshold,
                       "direction_reversal_count": direction_reversals,
                       "dominant_axis_reversal_count": axis_reversals,
                       "state_revisit_candidate": state_revisit,
                       "duplicate_frame_count": duplicate_frames,
                       "death_candidate": death, "boundary_exit": boundary_exit,
                       "appearance_change_candidate": appearance,
                       "geometry_change_candidate": geometry})
    camera_valid = [x for x in camera[1:] if x["valid"]]
    redetection = (_sparse_redetection_evidence(backend, frames, initial_all)
                   if sparse_redetection else None)
    fusion_split_events: List[Dict[str, Any]] = []
    if redetection and redetection["frames"]:
        final_redetection = redetection["frames"][-1]
        unmatched_centres = [x["center"] for x in final_redetection["assignments"]
                             if x["cluster"] is None]
        for track in tracks:
            observations = [x for x in track["observations"] if x["visible"]]
            if len(observations) < 3 or track["initial"]["area"] < .005:
                continue
            initial_area = observations[0]["area"]
            last_ratios = [x["area"] / max(initial_area, 1e-8) for x in observations[-3:]]
            centre = observations[-1]["center"]
            nearby_unmatched = any(math.dist(centre, x) <= .12 for x in unmatched_centres)
            if sum(x < .82 for x in last_ratios) >= 2 and nearby_unmatched:
                fusion_split_events.append({"track_id": track["id"],
                                            "last_area_ratios": last_ratios,
                                            "nearby_unmatched": True,
                                            "sample_frame": final_redetection["sample_frame"]})
    result = {
        "video": os.path.abspath(path), "model_id": MODEL_ID,
        "ground_truth_video": os.path.abspath(gt_path) if gt_path else None,
        "reference_activity_guided": bool(reference_activity and gt_path),
        "mask_source": "sam2_dense_grid_first_frame_and_video_propagation",
        "fps": fps, "source_indices": source_indices, "width": W, "height": H,
        "n_initial_objects": len(initial), "tracks": tracks, "camera": camera,
        "summary": {
            "entity_death_candidates": all_death_events,
            "entity_duplication_frames": all_duplication_frames,
            "trajectory_jump_candidates": all_jump_count,
            "trajectory_direction_reversals": all_direction_reversals,
            "trajectory_axis_reversals": all_axis_reversals,
            "trajectory_state_revisit_candidates": all_state_revisits,
            "identity_appearance_candidates": appearance_events,
            "identity_geometry_candidates": geometry_events,
            "camera_motion_median": float(np.median([x["motion"] for x in camera_valid])) if camera_valid else 0.0,
            "camera_translation_median": float(np.median([x["translation"] for x in camera_valid])) if camera_valid else 0.0,
            "camera_residual_median": float(np.median([x["residual"] for x in camera_valid])) if camera_valid else 0.0,
            "camera_valid_fraction": float(len(camera_valid) / max(len(camera) - 1, 1)),
        },
    }
    if redetection is not None:
        result["sparse_redetection"] = redetection
        result["fusion_split_events"] = fusion_split_events
        result["summary"]["redetected_duplication_frames"] = redetection["duplication_frame_count"]
        result["summary"]["redetected_positive_count_delta"] = redetection["max_positive_count_delta"]
        result["summary"]["fusion_split_candidates"] = len(fusion_split_events)
    if artifact_dir is not None:
        result["artifacts"] = _write_tracking_artifacts(
            frames, masks, source_indices, tracks, artifact_dir)
    return result


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unit-id", action="append", default=[])
    parser.add_argument("--subdimension", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=17)
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--sparse-redetection", action="store_true")
    parser.add_argument("--reference-activity", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-id", default=MODEL_ID,
                        help="SAM2 model ID or a local snapshot directory.")
    args = parser.parse_args()
    rows = [x for x in _load_jsonl(args.manifest)
            if x.get("diagnostic_dimension") == "scene_consistency"]
    if args.unit_id:
        wanted = set(args.unit_id)
        rows = [x for x in rows if x["unit_id"] in wanted]
    if args.subdimension:
        wanted_subdimensions = set(args.subdimension)
        rows = [x for x in rows if x["diagnostic_subdimension"] in wanted_subdimensions]
    if args.max_cases:
        rows = rows[:args.max_cases]
    backend = Sam2ObjectEvidence(model_id=args.model_id, device=args.device)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            case_artifacts = None
            if args.artifact_dir:
                case_artifacts = Path(args.artifact_dir) / f"case_{i:02d}"
            evidence = extract_sam2_evidence(
                row["gen_video"], backend, max_frames=args.max_frames,
                max_objects=args.max_objects, artifact_dir=case_artifacts,
                sparse_redetection=args.sparse_redetection,
                gt_path=row.get("gt_video"), reference_activity=args.reference_activity)
            evidence.update({"unit_id": row["unit_id"],
                             "subdimension": row["diagnostic_subdimension"],
                             "model_name": row.get("model_name"),
                             "scene_id": row.get("scene_id")})
            f.write(json.dumps(evidence, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(rows)}] {row['unit_id']} {evidence['summary']}", flush=True)


if __name__ == "__main__":
    main()
