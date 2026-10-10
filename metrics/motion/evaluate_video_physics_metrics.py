#!/usr/bin/env python
"""Evaluate generated physics videos against ground-truth videos.

The generated video name is expected to encode its ground-truth path:

    generated_model/Instability_Series.ball_collision_support_collapse.mp4
    -> SortedDataset/Instability_Series/ball_collision_support_collapse/ball_collision_support_collapse.mp4

Outputs:
  - results.csv: one row per generated/GT pair.
  - results.json: all pair summaries.
  - alignments/*_dtw_alignment.csv: IoU-based DTW alignment path.
  - alignments/*_tempo_alignment.csv: best global linear alignment path.
  - details/*.json: per-pair metadata and statistics.

Joint-empty frame policy:
  A frame pair whose motion-mask union is empty is not included in any
  framewise ST-IoU average.  During affine/DTW optimization, such a pair has
  IoU 0, so an alignment cannot gain score by matching silent frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


EPS = 1e-8

PAIR_STAT_METRICS = [
    "spatial_iou",
    "weighted_spatial_iou",
    "strict_st_iou",
    "tempo_aligned_st_iou",
    "dtw_aligned_st_iou",
    "strict_mse_mean",
    "strict_masked_mse_mean",
    "strict_psnr_mean",
    "strict_ssim_mean",
    "tempo_mse_mean",
    "tempo_masked_mse_mean",
    "tempo_psnr_mean",
    "tempo_ssim_mean",
    "dtw_mse_mean",
    "dtw_masked_mse_mean",
    "dtw_psnr_mean",
    "dtw_ssim_mean",
]


@dataclass
class VideoData:
    path: Path
    frames: np.ndarray
    fps: float
    original_frame_count: int
    original_width: int
    original_height: int

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration(self) -> float:
        if self.fps <= 0:
            return float(self.frame_count)
        return float(self.frame_count) / float(self.fps)


def parse_size(value: str) -> tuple[int, int]:
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("Use WIDTHxHEIGHT, for example 256x256")
    width_s, height_s = value.lower().split("x", 1)
    width, height = int(width_s), int(height_s)
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive")
    return width, height


def resolve_gt_path(dataset_root: Path, generated_path: Path) -> Path | None:
    stem = generated_path.stem
    if "." not in stem:
        return None
    series, case_name = stem.split(".", 1)
    gt_path = dataset_root / series / case_name / f"{case_name}.mp4"
    return gt_path if gt_path.exists() else None


def load_video(path: Path, target_size: tuple[int, int]) -> VideoData:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames = []
    target_width, target_height = target_size
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        resized = cv2.resize(
            frame_bgr,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb.astype(np.float32) / 255.0)
    capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    if fps <= 0:
        fps = 1.0

    return VideoData(
        path=path,
        frames=np.stack(frames, axis=0),
        fps=fps,
        original_frame_count=frame_count,
        original_width=width,
        original_height=height,
    )


def grayscale(frames: np.ndarray) -> np.ndarray:
    return (
        0.299 * frames[..., 0]
        + 0.587 * frames[..., 1]
        + 0.114 * frames[..., 2]
    ).astype(np.float32)


def make_motion_masks(
    frames: np.ndarray,
    threshold: float,
    bg_alpha: float,
    blur_kernel: int,
    morph_kernel: int,
) -> np.ndarray:
    """Build binary motion masks with an adaptive background model.

    This follows the static-camera spirit of Physics-IQ: frame/background
    difference, thresholding, and temporal background update. The first frame is
    the initial background, so mask[0] is usually empty.
    """
    gray = grayscale(frames)
    background = gray[0].copy()
    masks = []

    blur_kernel = int(blur_kernel)
    if blur_kernel > 1 and blur_kernel % 2 == 0:
        blur_kernel += 1
    morph_kernel = int(morph_kernel)
    morph = None
    if morph_kernel > 1:
        morph = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)

    for idx in range(gray.shape[0]):
        diff = np.abs(gray[idx] - background)
        if blur_kernel > 1:
            diff = cv2.GaussianBlur(diff, (blur_kernel, blur_kernel), 0)
        mask = (diff >= threshold).astype(np.uint8)
        if morph is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph)
        masks.append(mask.astype(bool))
        background = (1.0 - bg_alpha) * background + bg_alpha * gray[idx]

    return np.stack(masks, axis=0)


def sequence_by_reference_time(
    values: np.ndarray,
    src_fps: float,
    reference_duration: float,
    eval_fps: float,
    missing_policy: str,
) -> np.ndarray:
    target_count = max(1, int(round(reference_duration * eval_fps)))
    out = []
    zero = np.zeros_like(values[0])
    for target_idx in range(target_count):
        timestamp = float(target_idx) / float(eval_fps)
        src_idx = int(round(timestamp * src_fps))
        if 0 <= src_idx < values.shape[0]:
            out.append(values[src_idx])
        elif missing_policy == "zero":
            out.append(zero)
        elif missing_policy == "drop":
            continue
        else:
            out.append(values[min(max(src_idx, 0), values.shape[0] - 1)])
    if not out:
        return values[:0]
    return np.stack(out, axis=0)


def iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    if union <= 0:
        return 1.0
    return float(intersection / union)


def active_frame_pairs(
    gt_masks: np.ndarray, gen_masks: np.ndarray
) -> np.ndarray:
    """Return whether each same-index mask pair contains any motion."""
    count = min(gt_masks.shape[0], gen_masks.shape[0])
    if count <= 0:
        return np.zeros(0, dtype=bool)
    union = np.logical_or(gt_masks[:count], gen_masks[:count])
    return np.any(union, axis=(1, 2))


def mean_iou_over_active_pairs(
    iou_values: Iterable[float], active: Iterable[bool]
) -> tuple[float, int, int]:
    """Average IoU after removing joint-empty pairs.

    A pair with no motion in either video is not evidence of a successful
    match.  If a sequence has no active pair at all, return 0 rather than the
    old empty-empty IoU of 1, while exposing the zero valid-pair count.
    """
    iou_array = np.asarray(list(iou_values), dtype=np.float64)
    active_array = np.asarray(list(active), dtype=bool)
    if iou_array.shape != active_array.shape:
        raise ValueError("IoU and active-pair arrays must have the same shape")
    active_count = int(np.count_nonzero(active_array))
    total_count = int(active_array.size)
    if active_count <= 0:
        return 0.0, 0, total_count
    return float(np.mean(iou_array[active_array])), active_count, total_count


def spatial_iou(gt_masks: np.ndarray, gen_masks: np.ndarray) -> float:
    gt_map = np.any(gt_masks, axis=0)
    gen_map = np.any(gen_masks, axis=0)
    return iou_bool(gt_map, gen_map)


def spatiotemporal_iou(gt_masks: np.ndarray, gen_masks: np.ndarray) -> float:
    count = min(gt_masks.shape[0], gen_masks.shape[0])
    if count <= 0:
        return float("nan")
    active = active_frame_pairs(gt_masks[:count], gen_masks[:count])
    values = [
        iou_bool(gt_masks[idx], gen_masks[idx])
        for idx in range(count)
        if active[idx]
    ]
    if not values:
        return 0.0
    return float(np.mean(values))


def weighted_spatial_iou(gt_masks: np.ndarray, gen_masks: np.ndarray) -> float:
    gt_map = gt_masks.astype(np.float32).mean(axis=0)
    gen_map = gen_masks.astype(np.float32).mean(axis=0)
    numerator = np.minimum(gt_map, gen_map).sum(dtype=np.float64)
    denominator = np.maximum(gt_map, gen_map).sum(dtype=np.float64)
    if denominator <= 0:
        return 1.0
    return float(numerator / denominator)


def pairwise_iou_and_activity(
    gt_masks: np.ndarray,
    gen_masks: np.ndarray,
    empty_empty_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise motion IoU and whether each pair has nonempty union.

    ``empty_empty_value`` is deliberately 0 by default.  The mathematical
    helper ``iou_bool`` retains the conventional empty-empty value of 1 for
    spatial edge cases, but alignment objectives must not reward silence.
    """
    gt_flat = gt_masks.reshape(gt_masks.shape[0], -1).astype(np.uint8)
    gen_flat = gen_masks.reshape(gen_masks.shape[0], -1).astype(np.uint8)
    gt_packed = np.packbits(gt_flat, axis=1)
    gen_packed = np.packbits(gen_flat, axis=1)
    popcount = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)

    scores = np.empty((gt_masks.shape[0], gen_masks.shape[0]), dtype=np.float32)
    active = np.zeros_like(scores, dtype=bool)
    for gt_idx in range(gt_packed.shape[0]):
        inter = popcount[np.bitwise_and(gt_packed[gt_idx], gen_packed)].sum(axis=1)
        union = popcount[np.bitwise_or(gt_packed[gt_idx], gen_packed)].sum(axis=1)
        row = np.full(
            gen_masks.shape[0], float(empty_empty_value), dtype=np.float32
        )
        non_empty = union > 0
        row[non_empty] = inter[non_empty] / union[non_empty]
        scores[gt_idx] = row
        active[gt_idx] = union > 0
    return scores, active


def pairwise_iou_matrix(
    gt_masks: np.ndarray,
    gen_masks: np.ndarray,
    empty_empty_value: float = 0.0,
) -> np.ndarray:
    """Return pairwise IoU; joint-empty pairs default to zero."""
    scores, _ = pairwise_iou_and_activity(
        gt_masks, gen_masks, empty_empty_value=empty_empty_value
    )
    return scores


def tempo_aligned_iou(
    pairwise_iou: np.ndarray,
    pairwise_active: np.ndarray | None,
    a_steps: int,
    b_steps: int,
    a_min_scale: float,
    a_max_scale: float,
    max_shift_fraction: float,
    min_generated_span_coverage: float = 0.95,
) -> tuple[float, dict, list[dict]]:
    """Score affine time mappings with global and bidirectional coverage.

    The returned path is GT-indexed for frame-level metrics. The optimization
    score is the mean of full-sequence GT-to-generated and generated-to-GT
    IoU. Joint-empty pairs have already been assigned IoU 0, and therefore
    cannot improve the search objective. The returned score is an
    event-conditioned score: joint-empty pairs are removed from its
    denominator, while out-of-range mappings remain zero penalties.
    """
    gt_count, gen_count = pairwise_iou.shape
    if pairwise_active is None:
        pairwise_active = np.ones_like(pairwise_iou, dtype=bool)
    if pairwise_active.shape != pairwise_iou.shape:
        raise ValueError("pairwise_active must have the same shape as pairwise_iou")
    if gt_count <= 1:
        nominal_a = 1.0
    else:
        nominal_a = max(1.0, gen_count - 1) / max(1.0, gt_count - 1)
    a_min = max(0.01, nominal_a * a_min_scale)
    a_max = max(a_min, nominal_a * a_max_scale)
    a_values = np.linspace(a_min, a_max, max(1, a_steps))
    max_shift = int(round(max_shift_fraction * max(gt_count, gen_count)))
    b_values = np.linspace(-max_shift, max_shift, max(1, b_steps))

    best_score = -1.0
    best_a = float("nan")
    best_b = float("nan")
    best_valid = 0
    best_forward_score = 0.0
    best_reverse_score = 0.0
    best_report_score = 0.0
    best_reverse_valid = 0
    best_forward_active = 0
    best_reverse_active = 0
    best_forward_joint_empty = 0
    best_reverse_joint_empty = 0
    best_path: list[dict] = []
    best_span_coverage = 0.0
    best_unique_coverage = 0.0
    best_collision_rate = 0.0
    fallback = None

    for a in a_values:
        for b in b_values:
            forward_total = 0.0
            valid = 0
            forward_active = 0
            forward_joint_empty = 0
            selected_gen_indices = []
            path = []
            for gt_idx in range(gt_count):
                gen_idx = int(round(float(a) * gt_idx + float(b)))
                if 0 <= gen_idx < gen_count:
                    score = float(pairwise_iou[gt_idx, gen_idx])
                    is_active = bool(pairwise_active[gt_idx, gen_idx])
                    forward_total += score
                    valid += 1
                    if is_active:
                        forward_active += 1
                    else:
                        forward_joint_empty += 1
                    selected_gen_indices.append(gen_idx)
                    path.append(
                        {
                            "gt_index": gt_idx,
                            "gen_index": gen_idx,
                            "iou": score,
                            "is_mapped": True,
                            "is_active_pair": is_active,
                        }
                    )
                else:
                    path.append(
                        {
                            "gt_index": gt_idx,
                            "gen_index": None,
                            "iou": 0.0,
                            "is_mapped": False,
                            "is_active_pair": False,
                        }
                    )

            # Score the inverse map as well. This makes every generated frame
            # participate in the objective, including frames skipped by a>1.
            reverse_total = 0.0
            reverse_valid = 0
            reverse_active = 0
            reverse_joint_empty = 0
            if abs(float(a)) > 1e-12:
                for gen_idx in range(gen_count):
                    gt_idx = int(round((float(gen_idx) - float(b)) / float(a)))
                    if 0 <= gt_idx < gt_count:
                        reverse_total += float(pairwise_iou[gt_idx, gen_idx])
                        reverse_valid += 1
                        if pairwise_active[gt_idx, gen_idx]:
                            reverse_active += 1
                        else:
                            reverse_joint_empty += 1
            forward_score = forward_total / float(max(1, gt_count))
            reverse_score = reverse_total / float(max(1, gen_count))
            optimization_score = 0.5 * (forward_score + reverse_score)

            # Remove only joint-empty mapped pairs from the reported score.
            # Frames mapped outside the other sequence remain zero penalties,
            # preserving the full-coverage requirement.
            forward_report_den = forward_active + (gt_count - valid)
            reverse_report_den = reverse_active + (gen_count - reverse_valid)
            forward_report_score = forward_total / float(max(1, forward_report_den))
            reverse_report_score = reverse_total / float(max(1, reverse_report_den))
            report_score = 0.5 * (forward_report_score + reverse_report_score)
            if selected_gen_indices:
                span = max(selected_gen_indices) - min(selected_gen_indices) + 1
                span_coverage = min(1.0, span / float(max(1, gen_count)))
                unique_coverage = len(set(selected_gen_indices)) / float(max(1, gen_count))
                collision_rate = (valid - len(set(selected_gen_indices))) / float(valid)
            else:
                span_coverage = 0.0
                unique_coverage = 0.0
                collision_rate = 0.0

            candidate = (
                optimization_score,
                span_coverage,
                valid,
                float(a),
                float(b),
                path,
                unique_coverage,
                collision_rate,
                forward_score,
                reverse_score,
                reverse_valid,
                report_score,
                forward_active,
                reverse_active,
                forward_joint_empty,
                reverse_joint_empty,
            )
            if fallback is None or candidate[:3] > fallback[:3]:
                fallback = candidate

            # Require the affine line to span almost the whole generated
            # sequence. This prevents selecting only a short, high-IoU clip.
            if span_coverage < min_generated_span_coverage:
                continue
            if (
                optimization_score > best_score
                or (
                    optimization_score == best_score
                    and span_coverage > best_span_coverage
                )
            ):
                best_score = optimization_score
                best_a = float(a)
                best_b = float(b)
                best_valid = valid
                best_path = path
                best_span_coverage = span_coverage
                best_unique_coverage = unique_coverage
                best_collision_rate = collision_rate
                best_forward_score = forward_score
                best_reverse_score = reverse_score
                best_reverse_valid = reverse_valid
                best_report_score = report_score
                best_forward_active = forward_active
                best_reverse_active = reverse_active
                best_forward_joint_empty = forward_joint_empty
                best_reverse_joint_empty = reverse_joint_empty

    coverage_constraint_satisfied = True
    if not best_path:
        if fallback is None:
            raise RuntimeError("No affine alignment candidate was evaluated")
        (
            best_score,
            best_span_coverage,
            best_valid,
            best_a,
            best_b,
            best_path,
            best_unique_coverage,
            best_collision_rate,
            best_forward_score,
            best_reverse_score,
            best_reverse_valid,
            best_report_score,
            best_forward_active,
            best_reverse_active,
            best_forward_joint_empty,
            best_reverse_joint_empty,
        ) = fallback
        coverage_constraint_satisfied = False

    return best_report_score, {
        "a": best_a,
        "b": best_b,
        "optimization_score": best_score,
        "reported_score": best_report_score,
        "valid_pairs": best_valid,
        "reverse_valid_pairs": best_reverse_valid,
        "active_pairs": best_forward_active,
        "reverse_active_pairs": best_reverse_active,
        "joint_empty_pairs": best_forward_joint_empty,
        "reverse_joint_empty_pairs": best_reverse_joint_empty,
        "gt_to_gen_score": best_forward_score,
        "gen_to_gt_score": best_reverse_score,
        "generated_span_coverage": best_span_coverage,
        "generated_unique_coverage": best_unique_coverage,
        "generated_collision_rate": best_collision_rate,
        "min_generated_span_coverage": min_generated_span_coverage,
        "coverage_constraint_satisfied": coverage_constraint_satisfied,
    }, best_path


def dtw_alignment(
    pairwise_iou: np.ndarray,
    window_fraction: float = 0.15,
    max_non_diagonal_steps: int = 8,
) -> list[tuple[int, int, float]]:
    """Find a constrained monotonic DTW path.

    The normalized Sakoe-Chiba window prevents the path from spending most of
    its length on one row or column. Limiting consecutive horizontal/vertical
    steps prevents a long L-shaped path from reusing a static frame.
    """
    gt_count, gen_count = pairwise_iou.shape
    if gt_count <= 0 or gen_count <= 0:
        return []

    window_fraction = max(0.0, float(window_fraction))
    max_non_diagonal_steps = max(1, int(max_non_diagonal_steps))
    cost = 1.0 - pairwise_iou.astype(np.float64)

    # State k records k consecutive horizontal/vertical steps ending here.
    state_count = max_non_diagonal_steps + 1
    dp = np.full((gt_count, gen_count, state_count), np.inf, dtype=np.float64)
    prev = np.full(
        (gt_count, gen_count, state_count, 3), -1, dtype=np.int32
    )

    gt_axis = np.arange(gt_count, dtype=np.float64) / max(1, gt_count - 1)
    gen_axis = np.arange(gen_count, dtype=np.float64) / max(1, gen_count - 1)
    allowed = np.abs(gt_axis[:, None] - gen_axis[None, :]) <= (
        window_fraction + 1e-12
    )

    if not allowed[0, 0] or not allowed[gt_count - 1, gen_count - 1]:
        raise RuntimeError("DTW window excludes a required endpoint")
    dp[0, 0, 0] = cost[0, 0]

    for i in range(gt_count):
        for j in range(gen_count):
            if not allowed[i, j] or (i == 0 and j == 0):
                continue

            # Diagonal moves reset the non-diagonal run length.
            if i > 0 and j > 0:
                predecessor_state = int(np.argmin(dp[i - 1, j - 1]))
                predecessor_cost = dp[i - 1, j - 1, predecessor_state]
                if math.isfinite(predecessor_cost):
                    dp[i, j, 0] = predecessor_cost + cost[i, j]
                    prev[i, j, 0] = (i - 1, j - 1, predecessor_state)

            # Vectorized vertical/horizontal transitions increase the run
            # length while keeping the same state recurrence.
            previous_vertical = (
                dp[i - 1, j, :-1] if i > 0 else np.full(state_count - 1, np.inf)
            )
            previous_horizontal = (
                dp[i, j - 1, :-1] if j > 0 else np.full(state_count - 1, np.inf)
            )
            choose_horizontal = previous_horizontal < previous_vertical
            best_previous = np.minimum(previous_vertical, previous_horizontal)
            finite = np.isfinite(best_previous)
            dp[i, j, 1:] = best_previous + cost[i, j]
            run_states = np.arange(state_count - 1, dtype=np.int32)
            prev[i, j, 1:, 0] = np.where(choose_horizontal, i, i - 1)
            prev[i, j, 1:, 1] = np.where(choose_horizontal, j - 1, j)
            prev[i, j, 1:, 2] = run_states
            invalid_states = np.logical_not(finite)
            prev[i, j, 1:][invalid_states] = -1

    path = []
    final_state = int(np.argmin(dp[gt_count - 1, gen_count - 1]))
    if not math.isfinite(dp[gt_count - 1, gen_count - 1, final_state]):
        raise RuntimeError(
            "No DTW path satisfies the configured window and step constraints"
        )
    i, j, state = gt_count - 1, gen_count - 1, final_state
    while i >= 0 and j >= 0:
        path.append((i, j, float(pairwise_iou[i, j])))
        pi, pj, previous_state = prev[i, j, state]
        if pi < 0 or pj < 0 or previous_state < 0:
            break
        i, j, state = int(pi), int(pj), int(previous_state)
    path.reverse()
    return path


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def masked_mse(a: np.ndarray, b: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
    union = np.logical_or(mask_a, mask_b)
    if not np.any(union):
        return None
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff[union] ** 2))


def psnr_from_mse(value: float) -> float:
    if value <= 0:
        return 100.0
    return float(10.0 * math.log10(1.0 / value))


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """Windowed SSIM for RGB frames in [0, 1]."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    scores = []
    for channel in range(a.shape[-1]):
        x = a[..., channel].astype(np.float32)
        y = b[..., channel].astype(np.float32)
        mux = cv2.GaussianBlur(x, (11, 11), 1.5)
        muy = cv2.GaussianBlur(y, (11, 11), 1.5)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy
        sigx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mux2
        sigy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - muy2
        sigxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - muxy
        numerator = (2 * muxy + c1) * (2 * sigxy + c2)
        denominator = (mux2 + muy2 + c1) * (sigx + sigy + c2)
        ssim_map = numerator / np.maximum(denominator, EPS)
        scores.append(float(np.mean(ssim_map)))
    return float(np.mean(scores))


def summarize(values: Iterable[float], prefix: str) -> dict[str, float | int | None]:
    arr = np.array([v for v in values if v is not None and not math.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_p25": None,
            f"{prefix}_p75": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_count": int(arr.size),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
        f"{prefix}_max": float(np.max(arr)),
    }


def finite_for_json(value):
    if isinstance(value, dict):
        return {key: finite_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_for_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def safe_stem(path: Path) -> str:
    return path.stem.replace(".", "__")


def write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_pair_statistics_csv(path: Path, rows: list[dict], metrics: list[str]) -> None:
    stat_rows = []
    for metric in metrics:
        values = []
        for row in rows:
            value = row.get(metric)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(value) or math.isinf(value):
                continue
            values.append(value)
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        stat_rows.append(
            {
                "metric": metric,
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
                "sem": float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else None,
            }
        )
    write_dict_csv(path, stat_rows)


def write_alignment_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "path_index",
            "gt_index",
            "gen_index",
            "iou",
            "is_mapped",
            "is_active_pair",
            "mse",
            "masked_mse",
            "psnr",
            "ssim",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow({"path_index": idx, **row})


def evaluate_pair(
    gt_video: VideoData,
    gen_video: VideoData,
    args: argparse.Namespace,
) -> tuple[dict, dict, list[dict], list[dict]]:
    gt_masks = make_motion_masks(
        gt_video.frames,
        threshold=args.motion_threshold,
        bg_alpha=args.background_alpha,
        blur_kernel=args.blur_kernel,
        morph_kernel=args.morph_kernel,
    )
    gen_masks = make_motion_masks(
        gen_video.frames,
        threshold=args.motion_threshold,
        bg_alpha=args.background_alpha,
        blur_kernel=args.blur_kernel,
        morph_kernel=args.morph_kernel,
    )

    eval_fps = args.eval_fps if args.eval_fps else gt_video.fps
    reference_duration = gt_video.duration
    strict_gt_masks = sequence_by_reference_time(
        gt_masks,
        gt_video.fps,
        reference_duration,
        eval_fps,
        args.short_video_policy,
    )
    strict_gen_masks = sequence_by_reference_time(
        gen_masks,
        gen_video.fps,
        reference_duration,
        eval_fps,
        args.short_video_policy,
    )
    strict_gt_frames = sequence_by_reference_time(
        gt_video.frames,
        gt_video.fps,
        reference_duration,
        eval_fps,
        args.short_video_policy,
    )
    strict_gen_frames = sequence_by_reference_time(
        gen_video.frames,
        gen_video.fps,
        reference_duration,
        eval_fps,
        args.short_video_policy,
    )

    strict_spatial = spatial_iou(strict_gt_masks, strict_gen_masks)
    strict_st = spatiotemporal_iou(strict_gt_masks, strict_gen_masks)
    strict_weighted = weighted_spatial_iou(strict_gt_masks, strict_gen_masks)
    strict_active = active_frame_pairs(strict_gt_masks, strict_gen_masks)
    strict_active_count = int(np.count_nonzero(strict_active))
    strict_joint_empty_count = int(strict_active.size - strict_active_count)

    strict_rows = []
    strict_count = min(strict_gt_frames.shape[0], strict_gen_frames.shape[0])
    for idx in range(strict_count):
        pair_mse = mse(strict_gt_frames[idx], strict_gen_frames[idx])
        strict_rows.append(
            {
                "index": idx,
                "mse": pair_mse,
                "masked_mse": masked_mse(
                    strict_gt_frames[idx],
                    strict_gen_frames[idx],
                    strict_gt_masks[idx],
                    strict_gen_masks[idx],
                ),
                "psnr": psnr_from_mse(pair_mse),
                "ssim": ssim_rgb(strict_gt_frames[idx], strict_gen_frames[idx]),
            }
        )

    pair_iou, pair_active = pairwise_iou_and_activity(gt_masks, gen_masks)
    tempo_score, tempo_params, tempo_path = tempo_aligned_iou(
        pair_iou,
        pair_active,
        a_steps=args.tempo_a_steps,
        b_steps=args.tempo_b_steps,
        a_min_scale=args.tempo_a_min_scale,
        a_max_scale=args.tempo_a_max_scale,
        max_shift_fraction=args.tempo_max_shift_fraction,
        min_generated_span_coverage=args.tempo_min_generated_span_coverage,
    )

    tempo_rows = []
    for row in tempo_path:
        gt_idx = row["gt_index"]
        gen_idx = row["gen_index"]
        out_row = {
            "gt_index": gt_idx,
            "gen_index": gen_idx,
            "iou": row["iou"],
            "is_mapped": row["is_mapped"],
            "is_active_pair": row["is_active_pair"],
            "mse": None,
            "masked_mse": None,
            "psnr": None,
            "ssim": None,
        }
        if gen_idx is not None:
            pair_mse = mse(gt_video.frames[gt_idx], gen_video.frames[gen_idx])
            out_row.update(
                {
                    "mse": pair_mse,
                    "masked_mse": masked_mse(
                        gt_video.frames[gt_idx],
                        gen_video.frames[gen_idx],
                        gt_masks[gt_idx],
                        gen_masks[gen_idx],
                    ),
                    "psnr": psnr_from_mse(pair_mse),
                    "ssim": ssim_rgb(gt_video.frames[gt_idx], gen_video.frames[gen_idx]),
                }
            )
        tempo_rows.append(out_row)

    dtw_path_raw = dtw_alignment(
        pair_iou,
        window_fraction=args.dtw_window_fraction,
        max_non_diagonal_steps=args.dtw_max_non_diagonal_steps,
    )
    dtw_rows = []
    for gt_idx, gen_idx, iou_value in dtw_path_raw:
        pair_mse = mse(gt_video.frames[gt_idx], gen_video.frames[gen_idx])
        dtw_rows.append(
            {
                "gt_index": int(gt_idx),
                "gen_index": int(gen_idx),
                "iou": float(iou_value),
                "is_mapped": True,
                "is_active_pair": bool(pair_active[gt_idx, gen_idx]),
                "mse": pair_mse,
                "masked_mse": masked_mse(
                    gt_video.frames[gt_idx],
                    gen_video.frames[gen_idx],
                    gt_masks[gt_idx],
                    gen_masks[gen_idx],
                ),
                "psnr": psnr_from_mse(pair_mse),
                "ssim": ssim_rgb(gt_video.frames[gt_idx], gen_video.frames[gen_idx]),
            }
        )

    dtw_active_iou_values = [
        row["iou"] for row in dtw_rows if row["is_active_pair"]
    ]
    dtw_iou_stats = summarize(dtw_active_iou_values, "dtw_iou")
    dtw_mse_stats = summarize((row["mse"] for row in dtw_rows), "dtw_mse")
    dtw_masked_mse_stats = summarize((row["masked_mse"] for row in dtw_rows), "dtw_masked_mse")
    dtw_psnr_stats = summarize((row["psnr"] for row in dtw_rows), "dtw_psnr")
    dtw_ssim_stats = summarize((row["ssim"] for row in dtw_rows), "dtw_ssim")
    tempo_active_iou_values = [
        row["iou"] for row in tempo_rows if row["is_active_pair"]
    ]
    tempo_iou_stats = summarize(tempo_active_iou_values, "tempo_iou")
    tempo_mse_stats = summarize((row["mse"] for row in tempo_rows), "tempo_mse")
    tempo_masked_mse_stats = summarize((row["masked_mse"] for row in tempo_rows), "tempo_masked_mse")
    tempo_psnr_stats = summarize((row["psnr"] for row in tempo_rows), "tempo_psnr")
    tempo_ssim_stats = summarize((row["ssim"] for row in tempo_rows), "tempo_ssim")
    strict_mse_stats = summarize((row["mse"] for row in strict_rows), "strict_mse")
    strict_masked_mse_stats = summarize((row["masked_mse"] for row in strict_rows), "strict_masked_mse")
    strict_psnr_stats = summarize((row["psnr"] for row in strict_rows), "strict_psnr")
    strict_ssim_stats = summarize((row["ssim"] for row in strict_rows), "strict_ssim")

    dtw_st_iou, dtw_active_count, dtw_total_count = mean_iou_over_active_pairs(
        (row["iou"] for row in dtw_rows),
        (row["is_active_pair"] for row in dtw_rows),
    )

    summary = {
        "generated_video": str(gen_video.path),
        "ground_truth_video": str(gt_video.path),
        "gt_frames": gt_video.frame_count,
        "gen_frames": gen_video.frame_count,
        "gt_fps": gt_video.fps,
        "gen_fps": gen_video.fps,
        "gt_duration": gt_video.duration,
        "gen_duration": gen_video.duration,
        "eval_fps": eval_fps,
        "strict_eval_frames": int(strict_gt_masks.shape[0]),
        "strict_active_frames": strict_active_count,
        "strict_joint_empty_frames": strict_joint_empty_count,
        "strict_joint_empty_fraction": (
            float(strict_joint_empty_count / max(1, strict_active.size))
        ),
        "spatial_iou": strict_spatial,
        "strict_st_iou": strict_st,
        "weighted_spatial_iou": strict_weighted,
        "tempo_aligned_st_iou": tempo_score,
        "tempo_a": tempo_params["a"],
        "tempo_b": tempo_params["b"],
        "tempo_valid_pairs": tempo_params["valid_pairs"],
        "tempo_reverse_valid_pairs": tempo_params["reverse_valid_pairs"],
        "tempo_active_pairs": tempo_params["active_pairs"],
        "tempo_reverse_active_pairs": tempo_params["reverse_active_pairs"],
        "tempo_joint_empty_pairs": tempo_params["joint_empty_pairs"],
        "tempo_reverse_joint_empty_pairs": tempo_params["reverse_joint_empty_pairs"],
        "tempo_gt_to_gen_score": tempo_params["gt_to_gen_score"],
        "tempo_gen_to_gt_score": tempo_params["gen_to_gt_score"],
        "tempo_generated_span_coverage": tempo_params["generated_span_coverage"],
        "tempo_generated_unique_coverage": tempo_params["generated_unique_coverage"],
        "tempo_generated_collision_rate": tempo_params["generated_collision_rate"],
        "tempo_coverage_constraint_satisfied": tempo_params["coverage_constraint_satisfied"],
        "tempo_path_length": len(tempo_rows),
        "dtw_aligned_st_iou": dtw_st_iou,
        "dtw_active_pairs": dtw_active_count,
        "dtw_joint_empty_pairs": int(dtw_total_count - dtw_active_count),
        "dtw_path_length": len(dtw_rows),
        "dtw_unique_gt_frames": len({row["gt_index"] for row in dtw_rows}),
        "dtw_unique_gen_frames": len({row["gen_index"] for row in dtw_rows}),
        "dtw_window_fraction": args.dtw_window_fraction,
        "dtw_max_non_diagonal_steps": args.dtw_max_non_diagonal_steps,
        **strict_mse_stats,
        **strict_masked_mse_stats,
        **strict_psnr_stats,
        **strict_ssim_stats,
        **tempo_iou_stats,
        **tempo_mse_stats,
        **tempo_masked_mse_stats,
        **tempo_psnr_stats,
        **tempo_ssim_stats,
        **dtw_iou_stats,
        **dtw_mse_stats,
        **dtw_masked_mse_stats,
        **dtw_psnr_stats,
        **dtw_ssim_stats,
    }
    details = {
        "summary": summary,
        "motion_mask": {
            "threshold": args.motion_threshold,
            "background_alpha": args.background_alpha,
            "blur_kernel": args.blur_kernel,
            "morph_kernel": args.morph_kernel,
        },
        "strict_policy": {
            "reference": "ground_truth_duration",
            "short_video_policy": args.short_video_policy,
            "joint_empty_frame_policy": "exclude_from_st_iou_average",
        },
        "st_iou_policy": {
            "joint_empty_pair_iou_for_alignment": 0.0,
            "joint_empty_pairs_in_reported_st_iou": "excluded",
            "out_of_range_affine_mappings": "zero_penalty",
        },
        "strict_statistics": {
            **strict_mse_stats,
            **strict_masked_mse_stats,
            **strict_psnr_stats,
            **strict_ssim_stats,
        },
        "tempo_alignment": {
            **tempo_params,
            "score": tempo_score,
            "a_steps": args.tempo_a_steps,
            "b_steps": args.tempo_b_steps,
            "a_min_scale": args.tempo_a_min_scale,
            "a_max_scale": args.tempo_a_max_scale,
            "max_shift_fraction": args.tempo_max_shift_fraction,
            "min_generated_span_coverage": args.tempo_min_generated_span_coverage,
        },
        "dtw_alignment": {
            "window_fraction": args.dtw_window_fraction,
            "max_non_diagonal_steps": args.dtw_max_non_diagonal_steps,
            "path_length": len(dtw_rows),
        },
        "dtw_statistics": {
            **dtw_iou_stats,
            **dtw_mse_stats,
            **dtw_masked_mse_stats,
            **dtw_psnr_stats,
            **dtw_ssim_stats,
        },
        "tempo_statistics": {
            **tempo_iou_stats,
            **tempo_mse_stats,
            **tempo_masked_mse_stats,
            **tempo_psnr_stats,
            **tempo_ssim_stats,
        },
    }
    return summary, details, dtw_rows, tempo_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Physics-IQ style and IoU-aligned video metrics."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=parse_size, default=(256, 256))
    parser.add_argument("--eval-fps", type=float, default=None)
    parser.add_argument(
        "--short-video-policy",
        choices=("hold", "zero", "drop"),
        default="hold",
        help="How strict physical-time resampling handles generated videos shorter than GT.",
    )
    parser.add_argument("--motion-threshold", type=float, default=0.06)
    parser.add_argument("--background-alpha", type=float, default=0.03)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--morph-kernel", type=int, default=3)
    parser.add_argument("--tempo-a-steps", type=int, default=41)
    parser.add_argument("--tempo-b-steps", type=int, default=61)
    parser.add_argument("--tempo-a-min-scale", type=float, default=0.5)
    parser.add_argument("--tempo-a-max-scale", type=float, default=1.5)
    parser.add_argument("--tempo-max-shift-fraction", type=float, default=0.25)
    parser.add_argument(
        "--tempo-min-generated-span-coverage",
        type=float,
        default=0.95,
        help="Minimum fraction of the generated index span covered by affine alignment.",
    )
    parser.add_argument(
        "--dtw-window-fraction",
        type=float,
        default=0.15,
        help="Normalized Sakoe-Chiba window for constrained DTW.",
    )
    parser.add_argument(
        "--dtw-max-non-diagonal-steps",
        type=int,
        default=8,
        help="Maximum consecutive horizontal/vertical DTW steps.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    generated_dir = args.generated_dir
    dataset_root = args.dataset_root
    output_dir = args.output_dir
    alignment_dir = output_dir / "alignments"
    details_dir = output_dir / "details"
    alignment_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    generated_videos = sorted(generated_dir.glob("*.mp4"))
    if args.limit is not None:
        generated_videos = generated_videos[: args.limit]
    if not generated_videos:
        raise SystemExit(f"No .mp4 files found in {generated_dir}")

    summaries = []
    skipped = []
    for generated_path in generated_videos:
        gt_path = resolve_gt_path(dataset_root, generated_path)
        if gt_path is None:
            skipped.append({"generated_video": str(generated_path), "reason": "GT not found"})
            print(f"[skip] GT not found for {generated_path}")
            continue

        print(f"[eval] {generated_path.name} -> {gt_path}")
        gt_video = load_video(gt_path, args.size)
        gen_video = load_video(generated_path, args.size)
        summary, details, dtw_rows, tempo_rows = evaluate_pair(gt_video, gen_video, args)
        summaries.append(summary)

        stem = safe_stem(generated_path)
        write_alignment_csv(alignment_dir / f"{stem}_dtw_alignment.csv", dtw_rows)
        write_alignment_csv(alignment_dir / f"{stem}_tempo_alignment.csv", tempo_rows)
        with (details_dir / f"{stem}.json").open("w", encoding="utf-8") as file:
            json.dump(finite_for_json(details), file, ensure_ascii=False, indent=2)

    write_dict_csv(output_dir / "results.csv", summaries)
    write_pair_statistics_csv(output_dir / "metric_pair_statistics.csv", summaries, PAIR_STAT_METRICS)
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(
            finite_for_json({"results": summaries, "skipped": skipped}),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[done] evaluated={len(summaries)} skipped={len(skipped)} output={output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
