from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    width: int = 256
    height: int = 256
    threshold: float = 0.06
    background_alpha: float = 0.03
    blur_kernel: int = 5
    morph_kernel: int = 3


def cache_key(video: Path, config: MotionConfig) -> str:
    stat = video.stat()
    identity = {
        "path": str(video.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "config": asdict(config),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def extract_motion_maps(video: Path, config: MotionConfig) -> dict:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    background = None
    spatial = np.zeros((config.height, config.width), dtype=bool)
    weighted_sum = np.zeros((config.height, config.width), dtype=np.uint32)
    frame_count = 0
    blur_kernel = config.blur_kernel + (config.blur_kernel % 2 == 0)
    morph = None
    if config.morph_kernel > 1:
        morph = np.ones((config.morph_kernel, config.morph_kernel), dtype=np.uint8)

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        resized = cv2.resize(
            frame_bgr, (config.width, config.height), interpolation=cv2.INTER_AREA
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gray = (
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        ).astype(np.float32)
        if background is None:
            background = gray.copy()
        difference = np.abs(gray - background)
        if blur_kernel > 1:
            difference = cv2.GaussianBlur(difference, (blur_kernel, blur_kernel), 0)
        mask = (difference >= config.threshold).astype(np.uint8)
        if morph is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph)
        active = mask.astype(bool)
        spatial |= active
        weighted_sum += mask
        background = (
            (1.0 - config.background_alpha) * background
            + config.background_alpha * gray
        )
        frame_count += 1
    capture.release()
    if frame_count == 0:
        raise RuntimeError(f"No frames decoded from video: {video}")
    return {
        "spatial": spatial,
        "weighted": weighted_sum.astype(np.float32) / float(frame_count),
        "frame_count": frame_count,
        "fps": source_fps,
        "source_width": source_width,
        "source_height": source_height,
    }


class MotionMapCache:
    def __init__(self, root: Path, config: MotionConfig):
        self.root = root
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, video: Path) -> Path:
        return self.root / f"{cache_key(video, self.config)}.npz"

    def get(self, video: Path) -> dict:
        cache_path = self.path_for(video)
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as data:
                return {
                    "spatial": data["spatial"].astype(bool),
                    "weighted": data["weighted"].astype(np.float32),
                    "frame_count": int(data["frame_count"]),
                    "fps": float(data["fps"]),
                    "source_width": int(data["source_width"]),
                    "source_height": int(data["source_height"]),
                }
        maps = extract_motion_maps(video, self.config)
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            spatial=maps["spatial"].astype(np.uint8),
            weighted=maps["weighted"].astype(np.float32),
            frame_count=np.int32(maps["frame_count"]),
            fps=np.float32(maps["fps"]),
            source_width=np.int32(maps["source_width"]),
            source_height=np.int32(maps["source_height"]),
        )
        os.replace(temporary, cache_path)
        return maps


def binary_iou(reference: np.ndarray, prediction: np.ndarray) -> float:
    intersection = np.logical_and(reference, prediction).sum(dtype=np.float64)
    union = np.logical_or(reference, prediction).sum(dtype=np.float64)
    return 1.0 if union == 0 else float(intersection / union)


def fuzzy_iou(reference: np.ndarray, prediction: np.ndarray) -> float:
    intersection = np.minimum(reference, prediction).sum(dtype=np.float64)
    union = np.maximum(reference, prediction).sum(dtype=np.float64)
    return 1.0 if union == 0 else float(intersection / union)


def binary_decomposition(factual: np.ndarray, counterfactual: np.ndarray) -> dict:
    return {
        "preserved": np.logical_and(factual, counterfactual),
        "suppressed": np.logical_and(factual, np.logical_not(counterfactual)),
        "induced": np.logical_and(counterfactual, np.logical_not(factual)),
    }


def weighted_decomposition(factual: np.ndarray, counterfactual: np.ndarray) -> dict:
    return {
        "preserved": np.minimum(factual, counterfactual),
        "suppressed": np.maximum(factual - counterfactual, 0.0),
        "induced": np.maximum(counterfactual - factual, 0.0),
    }


def two_channel_binary_iou(reference: dict, prediction: dict) -> float:
    intersection = sum(
        np.logical_and(reference[name], prediction[name]).sum(dtype=np.float64)
        for name in ("suppressed", "induced")
    )
    union = sum(
        np.logical_or(reference[name], prediction[name]).sum(dtype=np.float64)
        for name in ("suppressed", "induced")
    )
    return 1.0 if union == 0 else float(intersection / union)


def two_channel_fuzzy_iou(reference: dict, prediction: dict) -> float:
    intersection = sum(
        np.minimum(reference[name], prediction[name]).sum(dtype=np.float64)
        for name in ("suppressed", "induced")
    )
    union = sum(
        np.maximum(reference[name], prediction[name]).sum(dtype=np.float64)
        for name in ("suppressed", "induced")
    )
    return 1.0 if union == 0 else float(intersection / union)


def score_pair(gt_factual: dict, gt_cf: dict, gen_factual: dict, gen_cf: dict) -> dict:
    gt_binary = binary_decomposition(gt_factual["spatial"], gt_cf["spatial"])
    gen_binary = binary_decomposition(gen_factual["spatial"], gen_cf["spatial"])
    gt_weighted = weighted_decomposition(gt_factual["weighted"], gt_cf["weighted"])
    gen_weighted = weighted_decomposition(gen_factual["weighted"], gen_cf["weighted"])

    return {
        "cf_spatial_effect_iou": two_channel_binary_iou(gt_binary, gen_binary),
        "cf_weighted_effect_iou": two_channel_fuzzy_iou(gt_weighted, gen_weighted),
        "cf_preserve_spatial_iou": binary_iou(gt_binary["preserved"], gen_binary["preserved"]),
        "cf_preserve_weighted_iou": fuzzy_iou(gt_weighted["preserved"], gen_weighted["preserved"]),
        "suppressed_spatial_iou_diagnostic": binary_iou(gt_binary["suppressed"], gen_binary["suppressed"]),
        "induced_spatial_iou_diagnostic": binary_iou(gt_binary["induced"], gen_binary["induced"]),
        "suppressed_weighted_iou_diagnostic": fuzzy_iou(gt_weighted["suppressed"], gen_weighted["suppressed"]),
        "induced_weighted_iou_diagnostic": fuzzy_iou(gt_weighted["induced"], gen_weighted["induced"]),
        "gt_preserved_spatial_pixels": int(gt_binary["preserved"].sum()),
        "gt_suppressed_spatial_pixels": int(gt_binary["suppressed"].sum()),
        "gt_induced_spatial_pixels": int(gt_binary["induced"].sum()),
        "gen_preserved_spatial_pixels": int(gen_binary["preserved"].sum()),
        "gen_suppressed_spatial_pixels": int(gen_binary["suppressed"].sum()),
        "gen_induced_spatial_pixels": int(gen_binary["induced"].sum()),
        "gt_preserved_weighted_mass": float(gt_weighted["preserved"].sum()),
        "gt_suppressed_weighted_mass": float(gt_weighted["suppressed"].sum()),
        "gt_induced_weighted_mass": float(gt_weighted["induced"].sum()),
        "gen_preserved_weighted_mass": float(gen_weighted["preserved"].sum()),
        "gen_suppressed_weighted_mass": float(gen_weighted["suppressed"].sum()),
        "gen_induced_weighted_mass": float(gen_weighted["induced"].sum()),
    }
