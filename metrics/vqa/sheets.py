#!/usr/bin/env python3
"""Sample a video over its full duration and compose a labeled contact sheet.

The current policy uses 20 uniformly spaced frames, including the first and
last frame.  ``legacy_stride`` remains available only for reproducing the old
v1 evaluation, which sampled the beginning densely and then jumped to the end.
"""
import json, os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames_cache")
SHEET_COLS = 5
SHEET_ROWS = 4
FRAMES_PER_SHEET = SHEET_COLS * SHEET_ROWS  # 20
CELL_W = 336          # display width per cell
LABEL_H = 22          # timestamp label height
BG = (24, 24, 24)


def pick_indices(n_frames, stride=3, limit=FRAMES_PER_SHEET, include_final=True,
                 policy="uniform"):
    """Return deterministic frame indices for a contact sheet.

    ``uniform`` covers the complete rollout. ``legacy_stride`` exactly retains
    the old v1 behavior for audit-only reproduction.
    """
    if n_frames <= 0 or limit <= 0:
        return []
    if policy == "legacy_stride":
        idx = list(range(0, n_frames, stride))[:limit]
        if include_final and idx and idx[-1] != n_frames - 1:
            idx = idx[:-1] + [n_frames - 1]
        return sorted(set(idx))
    if policy != "uniform":
        raise ValueError(f"unknown sampling policy: {policy}")
    if n_frames <= limit:
        return list(range(n_frames))
    stop = n_frames - 1 if include_final else n_frames
    idx = np.linspace(0, stop, num=limit, endpoint=include_final)
    return sorted(set(int(round(i)) for i in idx))


def decode_frames(video_path):
    """Return RGB uint8 frames and FPS using OpenCV."""
    reader = cv2.VideoCapture(video_path)
    if not reader.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = reader.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    reader.release()
    if not frames:
        raise ValueError(f"video contains no decodable frames: {video_path}")
    return frames, fps


def make_sheet(frames, idx_list, fps, cell_w=CELL_W, out_png=None):
    """Compose frames (H,W,3) at idx_list into a labeled grid sheet."""
    cells = []
    for i in idx_list:
        f = frames[i]
        h, w = f.shape[:2]
        th = int(round(cell_w * h / w))
        im = Image.fromarray(f).resize((cell_w, th), Image.BILINEAR)
        cells.append((im, i))
    if not cells:
        raise ValueError("no frames")
    cell_h = max(im.height for im, _ in cells)
    sheet = Image.new("RGB", (SHEET_COLS * cell_w, SHEET_ROWS * (cell_h + LABEL_H)), BG)
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    for k, (im, i) in enumerate(cells):
        r, c = k // SHEET_COLS, k % SHEET_COLS
        x, y = c * cell_w, r * (cell_h + LABEL_H)
        sheet.paste(im, (x, y + LABEL_H))
        t = i / fps
        draw.text((x + 4, y + 4), f"f{i:04d}@{t:.2f}s", fill=(255, 255, 0), font=font)
    if out_png:
        sheet.save(out_png, quality=95)
    return sheet


def _source_signature(video_path):
    st = os.stat(video_path)
    return {
        "path": os.path.realpath(video_path),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def sheet_for_video(video_path, key, force=False, policy="uniform"):
    """Cache-aware entry: returns (png_path, meta_json_path)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    png_path = os.path.join(CACHE_DIR, key + ".png")
    meta_path = os.path.join(CACHE_DIR, key + ".json")
    sig = _source_signature(video_path)
    if os.path.exists(png_path) and os.path.exists(meta_path) and not force:
        try:
            old = json.load(open(meta_path))
            if old.get("source_signature") == sig and old.get("sampling_policy") == policy:
                return png_path, meta_path
        except Exception:
            pass
    frames, fps = decode_frames(video_path)
    idx = pick_indices(len(frames), stride=3, limit=FRAMES_PER_SHEET,
                       include_final=True, policy=policy)
    make_sheet(frames, idx, fps, out_png=png_path)
    meta = {
        "video": video_path, "fps": fps, "n_frames": len(frames),
        "stride": 3, "limit": FRAMES_PER_SHEET, "include_final": True,
        "sampling_policy": policy, "indices": idx,
        "source_signature": sig,
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return png_path, meta_path


if __name__ == "__main__":
    vp = sys.argv[1]
    key = os.path.splitext(os.path.basename(vp))[0]
    png, meta = sheet_for_video(vp, key, force=True)
    print("sheet:", png, "| meta:", meta)
    print(json.dumps(json.load(open(meta)), ensure_ascii=False))
