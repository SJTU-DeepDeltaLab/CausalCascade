#!/usr/bin/env python3
"""Pure-CV core for the agentic judge: flow, tracks, segments, alignment, renders.

Only cv2/numpy/PIL. Fixed-camera scenes. Tracker = bg-diff blobs + greedy IoU
association + PyrLK refinement; teleport = death + near rebirth marked as jump.
"""
import json, os, math
import numpy as np
import cv2
from PIL import Image, ImageDraw

BG_ALPHA = 0.03
TAU = 6


# ---------------------------------------------------------------- video io
def read_video(video_path, scale=1.0, gray=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if scale != 1.0:
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if gray:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(f)
    cap.release()
    return frames, fps


def bgdiff_mask(gray, bg=None):
    """frame differencing against adaptive background; returns (mask, new_bg)."""
    if isinstance(gray, np.ndarray) and gray.ndim == 2:
        pass
    if bg is None:
        bg = gray.astype(np.float32)
    d = cv2.GaussianBlur(np.abs(gray.astype(np.float32) - bg), (5, 5), 0)
    mask = (d >= TAU).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask, (1 - BG_ALPHA) * bg + BG_ALPHA * gray.astype(np.float32)


# ---------------------------------------------------------------- flow
def flow_summary(video_path, scale=0.5, grid=16):
    """DIS flow per pair -> (per-frame mag mean/p95, grid mags, fps, corrupt flags)."""
    frames, fps = read_video(video_path, scale=scale, gray=True)
    n = len(frames)
    H, W = frames[0].shape
    eng = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    mags = []
    grid_mags = np.zeros((n, grid, grid), np.float32)
    for i in range(1, n):
        f = eng.calc(frames[i - 1], frames[i], None)
        mag = np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)
        mags.append(float(mag.mean()))
        # grid pooling
        gh, gw = H // grid, W // grid
        for gy in range(grid):
            for gx in range(grid):
                grid_mags[i, gy, gx] = float(mag[gy * gh:(gy + 1) * gh, gx * gw:(gx + 1) * gw].mean())
    mags = np.array(mags, np.float32)
    th = float(np.quantile(mags, 0.9) * 0.5) if len(mags) else 1.0
    active = np.zeros(n, bool)
    active[1:] = (mags > th)[: n - 1]
    return {"fps": fps, "n_frames": n, "W": W, "H": H, "mags": mags.tolist(),
            "p95": float(np.quantile(mags, 0.95)) if len(mags) else 0.0,
            "grid": grid_mags.tolist()}


def segments_from_flow(fs, quantile=0.3, min_dur=0.25):
    """Active windows from flow magnitude > quantile, min duration."""
    mags = np.array(fs["mags"])
    fps = fs["fps"]
    th = float(np.quantile(mags, quantile)) if len(mags) else 0.0
    n = len(mags)
    # ``mags[i]`` describes the transition from frame i to frame i+1, so the
    # activity mask has exactly the same length.  The former ``on[1:] = mags``
    # assignment was off by one and failed for every non-empty video.
    on = mags > th
    walls = []
    i = 0
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            if (j - i + 1) >= max(2, int(min_dur * fps)):
                walls.append({"t0": round(i / fps, 2), "t1": round((j + 1) / fps, 2),
                              "peak": round(float(mags[i:j + 1].max()), 3)})
            i = j + 1
        else:
            i += 1
    return {"threshold": round(th, 4), "active_windows": walls}


# ---------------------------------------------------------------- tracker
class Track:
    def __init__(self, tid, t, x, y, area):
        self.id = tid
        self.pts = [(t, x, y, area)]
        self.dead_frames = 0
        self.overlaps = 0
        self.jumps = 0          # hard discontinuities (teleport/motion ghosting)
        self.max_jump_px = 0.0
        self.died = False
        self.rebirth_tracked = False

    @property
    def t_start(self):
        return self.pts[0][0]

    @property
    def t_end(self):
        return self.pts[-1][0]

    def alive_len(self):
        return len(self.pts)

    def step_stats(self, frame_w):
        xs = [p[1] for p in self.pts]
        ys = [p[2] for p in self.pts]
        ts = [p[0] for p in self.pts]
        if len(self.pts) < 2:
            return dict(dist=0.0, med_v=0.0, max_v=0.0)
        d = [math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(len(xs) - 1)]
        dt = [max(ts[i + 1] - ts[i], 1e-3) for i in range(len(ts) - 1)]
        v = [d[i] / dt[i] for i in range(len(d))]
        st = [d[i] / frame_w for i in range(len(d))]  # widths
        # jump detection: any step > max(4*p95 of local speed, abs self-scale)
        p95v = float(np.quantile(v, 0.95)) if len(v) > 3 else (max(v) if v else 0)
        # jump = instantaneous speed > 4x P95 speed (cap interpolation artifacts)
        jmax = 0.0
        jcnt = 0
        for i in range(len(d)):
            if v[i] > 4 * p95v:
                jcnt += 1
            jmax = max(jmax, d[i])
        self.jumps += jcnt                     # aggregate, keep teleport counts
        self.max_jump_px = max(self.max_jump_px, jmax)
        return dict(dist=round(float(sum(d)), 1), med_v=round(float(np.median(v)), 1),
                    max_v=round(float(max(v)), 1), step_p95=round(float(np.quantile(v, 0.95)), 1))


def track_video(video_path, scale=0.5, min_area_frac=0.002, max_area_frac=0.25,
                iou_thr=0.10, cent_thr=0.08):
    """Blob-IoU tracker with PyrLK refinement.  Returns dict with 'tracks'."""
    frames, fps = read_video(video_path, scale=scale, gray=True)
    n = len(frames)
    H, W = frames[0].shape
    bg = None
    tracks = []
    tcur = {}
    min_area = max(30, int(min_area_frac * H * W))
    max_area = int(max_area_frac * H * W)
    corrupt_frames = []
    # first pass: per-frame blobs
    layer = []  # list of dict(frame, blob fields)
    prev = None
    FR = 1.0 / fps
    for i, g in enumerate(frames):
        mask, bg = bgdiff_mask(g, bg)
        area_all = int(mask.sum())
        if area_all > 0.35 * H * W:
            corrupt_frames.append(i)
            layer.append([])          # corrupt/white-flash frames produce no blobs
            continue
        num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        blobs = []
        for k in range(1, num):
            x, y, w, h, a = stats[k]
            if not (min_area <= a <= max_area):
                continue
            blobs.append((float(cents[k][0]), float(cents[k][1]), a, x, y, w, h))
        layer.append(blobs)
        prev = g

    # associate (greedy IoU, deterministic)
    tid = 0
    for i, blobs in enumerate(layer):
        t = i * FR
        new_tcur = {}
        if i > 0:
            prevblobs = layer[i - 1]
            used_prev = set()
            used_cur = set()
            cands = []
            for pj, pb in enumerate(prevblobs):
                for bj, b in enumerate(blobs):
                    ix = max(min(pb[3] + pb[5], b[3] + b[5]) - max(pb[3], b[3]), 0)
                    iy = max(min(pb[4] + pb[6], b[4] + b[6]) - max(pb[4], b[4]), 0)
                    inter = ix * iy
                    uni = pb[2] + b[2] - inter
                    iou = inter / uni if uni > 0 else 0
                    cdist = math.hypot(pb[0] - b[0], pb[1] - b[1])
                    if iou >= iou_thr and cdist <= cent_thr * math.hypot(W, H):
                        cands.append((iou, pj, bj, b))
            cands.sort(key=lambda c: -c[0])
            for iou, pj, bj, b in cands:
                if pj in used_prev or bj in used_cur:
                    continue
                used_prev.add(pj); used_cur.add(bj)
                tr = tcur.get(pj)
                if tr is None or tr.died:
                    continue
                tr.pts.append((t, b[0], b[1], b[2]))
                tr.dead_frames = 0
                new_tcur[bj] = tr
                # proximity (ball contact or blob merging) heuristic
                for other in tracks:
                    if other is tr or other.died:
                        continue
                    if len(other.pts) and abs(other.pts[-1][0] - t) < 1e-6 and \
                       math.hypot(other.pts[-1][1] - b[0], other.pts[-1][2] - b[1]) < 0.03 * math.hypot(W, H):
                        tr.overlaps += 1
            # unmatched prev blobs -> aging/death (with teleport reacquisition scan)
            for pj in range(len(prevblobs)):
                if pj in used_prev or pj not in tcur:
                    continue
                tr = tcur[pj]
                tr.dead_frames += 1
                if tr.dead_frames >= 2:
                    tr.died = True
                    medv = 2.0
                    if len(tr.pts) > 3:
                        ds = [math.hypot(tr.pts[q + 1][1] - tr.pts[q][1], tr.pts[q + 1][2] - tr.pts[q][2])
                              for q in range(len(tr.pts) - 1)]
                        medv = float(np.median([d / FR for d in ds])) if ds else 2.0
                    for k in range(i + 1, min(i + 4, len(layer))):
                        for b2 in layer[k]:
                            d2 = math.hypot(tr.pts[-1][1] - b2[0], tr.pts[-1][2] - b2[1])
                            expected = medv * (k - i) * FR
                            # teleport: similar-size blob far AHEAD of expected step
                            if d2 > max(3.0 * expected, 0.03 * W) and \
                               0.5 <= b2[2] / max(tr.pts[-1][3], 1) <= 2.0:
                                tr.jumps += 1
                                tr.max_jump_px = max(tr.max_jump_px, d2)
                                tr.pts.append((k * FR, b2[0], b2[1], b2[2]))
                                tr.rebirth_tracked = True
                                break
        for bj, b in enumerate(blobs):
            if bj in used_cur:
                continue
            tr = Track(tid, t, b[0], b[1], b[2])
            tracks.append(tr)
            new_tcur[bj] = tr
            tid += 1
        tcur = new_tcur

    # attribute short-lived (<=3 frames) fragment blobs as jumps of a nearby track
    frags = [tr for tr in tracks if len(tr.pts) <= 3]
    frag_total = len(frags)
    keep = [tr for tr in tracks if len(tr.pts) > 3]
    for frag in frags:
        fx, fy, fa = frag.pts[0][1], frag.pts[0][2], frag.pts[0][3]
        for s in keep:
            if s.died:
                continue
            if abs(s.pts[-1][0] - frag.pts[0][0]) > 0.35:
                continue
            d2 = math.hypot(s.pts[-1][1] - fx, s.pts[-1][2] - fy)
            ds = [math.hypot(s.pts[q + 1][1] - s.pts[q][1], s.pts[q + 1][2] - s.pts[q][2])
                  for q in range(len(s.pts) - 1)]
            med = float(np.median([d / FR for d in ds])) if ds else 2.0
            if d2 > max(3.0 * med * FR * 1.0, 0.03 * W) and 0.5 <= fa / max(s.pts[-1][3], 1) <= 2.0:
                s.jumps += 1
                s.max_jump_px = max(s.max_jump_px, d2)
                break
    tracks = keep

    # cross-frame "same-size blob appears far from every previous-frame blob" events
    # (robust teleport / ghost proxy: an object-magnitude blob with no plausible predecessor)
    sync_events = []
    spawn_count = 0
    for i in range(1, len(layer)):
        prev_layer = layer[i - 1]
        for b in layer[i]:
            best = 1e9
            for pb in prev_layer:
                if 0.35 <= max(b[2], 1) / max(pb[2], 1) <= 3.0:
                    best = min(best, math.hypot(b[0] - pb[0], b[1] - pb[1]))
            if best > 0.08 * W:
                if best < 1e8:
                    sync_events.append({"t": round(i * FR, 2), "d_pxW": round(best / W, 3),
                                        "area": int(b[2])})
                else:
                    spawn_count += 1

    stats = []
    for tr in tracks:
        if len(tr.pts) < 2:
            continue
        xs = [p[1] / W for p in tr.pts]  # normalize
        ys = [p[2] / H for p in tr.pts]
        ts = [p[0] for p in tr.pts]
        st = tr.step_stats(W)
        life = tr.t_end - tr.t_start
        if life < 0.4:
            continue
        stats.append({
            "id": f"g{tr.id}" if False else f"tr{tr.id}",
            "life": round(life, 2), "start_t": round(ts[0], 2),
            "start_norm": [round(xs[0], 3), round(ys[0], 3)],
            "end_norm": [round(xs[-1], 3), round(ys[-1], 3)],
            "path_norm": [[round(xs[j], 3), round(ys[j], 3)] for j in range(0, len(xs), max(1, len(xs) // 24))],
            "ts_norm": [round(ts[j], 2) for j in range(0, len(ts), max(1, len(ts) // 24))],
            "area_px": int(tr.pts[-1][3]), "dist": st["dist"], "med_vW": st["med_v"] / W,
            "jumps": tr.jumps, "max_jump_pxW": round(tr.max_jump_px / W, 3),
            "overlaps": tr.overlaps, "quality": round(len(tr.pts) / (n + 1e-9), 2),
        })
    stats.sort(key=lambda s: -s["life"])
    return {"fps": fps, "n_frames": n, "W": W, "H": H, "corrupt_frames": corrupt_frames,
            "tracks": stats, "frame_sync_events": sync_events,
            "spawn_count": spawn_count, "fragment_count": frag_total}


def track_at_time(tr, t):
    """Interpolated normalized position of a track at time t (None if outside life)."""
    ts = tr["ts_norm"]
    path = tr["path_norm"]
    if t < ts[0] - 1e-3 or t > ts[-1] + 1e-3:
        return None
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    return (float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys)))


def overlay_sheet(video_path, tracks_out, scale=0.5, sample_step=6, max_cells=20):
    """Draw each track as a line through its actual (time, position) points, plus
    the full path history at each sampled frame; red X marks any jump steps."""
    frames, fps = read_video(video_path, scale=scale)
    n = len(frames)
    idx = list(range(0, n, sample_step))[:max_cells]
    if len(idx) < max_cells and idx[-1] != n - 1:
        idx = idx[:-1] + [n - 1]
    cells = []
    for i in idx:
        t = i / fps
        im = Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)).resize(
            (336, int(336 * frames[i].shape[0] / frames[i].shape[1])))
        dr = ImageDraw.Draw(im)
        for tr in tracks_out["tracks"]:
            # path as polyline in normalized coords (path_norm sampled ~equally in time)
            pts = [(p[0] * im.width, (1 - p[1]) * im.height) for p in tr["path_norm"]]
            if len(pts) > 1:
                dr.line(pts, fill=(255, 255, 0), width=2)
            pos = track_at_time(tr, t)
            if pos:
                px, py = pos[0] * im.width, (1 - pos[1]) * im.height
                dr.ellipse([px - 7, py - 7, px + 7, py + 7], outline=(0, 255, 0), width=3)
            if tr["jumps"]:
                t0 = None
                # draw a red X at the jump location(s) (end of track life region)
                dr.text((4, 16), f"jumps={tr['jumps']}", fill=(255, 60, 60))
        dr.text((4, 2), f"f{i:04d}@{t:.2f}s", fill=(255, 255, 0))
        cells.append(im)
    w = max(c.width for c in cells)
    h = max(c.height for c in cells)
    cols = min(len(cells), 5)
    sheet = Image.new("RGB", (w * cols, h * math.ceil(len(cells) / cols)), (20, 20, 20))
    for k, c in enumerate(cells):
        sheet.paste(c, ((k % cols) * w, (k // cols) * h))
    return sheet


# ---------------------------------------------------------------- alignment
def _resample_ts(pts, grid_fps=8.0):
    """pts: [(t,x,y,area)] -> [[x,y] at 1/grid_fps grid] (linear interp)."""
    ts = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    t0, t1 = ts[0], ts[-1]
    grid = np.arange(t0, t1 + 1e-9, 1.0 / grid_fps)
    xs_i = np.interp(grid, ts, xs)
    ys_i = np.interp(grid, ts, ys)
    return grid, np.stack([xs_i, ys_i], axis=1), (t0, t1)


def dtw_align(a, b, motion_a=None, motion_b=None, band_w=2, diag_w=2.0):
    """Constrained monotonic DTW with motion-aware cost on normalized paths.
    a,b: Nx2 arrays (same coordinate space, normalized). motion_*: per-grid motion amp."""
    n, m = len(a), len(b)
    if motion_a is None:
        motion_a = np.zeros(n)
    if motion_b is None:
        motion_b = np.zeros(m)
    C = np.zeros((n, m))
    D = np.full((n, m), np.inf)
    P = {}
    for i in range(n):
        for j in range(m):
            if abs(i / max(n - 1, 1) - j / max(m - 1, 1)) * max(n - 1, 1) > band_w * 1.0:
                C[i, j] = np.inf
                continue
            cost = np.linalg.norm(a[i] - b[j]) / (1.0 + motion_a[i] + motion_b[j])
            C[i, j] = cost
    D[0, 0] = C[0, 0]
    for i in range(n):
        for j in range(m):
            if not np.isfinite(C[i, j]):
                continue
            steps = [((i - 1, j - 1), diag_w), ((i - 1, j), 1.0), ((i, j - 1), 1.0)]
            for (pi, pj), w in steps:
                if pi >= 0 and pj >= 0 and np.isfinite(D[pi, pj]):
                    if D[pi, pj] + w * C[i, j] < D[i, j]:
                        D[i, j] = D[pi, pj] + w * C[i, j]
                        P[(i, j)] = (pi, pj)
    if not np.isfinite(D[n - 1, m - 1]):
        return None
    path = [(n - 1, m - 1)]
    while path[-1] != (0, 0):
        path.append(P[path[-1]])
    path.reverse()
    resid = np.array([math.hypot(*(a[i] - b[j])) for i, j in path])
    return {"cost": float(D[n - 1, m - 1]), "n": n, "m": m, "path": path,
            "mean_resid": float(resid.mean()), "max_resid": float(resid.max()),
            "resid": resid.tolist()}


def align_pair(gen_pts, gt_pts, gen_fs=None, gt_fs=None):
    """Full pipeline for one track pair: resample to 4Hz, DTW. Returns dict."""
    g_grid, g_xy, g_t = _resample_ts(gen_pts)
    t_grid, t_xy, t_t = _resample_ts(gt_pts)
    # motion amp resampled
    def amp_grid(fs, grid):
        if fs is None:
            return None
        mg = np.asarray(fs["mags"])
        fps = fs["fps"]
        idx = np.clip((grid * fps).astype(int), 0, len(mg) - 1)
        return mg[idx] / (mg.max() + 1e-9)
    ga = amp_grid(gen_fs, g_grid)
    ta = amp_grid(gt_fs, t_grid)
    # normalize both to same scale using gen frame width? Both normalized already (x/W) with own W; use gen W as reference scale later.
    r = dtw_align(g_xy, t_xy, ga, ta)
    if r is None:
        return None
    return {"gen_t": [g_grid[0], g_grid[-1]], "gt_t": [t_grid[0], t_grid[-1]],
            "warp_every1s": [[round(float(t_grid[j]), 2), round(float(g_grid[i]), 2)]
                             for (i, j) in r["path"][::max(1, len(r["path"]) // 12)]],
            "mean_resid_fracW": round(r["mean_resid"] * 1.0, 4),  # normalized-path units
            "max_resid_fracW": round(r["max_resid"], 4),
            "dtw_cost": round(r["cost"], 3),
            "path_len": len(r["path"])}


def auto_pair(gen_tracks, gt_tracks, gen_fs, gt_fs, time_tol=1.2):
    """Greedy pairing: spatial start-dist <= 0.15, direction cos >= 0.5, AND
    temporal overlap: |start_t diff| <= time_tol, shared window >= 0.4s."""
    pairs = []
    for g in gen_tracks:
        gv = np.array(g["path_norm"])
        gd = gv[-1] - gv[0] if len(gv) > 1 else np.zeros(2)
        for t in gt_tracks:
            tv = np.array(t["path_norm"])
            td = tv[-1] - tv[0] if len(tv) > 1 else np.zeros(2)
            if abs(g["start_t"] - t["start_t"]) > time_tol:
                continue
            overlap = min(g["start_t"] + g["life"], t["start_t"] + t["life"]) - max(g["start_t"], t["start_t"])
            if overlap < 0.4:
                continue
            if np.linalg.norm(np.array(g["start_norm"]) - np.array(t["start_norm"])) > 0.15:
                continue
            cos = float(np.dot(gd, td) / (np.linalg.norm(gd) * np.linalg.norm(td) + 1e-9))
            if cos < 0.5:
                continue
            pairs.append((g, t, cos))
    return pairs
