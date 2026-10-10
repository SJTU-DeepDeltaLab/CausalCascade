#!/usr/bin/env python3
"""Judge-facing tool registry over tools_core, with per-scene disk precompute."""
import base64, hashlib, io, json, math, os, re
from functools import lru_cache
import numpy as np
import cv2
from PIL import Image, ImageDraw

from tools_core import (flow_summary, track_video, overlay_sheet,
                        auto_pair, align_pair, segments_from_flow,
                        bgdiff_mask, read_video, _resample_ts)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_cache')
os.makedirs(CACHE_DIR, exist_ok=True)


def _b64_img(im, max_w=900):
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.BILINEAR)
    buf = io.BytesIO()
    im.convert('RGB').save(buf, 'JPEG', quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode()


class Ctx:
    """Per-scene context: precomputed flow/tracks for gen and gt."""
    def __init__(self, scene_key, gen_video, gt_video, mode='B'):
        self.scene_key = scene_key
        self.gen = os.path.realpath(gen_video)
        self.gt = os.path.realpath(gt_video)
        self.mode = mode
        self._cache = {}
        def source_signature(path):
            stat = os.stat(path)
            return [path, stat.st_size, stat.st_mtime_ns]

        digest = hashlib.sha256(json.dumps(
            [source_signature(self.gen), source_signature(self.gt), mode],
            sort_keys=True,
        ).encode()).hexdigest()[:16]
        slug = re.sub(r'[^A-Za-z0-9_.-]+', '_', scene_key)[:64]
        self.cache_key = f"{slug}__{digest}"

    def _load(self, which, key):
        ck = f"{self.cache_key}__{which}__{key}"
        p = os.path.join(CACHE_DIR, ck + '.json')
        if os.path.exists(p):
            return json.load(open(p))
        fn = self.gen if which == 'gen' else self.gt
        if key == 'flow':
            val = flow_summary(fn)
        elif key == 'tracks':
            val = track_video(fn)
        elif key == 'segments':
            val = segments_from_flow(flow_summary(fn))
        else:
            raise KeyError(key)
        json.dump(val, open(p, 'w'))
        return val

    def flow(self, which):
        return self._load(which, 'flow')

    def tracks(self, which):
        return self._load(which, 'tracks')

    def segments(self, which):
        return self._load(which, 'segments')


@lru_cache(maxsize=16)
def _shared_context_image(path: str, size: int, mtime_ns: int) -> str:
    """Read one stable contact sheet and encode it for repeated API prefixes.

    The bounded cache avoids retaining all 50 scene pairs in memory at once.
    The contact-sheet files themselves are persisted by ``sheets.py`` and are
    invalidated when their source video signature changes.
    """
    # Keep the original PNG payload as the default.  If the provider rejects
    # an image, agent_client.py performs a one-time compact-image retry.
    with open(path, 'rb') as fh:
        return base64.standard_b64encode(fh.read()).decode('ascii')


def _cached_sheet_payload(video_path: str, key: str) -> dict:
    from sheets import sheet_for_video

    png_path, meta_path = sheet_for_video(video_path, key, force=False)
    stat = os.stat(png_path)
    data = _shared_context_image(str(png_path), stat.st_size, stat.st_mtime_ns)
    meta = json.load(open(meta_path))
    return {
        'type': 'image',
        'source': {'type': 'base64', 'media_type': 'image/png', 'data': data},
    }


def build_shared_context(ctx):
    """Return a stable GEN/GT visual prefix shared by all questions of a scene.

    The prefix contains only visual reference material and metadata.  It never
    includes a question, answer option, simulator answer, or judge conclusion.
    Keeping it before the question makes the repeated video context eligible
    for the provider's prefix cache across question units.
    """
    gen_key = f"{ctx.cache_key}__gen_shared_context"
    gt_key = f"{ctx.cache_key}__gt_shared_context"
    return [
        {
            'type': 'text',
            'text': (
                'SHARED SCENE CONTEXT (visual reference only). The next two images are '
                'the generated rollout and the simulator reference for the same scene. '
                'Use them to identify participants and compare the requested process; '
                'do not copy or infer an answer from the reference. The question and '
                'answer options appear after this stable context.'
            ),
        },
        {'type': 'text', 'text': 'GENERATED ROLLOUT CONTACT SHEET:'},
        _cached_sheet_payload(ctx.gen, gen_key),
        {'type': 'text', 'text': 'GROUND-TRUTH ROLLOUT CONTACT SHEET:'},
        _cached_sheet_payload(ctx.gt, gt_key),
    ]


# ------------------------------------------------------------------ tools
def tool_observe(ctx, args):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sheets import sheet_for_video
    which = args.get('video', 'gen')
    vp = ctx.gen if which == 'gen' else ctx.gt
    key = f"{ctx.cache_key}__{which}"
    png, meta_path = sheet_for_video(vp, key, force=False)
    meta = json.load(open(meta_path))
    im = Image.open(png)
    # Keep the 20-frame contact sheet close to its native display width.  The
    # previous 900px cap compressed each 336px cell to roughly 180px, which
    # made small balls, dominoes, and short-lived duplicate instances hard to
    # inspect.  This is still a compact single image, but retains enough
    # spatial detail for semantic visual judgment.
    return {'image': _b64_img(im, max_w=1680), 'text': f"contact sheet {which}: {meta['n_frames']} frames, "
            f"{meta['fps']} fps, indices={meta['indices']}"}


def tool_flow_info(ctx, args):
    which = args.get('video', 'gen')
    fs = ctx.flow(which)
    region = args.get('region')  # normalized [x0,y0,x1,y1] or None
    rows = []
    for i in range(0, len(fs['mags']), 4):
        rows.append(f"{round(i / fs['fps'], 2):5.2f}s  mag={fs['mags'][i]:.3f}")
    rows = rows[-20:]
    seg = ctx.segments(which)['active_windows']
    return {'text': f"global flow ({which}) fps={fs['fps']}; peak={fs['p95']:.3f}\n"
                    + "\n".join(rows) +
                    "\nactive windows: " + json.dumps(seg[:8])}


def tool_segments(ctx, args):
    which = args.get('video', 'gen')
    seg = ctx.segments(which)
    return {'text': f"active windows ({which}): " + json.dumps(seg['active_windows'][:10])
            + f"\nthreshold={seg['threshold']}"}


def _track_table(tr):
    rows = []
    for t in tr['tracks'][:8]:
        rows.append(f"{t['id']:5} t={t['start_t']:.2f}-{t['start_t']+t['life']:.2f}s "
                    f"start={t['start_norm']} end={t['end_norm']} dist={t['dist']}"
                    f" jumps={t['jumps']} maxjW={t['max_jump_pxW']} q={t['quality']}")
    return "\n".join(rows)


def tool_tracks(ctx, args):
    which = args.get('video', 'gen')
    tr = ctx.tracks(which)
    txt = (f"tracks ({which}): n={len(tr['tracks'])} fragments={tr['fragment_count']} "
           f"sync_events={len(tr['frame_sync_events'])} spawn={tr['spawn_count']}\n"
           + _track_table(tr))
    txt = ("IMPORTANT: this is a generic pixel/blob tracker, not semantic object tracking. "
           "A fragment, spawn, jump, or failed correspondence can be caused by color/shape "
           "change, occlusion, camera/viewpoint change, or background segmentation. It is "
           "never by itself proof of duplication, disappearance, or teleportation; verify "
           "such claims in a visual zoom.\n" + txt)
    if tr['frame_sync_events']:
        txt += ("\nsync_events (same-size blob far from all predecessors): "
                + json.dumps(tr['frame_sync_events'][:6]))
    if tr['corrupt_frames']:
        txt += (f"\nWARNING: {len(tr['corrupt_frames'])} frames flagged as scene-corruption "
                f"(motion >35% of frame): first few={tr['corrupt_frames'][:8]}; blobs on those "
                f"frames were EXCLUDED. Many corruption frames => the scene collapses, "
                f"which itself is a severe inconsistency (not continuity).")
    if not tr['tracks'] and not tr['corrupt_frames']:
        txt += "\nNOTE: no tracks over the whole video (nothing moved or below detection). "
    res = {'text': txt}
    if args.get('overlay'):
        res['image'] = _b64_img(overlay_sheet(ctx.gen if which == 'gen' else ctx.gt, tr))
    return res


def tool_align(ctx, args):
    gn = ctx.tracks('gen')
    gt = ctx.tracks('gt')
    pairs = auto_pair(gn['tracks'], gt['tracks'], ctx.flow('gen'), ctx.flow('gt'))
    if not pairs:
        return {'text': (f"align unavailable: structure mismatch — gen tracks={len(gn['tracks'])} "
                         f"fragments={gn['fragment_count']} vs gt tracks={len(gt['tracks'])}; "
                         f"no time/space-correspondable pair. This itself is evidence of divergence.")}
    row = []
    al = None
    for g, t, cos in pairs[:3]:
        gp = [(ts, pn[0], pn[1], 0) for ts, pn in zip(g['ts_norm'], g['path_norm'])]
        tp = [(ts, pn[0], pn[1], 0) for ts, pn in zip(t['ts_norm'], t['path_norm'])]
        a = align_pair(gp, tp, ctx.flow('gen'), ctx.flow('gt'))
        if a is None:
            row.append(f"{g['id']}<->{t['id']} cos={cos:.2f}: unavailable (life mismatch)")
            continue
        al = a
        row.append(f"{g['id']}<->{t['id']} cos={cos:.2f} mean_residW={a['mean_resid_fracW']} "
                   f"max={a['max_resid_fracW']} dtw={a['dtw_cost']}")
    alive = [g for g in gn['tracks'] if g['life'] > 1.0]
    cov = round(sum(g['life'] for g, _, _ in pairs) / max(sum(g['life'] for g in alive), 1e-6), 3)
    return {'text': "align results:\n" + "\n".join(row) + f"\ncoverage={cov}"}


def tool_traj_diff(ctx, args):
    gn = ctx.tracks('gen')
    gt = ctx.tracks('gt')
    pairs = auto_pair(gn['tracks'], gt['tracks'], ctx.flow('gen'), ctx.flow('gt'))
    if not pairs:
        return {'text': "traj_diff unavailable: no correspondable pair (structure mismatch)."}
    g, t, cos = pairs[0]
    gp = [(ts, pn[0], pn[1], 0) for ts, pn in zip(g['ts_norm'], g['path_norm'])]
    tp = [(ts, pn[0], pn[1], 0) for ts, pn in zip(t['ts_norm'], t['path_norm'])]
    a = align_pair(gp, tp, ctx.flow('gen'), ctx.flow('gt'))
    if a is None:
        return {'text': f"traj_diff unavailable: {g['id']}<->{t['id']} life mismatch."}
    m = a['mean_resid_fracW']
    verdict = 'consistent' if m <= 0.035 else ('partial' if m <= 0.10 else 'mismatch')
    return {'text': (f"pair {g['id']}<->{t['id']}: mean_residW={m} max={a['max_resid_fracW']} "
                     f"-> {verdict}\nwarp(every 1s): " + json.dumps(a['warp_every1s'][:12]))}


def tool_zoom(ctx, args):
    which = args.get('video', 'gen')
    vp = ctx.gen if which == 'gen' else ctx.gt
    t0 = float(args.get('t0', 0))
    t1 = float(args.get('t1', t0 + 1.5))
    region = args.get('region')  # [x0,y0,x1,y1] normalized
    scale = float(args.get('scale', 2.0))
    frames, fps = read_video(vp, scale=1.0)
    i0, i1 = max(0, int(t0 * fps)), min(len(frames), int(t1 * fps) + 1)
    if i1 <= i0:
        return {'text': 'zoom: empty window'}
    requested_times = args.get('times')
    if requested_times:
        indices = []
        for value in requested_times[:12]:
            try:
                # Times are seconds; frame indices can also be supplied when the
                # caller explicitly uses a value outside the clip duration.
                idx = int(round(float(value) * fps))
            except Exception:
                continue
            if i0 <= idx < i1:
                indices.append(idx)
        indices = sorted(set(indices))
        if not indices:
            return {'text': 'zoom: requested times do not fall inside this window'}
    else:
        # Six samples were sufficient for broad motion but missed short terminal
        # intervals and brief contacts.  Use up to ten evenly spaced samples by
        # default; callers can still request exact `times` when an event boundary
        # is suspected.
        step = max(1, (i1 - i0) // 10)
        indices = list(range(i0, i1, step))
    cells = []
    for i in indices:
        f = frames[i]
        if region:
            x0, y0, x1, y1 = [float(v) for v in region]
            h, w = f.shape[:2]
            if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
                xa, xb = int(x0 * w), int(x1 * w)
                ya, yb = int(y0 * h), int(y1 * h)
            else:
                xa, xb, ya, yb = int(x0), int(x1), int(y0), int(y1)
            xa, xb = max(0, xa), min(w, xb)
            ya, yb = max(0, ya), min(h, yb)
            if xb <= xa or yb <= ya:
                return {'text': 'zoom: invalid crop region; use normalized [0,1] or pixel coordinates'}
            f = f[ya:yb, xa:xb]
        im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f.ndim == 3 else f)
        # A two-column layout keeps each sampled frame large enough to inspect
        # small balls, domino tips, and brief duplicate instances.  The former
        # three-column 420px layout made the relevant object only a handful of
        # pixels wide in a full-frame crop.
        dw = 600
        im = im.resize((dw, int(dw * im.height / im.width)), Image.BILINEAR)
        d = ImageDraw.Draw(im)
        d.text((4, 2), f"f{i:04d}@{i / fps:.2f}s", fill=(255, 255, 0))
        cells.append(im)
    w = max(c.width for c in cells)
    hgt = max(c.height for c in cells)
    cols = min(2, len(cells))
    sheet = Image.new('RGB', (w * cols, hgt * math.ceil(len(cells) / cols)), (20, 20, 20))
    for k, c in enumerate(cells):
        sheet.paste(c, ((k % cols) * w, (k // cols) * hgt))
    return {'image': _b64_img(sheet, max_w=1260),
            'text': f"zoom {which} [{t0}..{t1}s] frames {i0}-{i1}"}


def tool_paired_zoom(ctx, args):
    """Render generated and reference frames at identical nominal times.

    The paired view is a visual aid only: it does not compute or expose an answer.
    Keeping the two rollouts in the same image makes it easier to identify the named
    participant and compare the exact event endpoint without making the judge shuttle
    between separate tool results.
    """
    t0 = float(args.get('t0', 0))
    t1 = float(args.get('t1', t0 + 2.0))
    region = args.get('region')
    times = args.get('times')
    if times:
        ts = [float(v) for v in times[:12]]
    else:
        n = 8
        ts = [t0 + (t1 - t0) * j / max(n - 1, 1) for j in range(n)]
    videos = [('GEN', ctx.gen), ('GT', ctx.gt)]
    rows = []
    for label, vp in videos:
        frames, fps = read_video(vp, scale=1.0)
        cells = []
        for t in ts:
            i = max(0, min(len(frames) - 1, int(round(t * fps))))
            f = frames[i]
            if region:
                x0, y0, x1, y1 = [float(v) for v in region]
                h, w = f.shape[:2]
                if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
                    xa, xb = int(x0 * w), int(x1 * w)
                    ya, yb = int(y0 * h), int(y1 * h)
                else:
                    xa, xb, ya, yb = int(x0), int(x1), int(y0), int(y1)
                xa, xb = max(0, xa), min(w, xb)
                ya, yb = max(0, ya), min(h, yb)
                if xb <= xa or yb <= ya:
                    return {'text': 'paired_zoom: invalid crop region'}
                f = f[ya:yb, xa:xb]
            im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f.ndim == 3 else f)
            dw = 600
            im = im.resize((dw, int(dw * im.height / im.width)), Image.BILINEAR)
            ImageDraw.Draw(im).text((4, 2), f"{label} {t:.2f}s", fill=(255, 255, 0))
            cells.append(im)
        rows.append(cells)
    w = max(c.width for row in rows for c in row)
    h = max(c.height for row in rows for c in row)
    sheet = Image.new('RGB', (w * len(ts), h * 2), (20, 20, 20))
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            sheet.paste(cell, (ci * w, ri * h))
    # Eight paired cells at 320px are deliberately kept near their native
    # width; shrinking this strip to 1200px made each cell too small to compare
    # endpoint orientation and brief contacts reliably.
    return {'image': _b64_img(sheet, max_w=3360),
            'text': f"paired zoom GEN/GT at times {ts} [{t0}..{t1}s]"}


def tool_diff_pairs(ctx, args):
    times = args.get('times') or [0.5]
    return {'text': 'diff_pairs: implemented in a later iteration; use align+traj_diff instead'}


def tool_temporal_pair(ctx, args):
    """Show exact moments from one rollout side-by-side.

    This visual aid is useful for subtle support deformation, short-lived
    duplicates, identity switches, endpoint orientation, and terminal-state
    transitions.  It intentionally returns no computed verdict.
    """
    which = args.get('video', 'gen')
    vp = ctx.gen if which == 'gen' else ctx.gt
    try:
        ts = [float(v) for v in (args.get('times') or [0.0, 1.0])[:6]]
    except Exception:
        return {'text': 'temporal_pair: times must be numeric seconds'}
    frames, fps = read_video(vp, scale=1.0)
    if not frames:
        return {'text': 'temporal_pair: empty video'}
    region = args.get('region')
    cells = []
    for t in ts:
        i = max(0, min(len(frames) - 1, int(round(t * fps))))
        f = frames[i]
        if region:
            x0, y0, x1, y1 = [float(v) for v in region]
            h, w = f.shape[:2]
            if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
                xa, xb = int(x0 * w), int(x1 * w)
                ya, yb = int(y0 * h), int(y1 * h)
            else:
                xa, xb, ya, yb = int(x0), int(x1), int(y0), int(y1)
            xa, xb = max(0, xa), min(w, xb)
            ya, yb = max(0, ya), min(h, yb)
            if xb <= xa or yb <= ya:
                return {'text': 'temporal_pair: invalid crop region'}
            f = f[ya:yb, xa:xb]
        im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f.ndim == 3 else f)
        dw = 600
        im = im.resize((dw, int(dw * im.height / im.width)), Image.BILINEAR)
        ImageDraw.Draw(im).text((4, 2), f"{which.upper()} {t:.2f}s (f{i})", fill=(255, 255, 0))
        cells.append(im)
    cols = min(2, len(cells))
    w = max(c.width for c in cells)
    h = max(c.height for c in cells)
    sheet = Image.new('RGB', (w * cols, h * math.ceil(len(cells) / cols)), (20, 20, 20))
    for k, c in enumerate(cells):
        sheet.paste(c, ((k % cols) * w, (k // cols) * h))
    return {'image': _b64_img(sheet, max_w=1200),
            'text': f"temporal pair {which} at times {ts}; compare the same named objects across moments"}


TOOLS = {
    'observe': {'desc': 'contact sheet of the whole video',
                'schema': {'type': 'object', 'properties': {'video': {'enum': ['gen', 'gt'], 'default': 'gen'}}},
                'run': tool_observe, 'mode': 'AB'},
    'flow_info': {'desc': 'global optical-flow magnitude timeline',
                  'schema': {'type': 'object', 'properties': {'video': {'enum': ['gen', 'gt'], 'default': 'gen'}}},
                  'run': tool_flow_info, 'mode': 'AB'},
    'tracks': {'desc': 'generic pixel/blob-track diagnostics only (not semantic object identities); use visual zoom to verify any jump, spawn, disappearance, or duplication; overlay=true adds an annotated sheet',
               'schema': {'type': 'object', 'properties': {'video': {'enum': ['gen', 'gt'], 'default': 'gen'},
                                                            'overlay': {'type': 'boolean', 'default': False}}},
               'run': tool_tracks, 'mode': 'AB'},
    'segments': {'desc': 'active/still temporal windows',
                 'schema': {'type': 'object', 'properties': {'video': {'enum': ['gen', 'gt'], 'default': 'gen'}}},
                 'run': tool_segments, 'mode': 'AB'},
    'zoom': {'desc': 'crop a short time window at higher granularity',
             'schema': {'type': 'object', 'properties': {
                 'video': {'enum': ['gen', 'gt'], 'default': 'gen'},
                 't0': {'type': 'number'}, 't1': {'type': 'number'},
                 'times': {'type': 'array', 'items': {'type': 'number'},
                            'description': 'optional exact seconds to sample within [t0,t1]'},
                 'region': {'type': 'array', 'minItems': 4, 'maxItems': 4,
                            'description': '[x0,y0,x1,y1], normalized to [0,1] or in pixels'},
                 'scale': {'type': 'number', 'default': 2.0}}},
             'run': tool_zoom, 'mode': 'AB'},
    'paired_zoom': {'desc': 'side-by-side generated/GT frames at the same times; visual comparison aid, not an answer or semantic annotation',
                    'schema': {'type': 'object', 'properties': {
                        't0': {'type': 'number'}, 't1': {'type': 'number'},
                        'times': {'type': 'array', 'items': {'type': 'number'},
                                  'description': 'optional exact seconds to sample'},
                        'region': {'type': 'array', 'minItems': 4, 'maxItems': 4,
                                   'description': '[x0,y0,x1,y1], normalized to [0,1] or in pixels'}}},
                    'run': tool_paired_zoom, 'mode': 'B'},
    'temporal_pair': {'desc': 'side-by-side moments from one rollout at exact times; visual aid for endpoint, identity, support, trajectory, and terminal comparisons',
                      'schema': {'type': 'object', 'properties': {
                          'video': {'enum': ['gen', 'gt'], 'default': 'gen'},
                          'times': {'type': 'array', 'items': {'type': 'number'},
                                    'description': 'exact seconds, up to six moments'},
                          'region': {'type': 'array', 'minItems': 4, 'maxItems': 4,
                                     'description': '[x0,y0,x1,y1], normalized to [0,1] or in pixels'}}},
                      'run': tool_temporal_pair, 'mode': 'AB'},
    'align': {'desc': 'GT<->gen track alignment (structure check first)',
              'schema': {'type': 'object', 'properties': {}}, 'run': tool_align, 'mode': 'B'},
    'traj_diff': {'desc': 'aligned trajectory residual for the best pair',
                  'schema': {'type': 'object', 'properties': {}}, 'run': tool_traj_diff, 'mode': 'B'},
}


def system_tool_doc(mode):
    lines = []
    for name, t in TOOLS.items():
        if mode == 'A' and t['mode'] == 'B':
            continue
        lines.append(f"- {name}: {t['desc']}")
    return "\n".join(lines)


def exec_tool(name, args, ctx):
    if name not in TOOLS:
        return {"text": f"unknown tool {name}"}
    if ctx.mode == 'A' and TOOLS[name]['mode'] == 'B':
        return {"text": f"tool {name} not available in this mode (GT hidden)"}
    if ctx.mode == 'A' and (args or {}).get('video') == 'gt':
        return {"text": f"tool {name}: GT access is disabled in mode A"}
    try:
        return TOOLS[name]['run'](ctx, args or {})
    except Exception as e:
        return {"text": f"tool {name} error: {e}"}
