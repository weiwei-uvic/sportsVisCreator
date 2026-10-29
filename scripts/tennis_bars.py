#!/usr/bin/env python3
"""
tennis_bars.py
==============
Segment tennis players with SAM 3 and draw a horizontal bar gauge above each
player's head. The bar always faces the camera (it is not drawn flat on the
court) but scales with the player's distance: it gets smaller as the player
moves away and bigger as they come closer, exactly like an object of fixed
real size hovering above them.

Bar size matches the ring of tennis_donuts.py, in metres:
    white/red band length  = ring perimeter = 2 pi x r                  (5.34 m)
                             r = --radius, the outer edge of the ring's
                             white/red band (0.85 m); --perimeter-at center
                             or inner measures it at the middle (4.46 m) or
                             inner edge (3.58 m) of the band instead
    white/red band height  = ring band width       = --ring-width   (0.28 m)
    black outline          = ring outline, all round = --outline    (0.045 m)
So a value fills the same length of red on the bar as on the ring.
Colours and look are the same as the ring: black outline #000000, white band
#FFFFFF, red #E90738 filling the band from the left by `value`
(0.25 -> left quarter red), solid, exact colours.

Standalone. The SAM 3 cache is the same as tennis_donuts.py's
('<video name>.sam3cache/tennis-player/'), so frames already segmented by
either script are reused and SAM 3 is not run again for them.

How the scale is found
----------------------
The 4 court corners give the court->image homography H. At the player's
floor point, a 1 m segment across the court (court x direction, which is
parallel to the image for a broadcast camera behind the baseline) is
projected through H; its length in pixels is the metres-to-pixels scale at
that depth. For a pinhole camera an upright object at the same depth has the
same scale, so the bar is sized with it, and the player's height (head top
above the floor point) is measured with it too, which places the bar a fixed
real distance (--gap) above the head.

Install
-------
    # SAM 3 (needs Python 3.12+, PyTorch 2.7+, CUDA GPU)
    git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .
    hf auth login        # after access to facebook/sam3 is granted on Hugging Face
    pip install opencv-python numpy

Usage
-----
    python tennis_bars.py match.mp4 out.mp4 --values 0.25 0.7 \
        --court "575,305;1345,305;1640,905;280,905"

    --values takes NEAR player first, then FAR player.
    --values-csv file with columns frame,near,far (interpolated) for values over time.
    --court corners: far-left, far-right, near-right, near-left (doubles court;
    add --singles for singles corners). Without --court you click them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field

# reduce CUDA memory fragmentation (must be set before torch is imported)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
from tqdm import tqdm


# Court geometry (metres). Origin = far-left doubles corner, x to the right,
# y towards the camera.
COURT_LEN = 23.77
DOUBLES_W = 10.97
SINGLES_W = 8.23
NET_Y = COURT_LEN / 2.0


# ===========================================================================
# SAM 3 wrapper
# ===========================================================================


@dataclass
class Detection:
    mask: np.ndarray  # bool HxW
    score: float


class Sam3PersonSegmenter:
    """Thin wrapper around the official SAM 3 image API (facebookresearch/sam3)."""

    def __init__(self, prompt: str = "tennis player", device: str = "cuda", bf16_weights: bool = False):
        import torch
        from PIL import Image  # noqa: F401  (imported here so the module loads without SAM 3)
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.torch = torch
        self.prompt = prompt
        self.device = device
        model = build_sam3_image_model()
        if bf16_weights:
            # halves the weights' GPU memory (~3.4 GB -> ~1.7 GB) for small GPUs
            model = cast_weights_bf16(model)
        self.processor = Sam3Processor(model)
        self.use_amp = device.startswith("cuda") and torch.cuda.is_available()

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        from PIL import Image

        torch = self.torch
        h, w = frame_bgr.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if self.use_amp
               else torch.autocast("cpu", enabled=False))
        with torch.inference_mode(), ctx:
            state = self.processor.set_image(image)
            out = self.processor.set_text_prompt(state=state, prompt=self.prompt)

        masks = _to_numpy(out["masks"])
        scores = _to_numpy(out["scores"]).reshape(-1)
        if masks.size == 0:
            return []
        if masks.ndim == 4:          # (N, 1, H, W)
            masks = masks[:, 0]
        if masks.ndim == 2:          # single mask
            masks = masks[None]

        dets = []
        for m, s in zip(masks, scores):
            m = m.astype(np.float32)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            dets.append(Detection(mask=m > 0.5, score=float(s)))
        return dets


def cast_weights_bf16(model):
    """
    Store a SAM 3 model's weights in bfloat16 (about half the GPU memory).

    SAM 3 turns automatic mixed precision off in a few places (e.g. its
    decoder's feed-forward layers), where a float32 input would then hit a
    bfloat16 weight and fail. A pre-hook on every layer that owns weights
    converts floating-point inputs to that layer's dtype, so the whole model
    runs consistently in bfloat16.
    """
    import torch

    model.to(torch.bfloat16)

    def match_dtype(module, args, kwargs):
        dt = module._weights_dtype

        def cast(x):
            if torch.is_tensor(x) and x.is_floating_point() and x.dtype != dt:
                return x.to(dt)
            return x

        return tuple(cast(a) for a in args), {k: cast(v) for k, v in kwargs.items()}

    for m in model.modules():
        p = next(m.parameters(recurse=False), None)
        if p is not None and p.is_floating_point():
            m._weights_dtype = p.dtype
            m.register_forward_pre_hook(match_dtype, with_kwargs=True)
    return model


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


# ===========================================================================
# Court homography
# ===========================================================================


def court_corners_m(singles: bool) -> np.ndarray:
    if singles:
        x0 = (DOUBLES_W - SINGLES_W) / 2
        x1 = x0 + SINGLES_W
    else:
        x0, x1 = 0.0, DOUBLES_W
    # far-left, far-right, near-right, near-left
    return np.float32([[x0, 0], [x1, 0], [x1, COURT_LEN], [x0, COURT_LEN]])


def parse_points(s: str) -> np.ndarray:
    pts = [tuple(map(float, p.split(","))) for p in s.strip().split(";") if p.strip()]
    if len(pts) != 4:
        raise ValueError("--court needs exactly 4 points: 'x,y;x,y;x,y;x,y'")
    return np.float32(pts)


def click_court_corners(frame: np.ndarray) -> np.ndarray:
    """Let the user click the 4 court corners on the first frame."""
    labels = ["FAR-LEFT", "FAR-RIGHT", "NEAR-RIGHT", "NEAR-LEFT"]
    pts: list[tuple[int, int]] = []
    win = "Click court corners (Backspace=undo, Enter=done, Esc=quit)"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(vis, labels[i], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        if len(pts) > 1:
            cv2.polylines(vis, [np.int32(pts)], len(pts) == 4, (0, 255, 255), 2, cv2.LINE_AA)
        msg = f"Click {labels[len(pts)]} corner" if len(pts) < 4 else "Press Enter to confirm"
        cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts) == 4:
            break
        if k == 8 and pts:
            pts.pop()
        if k == 27:
            sys.exit("Aborted.")
    cv2.destroyWindow(win)
    print("Court corners:", ";".join(f"{x},{y}" for x, y in pts),
          "(pass with --court to skip clicking next time)")
    return np.float32(pts)


def to_court(H_inv: np.ndarray, pt_img) -> np.ndarray:
    p = cv2.perspectiveTransform(np.float32([[pt_img]]), H_inv)
    return p[0, 0]


def to_image(H: np.ndarray, pts_court: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts_court.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


# ===========================================================================
# Players: floor point, head, selection
# ===========================================================================


def ground_point(mask: np.ndarray, torso=(0.20, 0.55), feet_band=0.12,
                 planted_tol=0.10) -> tuple[float, float] | None:
    """
    Point on the floor directly under the centre of the body silhouette.

    x: median column of the torso band (rows 20-55 % from the top of the mask),
       i.e. the body's vertical axis. The median ignores an outstretched
       racket arm.
    y: floor line. The lowest 12 % of the mask (feet/lower legs) is split
       into left/right of the body axis, and each side's lowest pixel is a
       foot contact. Contacts within 10 % of body height of the lowest one
       count as planted and are averaged (in a wide stance the back foot sits
       higher in the image because of perspective). A clearly lifted foot is
       ignored, so the ring stays on the floor between the planted feet.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 30:
        return None
    top, bot = int(ys.min()), int(ys.max())
    height = bot - top + 1

    band = (ys >= top + torso[0] * height) & (ys <= top + torso[1] * height)
    cx = float(np.median(xs[band])) if band.sum() >= 10 else float(np.median(xs))

    fb = ys >= bot - feet_band * height
    fx, fy = xs[fb], ys[fb]
    contacts = [float(fy[side].max()) for side in (fx < cx, fx >= cx) if side.sum() >= 5]
    tol = max(2.0, planted_tol * height)
    planted = [c for c in contacts if c >= bot - tol] or [float(bot)]
    return cx, float(np.mean(planted))


@dataclass
class PlayerTrack:
    side: str                       # "near" | "far"
    pos: np.ndarray | None = None   # smoothed court position (m)
    missing: int = 0
    history: list = field(default_factory=list)

    def update(self, new_pos: np.ndarray | None, alpha: float, max_missing: int):
        if new_pos is None:
            self.missing += 1
            if self.missing > max_missing:
                self.pos = None
            return
        self.missing = 0
        if self.pos is None:
            self.pos = new_pos.astype(np.float64)
        else:
            # snap if the jump is huge (e.g. new rally / wrong previous pick)
            if np.linalg.norm(new_pos - self.pos) > 4.0:
                self.pos = new_pos.astype(np.float64)
            else:
                self.pos = alpha * new_pos + (1 - alpha) * self.pos


def select_players(dets: list[Detection], H_inv: np.ndarray, x_lo: float, x_hi: float,
                   tracks: dict[str, PlayerTrack], args) -> dict[str, tuple[np.ndarray, Detection] | None]:
    """
    Pick one detection per side of the net (same rules as tennis_donuts.py),
    returning its court position AND the detection, whose mask gives the
    top of the head.
    """
    cands = {"near": [], "far": []}
    for d in dets:
        if d.score < args.player_thresh:
            continue
        fp = ground_point(d.mask)
        if fp is None:
            continue
        c = to_court(H_inv, fp)
        if not (x_lo - args.side_margin <= c[0] <= x_hi + args.side_margin):
            continue
        if not (-args.back_margin <= c[1] <= COURT_LEN + args.back_margin):
            continue
        side = "near" if c[1] > NET_Y else "far"
        cands[side].append((d.score, c, d))

    picked = {}
    for side, lst in cands.items():
        if not lst:
            picked[side] = None
            continue
        prev = tracks[side].pos
        if prev is not None:
            best = max(lst, key=lambda sc: sc[0] - 0.15 * np.linalg.norm(sc[1] - prev))
        else:
            best = max(lst, key=lambda sc: sc[0])
        picked[side] = (best[1], best[2])
    return picked


def head_top_y(mask: np.ndarray, axis_x: float) -> float | None:
    """
    Top of the head: highest mask pixel in a narrow band around the body
    axis, so a racket held out to the side does not count.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    half = max(3.0, 0.08 * (ys.max() - ys.min() + 1))
    sel = np.abs(xs - axis_x) <= half
    return float(ys[sel].min()) if sel.any() else float(ys.min())


def scale_at(H: np.ndarray, pos_m: np.ndarray) -> float:
    """Pixels per metre at a court position (1 m across the court, projected)."""
    p = to_image(H, np.float32([[pos_m[0] - 0.5, pos_m[1]], [pos_m[0] + 0.5, pos_m[1]]]))
    return float(np.linalg.norm(p[1] - p[0]))


# ===========================================================================
# Bar renderer
# ===========================================================================


WHITE = np.array([230, 230, 230], np.float32)  # #E6E6E6
RED = np.array([56, 7, 233], np.float32)       # #E90738 (OpenCV stores BGR)
BLACK = np.array([60, 60, 60], np.float32)        # #3C3C3C


@dataclass
class BarStyle:
    band_len: float = 5.34     # m, = ring perimeter 2 pi r
    band_h: float = 0.28       # m, = ring band width
    outline: float = 0.045     # m, = ring outline
    gap: float = 0.25          # m, between the top of the head and the bar
    opacity: float = 0.75
    fill_from: str = "left"    # where the red fill starts: left | right
    soften: float = 0.6        # px blur of the bar's silhouette only
    ss: int = 4                # supersampling factor


@dataclass
class HeightTrack:
    """Smoothed player height (m) used to place the bar above the head."""
    h_m: float | None = None

    def update(self, new_h: float | None, alpha: float = 0.3):
        if new_h is None:
            return
        self.h_m = new_h if self.h_m is None else alpha * new_h + (1 - alpha) * self.h_m


def render_bar(frame: np.ndarray, cx: float, bottom_y: float, ppm: float, value: float,
               st: BarStyle) -> None:
    """
    Composite one bar into `frame` (in place). The bar is screen-aligned,
    centred on x = cx, with its lower outline edge at y = bottom_y, sized
    with `ppm` pixels per metre.
    """
    h, w = frame.shape[:2]
    value = float(np.clip(value, 0.0, 1.0))
    L, T, o = st.band_len * ppm, st.band_h * ppm, st.outline * ppm
    W_out, H_out = L + 2 * o, T + 2 * o
    left, top = cx - W_out / 2, bottom_y - H_out
    if top < 2:                                    # keep it inside the frame
        bottom_y += 2 - top
        top = 2.0

    pad = 3
    x0, y0 = int(np.floor(left)) - pad, int(np.floor(top)) - pad
    x1, y1 = int(np.ceil(left + W_out)) + pad, int(np.ceil(top + H_out)) + pad
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return
    rw, rh, ss = x1 - x0, y1 - y0, st.ss

    def rect_mask(rx0, ry0, rx1, ry1):
        m = np.zeros((rh * ss, rw * ss), np.uint8)
        if rx1 - rx0 <= 0 or ry1 - ry0 <= 0:
            return m.astype(np.float32)
        pts = (np.float32([[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1]]) - [x0, y0]) * ss
        cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
        return m.astype(np.float32) / 255.0

    a_all = rect_mask(left, top, left + W_out, top + H_out)            # outline + band
    a_band = rect_mask(left + o, top + o, left + o + L, top + o + T)    # white band
    if st.fill_from == "right":
        a_red = rect_mask(left + o + L * (1 - value), top + o, left + o + L, top + o + T)
    else:
        a_red = rect_mask(left + o, top + o, left + o + L * value, top + o + T)

    def down(a):
        return cv2.resize(a, (rw, rh), interpolation=cv2.INTER_AREA)

    a_all, a_band, a_red = down(a_all), down(a_band), down(a_red)
    if st.soften > 0:   # soften only the silhouette; inner boundaries stay crisp
        a_all = cv2.GaussianBlur(a_all, (0, 0), st.soften)

    col = np.empty((rh, rw, 3), np.float32)
    col[:] = BLACK
    col = col * (1 - a_band[..., None]) + WHITE * a_band[..., None]
    col = col * (1 - a_red[..., None]) + RED * a_red[..., None]
    alpha = a_all * st.opacity

    sx, sy = x0c - x0, y0c - y0
    ex, ey = sx + (x1c - x0c), sy + (y1c - y0c)
    col, alpha = col[sy:ey, sx:ex], alpha[sy:ey, sx:ex]
    bg = frame[y0c:y1c, x0c:x1c].astype(np.float32)
    out = bg * (1 - alpha[..., None]) + col * alpha[..., None]
    frame[y0c:y1c, x0c:x1c] = np.clip(out, 0, 255).astype(np.uint8)


# ===========================================================================
# Values, cache, progress
# ===========================================================================


def load_values_csv(path: str) -> dict[int, tuple[float, float]]:
    vals = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            vals[int(row["frame"])] = (float(row["near"]), float(row["far"]))
    return vals


def values_at(frame_idx: int, table: dict[int, tuple[float, float]] | None,
              default: tuple[float, float]) -> tuple[float, float]:
    """Linear interpolation between CSV keyframes, constant outside."""
    if not table:
        return default
    keys = sorted(table)
    if frame_idx <= keys[0]:
        return table[keys[0]]
    if frame_idx >= keys[-1]:
        return table[keys[-1]]
    i = np.searchsorted(keys, frame_idx)
    k0, k1 = keys[i - 1], keys[i]
    t = (frame_idx - k0) / (k1 - k0)
    a, b = np.array(table[k0]), np.array(table[k1])
    return tuple(a + t * (b - a))


def video_fingerprint(path: str, chunk: int = 4 << 20) -> str:
    """Fast identity check: file size + SHA-1 of the first and last 4 MB."""
    size = os.path.getsize(path)
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(chunk, size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()


class SegCache:
    """
    Stores raw SAM 3 detections (all masks + scores, before any player
    selection or thresholds) as one small compressed file per frame:

        <video name>.sam3cache/<prompt>/meta.json
        <video name>.sam3cache/<prompt>/frame_000123.npz

    Masks are cropped to their bounding box and bit-packed, so a frame with a
    few players takes only a few KB. Frames are keyed by absolute frame index,
    so --start/--max-frames runs and interrupted runs reuse whatever exists.
    If the video file changes, the cache for it is rebuilt automatically.
    """

    VERSION = 1

    def __init__(self, video_path: str, prompt: str, frame_shape, root: str | None = None,
                 refresh: bool = False):
        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-") or "prompt"
        root = root or os.path.splitext(video_path)[0] + ".sam3cache"
        self.dir = os.path.join(root, slug)
        self.shape = tuple(frame_shape[:2])
        meta = {"version": self.VERSION, "video": os.path.basename(video_path),
                "fingerprint": video_fingerprint(video_path), "prompt": prompt,
                "height": self.shape[0], "width": self.shape[1]}
        meta_path = os.path.join(self.dir, "meta.json")

        stale = False
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    old = json.load(f)
                stale = any(old.get(k) != meta[k] for k in ("version", "fingerprint", "height", "width"))
            except (OSError, ValueError):
                stale = True
        if (refresh or stale) and os.path.isdir(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _path(self, idx: int) -> str:
        return os.path.join(self.dir, f"frame_{idx:06d}.npz")

    def count(self) -> int:
        return sum(1 for n in os.listdir(self.dir) if n.startswith("frame_") and n.endswith(".npz"))

    def load(self, idx: int) -> list[Detection] | None:
        p = self._path(idx)
        if not os.path.exists(p):
            return None
        try:
            z = np.load(p)
            scores, boxes, bits, offs = z["scores"], z["boxes"], z["bits"], z["offsets"]
        except Exception:
            return None                      # unreadable -> recompute this frame
        dets = []
        for i in range(len(scores)):
            y0, x0, y1, x1 = (int(v) for v in boxes[i])
            n = (y1 - y0) * (x1 - x0)
            crop = np.unpackbits(bits[offs[i]:offs[i + 1]], count=n).reshape(y1 - y0, x1 - x0)
            mask = np.zeros(self.shape, bool)
            mask[y0:y1, x0:x1] = crop.astype(bool)
            dets.append(Detection(mask=mask, score=float(scores[i])))
        return dets

    def save(self, idx: int, dets: list[Detection]) -> None:
        scores, boxes, chunks, offs = [], [], [], [0]
        for d in dets:
            ys, xs = np.nonzero(d.mask)
            if len(ys) == 0:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            packed = np.packbits(d.mask[y0:y1, x0:x1].ravel())
            scores.append(d.score)
            boxes.append((y0, x0, y1, x1))
            chunks.append(packed)
            offs.append(offs[-1] + len(packed))
        tmp = self._path(idx) + ".tmp.npz"
        np.savez_compressed(
            tmp,
            scores=np.float32(scores),
            boxes=np.int32(boxes).reshape(-1, 4),
            bits=np.concatenate(chunks) if chunks else np.zeros(0, np.uint8),
            offsets=np.int64(offs),
        )
        os.replace(tmp, self._path(idx))    # atomic: an interrupted run never leaves a broken file


class FrameProgress:
    """
    tqdm bar over frames. A background thread refreshes it every 0.2 s, so the
    current stage ("SAM 3 ...") and its running time stay live even while the
    main thread is blocked inside a SAM 3 call.
    """

    def __init__(self, total: int | None):
        self.bar = tqdm(total=total, unit="frame", dynamic_ncols=True,
                        desc="Processing", smoothing=0.1)
        self.lock = threading.Lock()
        self.stage = "starting"
        self.stage_t0 = time.perf_counter()
        self.sam_avg = None        # EMA of SAM 3 seconds per frame
        self.players = "-"
        self.n_cached = 0          # frames read from the cache
        self.n_sam = 0             # frames run through SAM 3
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_stage(self, stage: str):
        with self.lock:
            self.stage = stage
            self.stage_t0 = time.perf_counter()

    def record_seg(self, n_players: int, sam_seconds: float | None):
        """sam_seconds=None means the detections came from the cache."""
        with self.lock:
            if sam_seconds is None:
                self.n_cached += 1
            else:
                self.n_sam += 1
                self.sam_avg = (sam_seconds if self.sam_avg is None
                                else 0.8 * self.sam_avg + 0.2 * sam_seconds)
            self.players = str(n_players)

    def step(self):
        with self.lock:
            self.bar.update(1)
            self._refresh_locked()

    def write(self, msg: str):
        with self.lock:
            self.bar.write(msg)

    def _refresh_locked(self):
        running = time.perf_counter() - self.stage_t0
        sam = "-" if self.sam_avg is None else f"{self.sam_avg:.2f}s/frame"
        self.bar.set_description(f"{self.stage} {running:4.1f}s", refresh=False)
        self.bar.set_postfix_str(f"cached {self.n_cached}, SAM 3 {self.n_sam} ({sam}), "
                                 f"players {self.players}", refresh=False)
        self.bar.refresh()

    def _run(self):
        while not self._stop.wait(0.2):
            with self.lock:
                self._refresh_locked()

    def close(self):
        self._stop.set()
        self._thread.join()
        with self.lock:
            self.stage = "done"
            self._refresh_locked()
            self.bar.close()


def scene_signature(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


# ===========================================================================
# Main
# ===========================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--values", type=float, nargs=2, default=[0.25, 0.25], metavar=("NEAR", "FAR"),
                    help="fraction of the bar that is red for the near / far player (0..1)")
    ap.add_argument("--values-csv", help="CSV with columns frame,near,far (interpolated)")
    ap.add_argument("--court", help="4 image points 'x,y;x,y;x,y;x,y': far-left, far-right, near-right, near-left")
    ap.add_argument("--singles", action="store_true", help="the 4 points are singles-court corners")
    ap.add_argument("--prompt", default="tennis player", help="SAM 3 text prompt")
    ap.add_argument("--player-thresh", type=float, default=0.5, help="min score to be a player candidate")
    ap.add_argument("--side-margin", type=float, default=4.0, help="m allowed outside the sidelines")
    ap.add_argument("--back-margin", type=float, default=7.0, help="m allowed behind the baselines")
    # bar size: same settings (and defaults) as the ring in tennis_donuts.py
    ap.add_argument("--radius", type=float, default=0.85,
                    help="ring radius (m) of tennis_donuts.py; bar band length = ring perimeter")
    ap.add_argument("--perimeter-at", choices=["outer", "center", "inner"], default="outer",
                    help="where the ring perimeter is measured: outer edge of the white/red band "
                         "(2 pi radius, default), its middle, or its inner edge")
    ap.add_argument("--ring-width", type=float, default=0.28,
                    help="ring band width (m) of tennis_donuts.py; = bar band height")
    ap.add_argument("--outline", type=float, default=0.045, help="black outline width (m)")
    ap.add_argument("--gap", type=float, default=0.25, help="space between head and bar (m)")
    ap.add_argument("--fill-from", choices=["left", "right"], default="left",
                    help="side of the bar where the red fill starts")
    ap.add_argument("--opacity", type=float, default=1.0, help="bar opacity (1 = solid)")
    ap.add_argument("--smooth", type=float, default=0.55, help="EMA factor for position (1 = no smoothing)")
    ap.add_argument("--hold", type=int, default=8, help="frames to keep a bar when the player is lost")
    ap.add_argument("--scene-check", type=float, default=0.5,
                    help="skip drawing when the frame no longer looks like the calibrated shot "
                         "(histogram correlation below this; 0 disables). Handles broadcast camera cuts.")
    ap.add_argument("--start", type=int, default=0, help="first frame to process")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--debug", action="store_true", help="draw masks, body axis, floor and head points")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16-weights", action="store_true",
                    help="store SAM 3 weights in bfloat16: about half the GPU memory, for GPUs under ~8 GB")
    ap.add_argument("--cache-dir", help="where to keep SAM 3 results "
                    "(default: '<video name>.sam3cache' next to the video, shared with tennis_donuts.py)")
    ap.add_argument("--no-cache", action="store_true", help="don't read or write the SAM 3 cache")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="delete this video's cached SAM 3 results for this prompt and recompute")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    ok, first = cap.read()
    if not ok:
        sys.exit("Video has no frames.")

    img_pts = parse_points(args.court) if args.court else click_court_corners(first)
    court_pts = court_corners_m(args.singles)
    H = cv2.getPerspectiveTransform(court_pts, img_pts)       # court m -> image px
    H_inv = np.linalg.inv(H)                                   # image px -> court m
    x_lo, x_hi = float(court_pts[:, 0].min()), float(court_pts[:, 0].max())
    ref_sig = scene_signature(first)

    r_perim = {"outer": args.radius, "center": args.radius - args.ring_width / 2,
               "inner": args.radius - args.ring_width}[args.perimeter_at]
    style = BarStyle(band_len=2 * np.pi * r_perim, band_h=args.ring_width, outline=args.outline,
                     gap=args.gap, opacity=args.opacity, fill_from=args.fill_from)
    table = load_values_csv(args.values_csv) if args.values_csv else None

    cache = None
    if not args.no_cache:
        cache = SegCache(args.input, args.prompt, first.shape, root=args.cache_dir,
                         refresh=args.refresh_cache)
        print(f"SAM 3 cache: {cache.dir} ({cache.count()} frames already cached)")

    segmenter = None   # SAM 3 is loaded only when a frame is missing from the cache

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tracks = {"near": PlayerTrack("near"), "far": PlayerTrack("far")}
    heights = {"near": HeightTrack(), "far": HeightTrack()}

    n_todo = max(0, total - args.start) if total > 0 else None
    if args.max_frames:
        n_todo = args.max_frames if n_todo is None else min(n_todo, args.max_frames)
    progress = FrameProgress(n_todo)
    n_skipped = 0

    idx = args.start
    n_done = 0
    frame = first
    while True:
        same_shot = (args.scene_check <= 0 or
                     cv2.compareHist(ref_sig, scene_signature(frame), cv2.HISTCMP_CORREL) >= args.scene_check)
        out = frame.copy()

        if same_shot:
            progress.set_stage("Cache")
            dets = cache.load(idx) if cache else None
            sam_seconds = None
            if dets is None:
                if segmenter is None:
                    progress.set_stage("Loading SAM 3")
                    segmenter = Sam3PersonSegmenter(prompt=args.prompt, device=args.device,
                                                    bf16_weights=args.bf16_weights)
                    progress.write("SAM 3 loaded.")
                progress.set_stage("SAM 3 warm-up" if progress.n_sam == 0 else "SAM 3")
                t0 = time.perf_counter()
                dets = segmenter(frame)
                sam_seconds = time.perf_counter() - t0
                if cache:
                    cache.save(idx, dets)
            n_players = sum(d.score >= args.player_thresh for d in dets)
            progress.record_seg(n_players, sam_seconds)
            progress.set_stage("Rendering")

            picked = select_players(dets, H_inv, x_lo, x_hi, tracks, args)
            for side in ("near", "far"):
                pos = picked[side][0] if picked[side] else None
                tracks[side].update(pos, args.smooth, args.hold)
                if tracks[side].pos is None:
                    heights[side].h_m = None
                elif picked[side]:
                    d = picked[side][1]
                    gx, gy = to_image(H, np.float32([pos]))[0]
                    top = head_top_y(d.mask, gx)
                    ppm = scale_at(H, pos)
                    if top is not None and ppm > 0:
                        heights[side].update((gy - top) / ppm)

            v_near, v_far = values_at(idx, table, tuple(args.values))
            # far bar first, then near (the near one is closer to the camera)
            for side, val in (("far", v_far), ("near", v_near)):
                pos, hm = tracks[side].pos, heights[side].h_m
                if pos is None or hm is None:
                    continue
                ppm = scale_at(H, pos)
                gx, gy = to_image(H, np.float32([pos]))[0]
                render_bar(out, float(gx), float(gy - (hm + style.gap) * ppm), ppm, val, style)

            if args.debug:
                for d in dets:
                    color = (0, 255, 0) if d.score >= args.player_thresh else (0, 140, 255)
                    cnts, _ = cv2.findContours(d.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(out, cnts, -1, color, 1, cv2.LINE_AA)
                    gp = ground_point(d.mask)
                    if gp:
                        ty = head_top_y(d.mask, gp[0])
                        cv2.line(out, (int(gp[0]), int(ty)), (int(gp[0]), int(gp[1])),
                                 (255, 0, 255), 1, cv2.LINE_AA)
                        cv2.circle(out, (int(gp[0]), int(gp[1])), 4, (255, 0, 255), -1)
                        cv2.circle(out, (int(gp[0]), int(ty)), 4, (255, 255, 0), -1)
                cv2.polylines(out, [np.int32(img_pts)], True, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            progress.set_stage("Skipped (other camera)")
            n_skipped += 1
            for side in tracks:          # new shot: forget positions
                tracks[side].pos, tracks[side].missing = None, 0
                heights[side].h_m = None

        progress.set_stage("Writing")
        writer.write(out)
        n_done += 1
        idx += 1
        progress.step()
        if args.max_frames and n_done >= args.max_frames:
            break
        progress.set_stage("Reading")
        ok, frame = cap.read()
        if not ok:
            break

    progress.close()
    cap.release()
    writer.release()
    print(f"SAM 3 ran on {progress.n_sam} frames, {progress.n_cached} came from the cache.")
    print(f"Done: {n_done} frames -> {args.output}"
          + (f" ({n_skipped} frames from other camera angles left untouched)" if n_skipped else ""))
    print("Tip: re-attach the original audio with\n"
          f"  ffmpeg -i {args.output} -i {args.input} -map 0:v -map 1:a? -c:v libx264 -crf 18 -c:a copy final.mp4")


if __name__ == "__main__":
    main()