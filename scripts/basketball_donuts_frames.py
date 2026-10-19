#!/usr/bin/env python3
"""
basketball_donuts_frames.py
===========================
Donut rings under two basketball players, for small GPUs (about 6 GB) and
FIXED-camera clips.

SAM 3 runs in image mode, one frame at a time with only the detector loaded,
so GPU memory stays flat however long the clip is or however many players are
on court. Player identities are kept by this script's own tracker, which
matches players frame to frame by mask overlap, court position and jersey
colour. Then a donut ring is painted on the floor under TWO chosen players,
occluded by everyone standing on it.

Standalone: needs only SAM 3, OpenCV, NumPy and tqdm (SciPy optional).

Pipeline
--------
1. Court calibration on the first clip frame (4 court landmarks, see
   below). The camera is assumed not to move.
2. SAM 3 image model, prompts "basketball player" and "referee", frame by
   frame (the image is encoded once and both prompts reuse it). Results
   are cached per frame (shared with any other run on the same video and
   prompt), so it only ever runs once per frame.
3. Referees occlude the rings and are shown as REF, but never get player
   IDs. A person found by both prompts gets the higher-scoring label.
   People whose feet are off the court (bench, crowd) are ignored for
   identity, but still occlude the rings.
4. Identity tracking assigns stable IDs. A player hidden behind others keeps
   their ID for up to --max-age frames.
5. You choose the two players by ID (preview image or --pick), then the
   rings are rendered.

Usage
-----
    # 1st run: calibrate, segment, track; writes out_ids.jpg and stops
    python basketball_donuts_frames.py game.mp4 out.mp4 --start 1500 --max-frames 300 \
        --court-preset left-half --bf16-weights

    # pick the two players by ID (no SAM 3 needed any more)
    python basketball_donuts_frames.py game.mp4 out.mp4 --start 1500 --max-frames 300 \
        --court-preset left-half --court "x,y;x,y;x,y;x,y" --targets 4 9 --values 0.25 0.6

Court calibration
-----------------
Coordinates are in metres. Origin is the far-left court corner as seen on
screen, x runs along the court to the right and y runs towards the camera.
Click or list 4 points in this order: FAR-LEFT, FAR-RIGHT, NEAR-RIGHT,
NEAR-LEFT, for the rectangle chosen with --court-preset:

    full        whole court (4 corners)
    left-half   left baseline corners + both ends of the half-court line
    right-half  both ends of the half-court line + right baseline corners
    left-paint  4 corners of the left painted area (key)
    right-paint 4 corners of the right painted area (key)

--rules fiba (default, 28 x 15 m) or nba (28.65 x 15.24 m).
Advanced: any 4+ landmarks as "u,v=X,Y" pairs (pixels = metres), e.g.
    --court "412,288=0,0;1260,301=14,0;1700,860=14,15;95,840=0,15"

IDs depend on the clip start, so keep --start the same between the preview
run and the render run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import csv
import hashlib
import json
import re
import shutil
import threading
from dataclasses import dataclass, field

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
from tqdm import tqdm

# ===========================================================================
# SAM 3 image segmentation (frame by frame)
# ===========================================================================


@dataclass
class Detection:
    mask: np.ndarray  # bool HxW
    score: float


class Sam3PersonSegmenter:
    """Thin wrapper around the official SAM 3 image API (facebookresearch/sam3)."""

    def __init__(self, prompt: str = "basketball player", device: str = "cuda", bf16_weights: bool = False):
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
        return self.segment(frame_bgr, [self.prompt])[self.prompt]

    def segment(self, frame_bgr: np.ndarray, prompts: list[str]) -> dict[str, list[Detection]]:
        """Run several text prompts on one frame; the image is encoded only once."""
        from PIL import Image

        torch = self.torch
        h, w = frame_bgr.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if self.use_amp
               else torch.autocast("cpu", enabled=False))
        results = {}
        with torch.inference_mode(), ctx:
            state = self.processor.set_image(image)
            for prompt in prompts:
                out = self.processor.set_text_prompt(state=state, prompt=prompt)
                results[prompt] = self._to_dets(out, h, w)
        return results

    @staticmethod
    def _to_dets(out, h: int, w: int) -> list[Detection]:
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
# Court geometry & calibration
# ===========================================================================


COURTS = {
    #        length  width  key width  key depth (from baseline)
    "fiba": dict(L=28.0, W=15.0, kw=4.9, kd=5.8),
    "nba": dict(L=28.65, W=15.24, kw=4.877, kd=5.791),
}


def preset_rect(name: str, c: dict) -> np.ndarray:
    """Court-metre corners in click order: far-left, far-right, near-right, near-left."""
    L, W, kw, kd = c["L"], c["W"], c["kw"], c["kd"]
    y0, y1 = (W - kw) / 2, (W + kw) / 2
    rects = {
        "full": [(0, 0), (L, 0), (L, W), (0, W)],
        "left-half": [(0, 0), (L / 2, 0), (L / 2, W), (0, W)],
        "right-half": [(L / 2, 0), (L, 0), (L, W), (L / 2, W)],
        "left-paint": [(0, y0), (kd, y0), (kd, y1), (0, y1)],
        "right-paint": [(L - kd, y0), (L, y0), (L, y1), (L - kd, y1)],
    }
    if name not in rects:
        sys.exit(f"Unknown --court-preset '{name}'. Choose from: {', '.join(rects)}")
    return np.float32(rects[name])


def calibrate(frame: np.ndarray, args, court: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (H court->image, image points used)."""
    if args.court and "=" in args.court:
        img, world = [], []
        for pair in args.court.split(";"):
            if not pair.strip():
                continue
            uv, xy = pair.split("=")
            img.append(tuple(map(float, uv.split(","))))
            world.append(tuple(map(float, xy.split(","))))
        if len(img) < 4:
            sys.exit("--court with '=' needs at least 4 'u,v=X,Y' pairs")
        img, world = np.float32(img), np.float32(world)
        H, _ = cv2.findHomography(world, img, 0)
        return H, img
    world = preset_rect(args.court_preset, court)
    if args.court:
        img = np.float32([tuple(map(float, p.split(","))) for p in args.court.split(";") if p.strip()])
        if len(img) != 4:
            sys.exit("--court needs exactly 4 points 'x,y;x,y;x,y;x,y'")
    else:
        print(f"Click the 4 corners of the '{args.court_preset}' rectangle "
              "(far-left, far-right, near-right, near-left).")
        img = click_court_corners(frame)
    return cv2.getPerspectiveTransform(world, img), img


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


def court_lines(court: dict) -> list[np.ndarray]:
    """Court markings (metres) for the --debug overlay."""
    L, W, kw, kd = court["L"], court["W"], court["kw"], court["kd"]
    y0, y1 = (W - kw) / 2, (W + kw) / 2
    t = np.linspace(0, 2 * np.pi, 64)
    circ = np.stack([L / 2 + 1.8 * np.cos(t), W / 2 + 1.8 * np.sin(t)], 1)
    return [
        np.float32([(0, 0), (L, 0), (L, W), (0, W), (0, 0)]),
        np.float32([(L / 2, 0), (L / 2, W)]),
        np.float32([(0, y0), (kd, y0), (kd, y1), (0, y1)]),
        np.float32([(L, y0), (L - kd, y0), (L - kd, y1), (L, y1)]),
        np.float32(circ),
    ]


def draw_court(img: np.ndarray, H: np.ndarray, court: dict) -> None:
    for line in court_lines(court):
        # densify so lines stay straight under perspective clipping
        dense = np.concatenate([np.linspace(a, b, 20) for a, b in zip(line[:-1], line[1:])])
        p = to_image(H, dense)
        ok = np.all(np.isfinite(p), 1) & (np.abs(p) < 1e5).all(1)
        if ok.sum() > 1:
            cv2.polylines(img, [np.int32(p[ok])], False, (0, 255, 255), 1, cv2.LINE_AA)


# ===========================================================================
# Floor point under a player
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


# ===========================================================================
# Donut ring renderer
# ===========================================================================


@dataclass
class DonutStyle:
    outer_r: float = 0.85     # metres
    ring_w: float = 0.28      # metres (white/red band)
    outline: float = 0.045    # metres (black outline, each side)
    start_deg: float = 90.0   # where the red arc starts: 0 = 12 o'clock (far baseline), 90 = 3 o'clock
    opacity: float = 0.92
    shade: float = 0.85       # 0 = flat colours, 1 = fully follow court lighting
    soften: float = 0.6       # px gaussian sigma to match footage softness
    ss: int = 4               # supersampling factor
    n_pts: int = 256


WHITE = np.array([245, 245, 245], np.float32)
RED = np.array([40, 30, 215], np.float32)      # BGR
BLACK = np.array([10, 10, 10], np.float32)


def _circle_pts(center: np.ndarray, r: float, n: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # theta = 0 -> far side (-y), increasing clockwise seen from above
    return np.stack([center[0] + r * np.sin(t), center[1] - r * np.cos(t)], 1)


def _wedge_pts(center: np.ndarray, r: float, a0: float, a1: float, n: int) -> np.ndarray:
    t = np.linspace(a0, a1, max(3, int(n * (a1 - a0) / (2 * np.pi)) + 2))
    arc = np.stack([center[0] + r * np.sin(t), center[1] - r * np.cos(t)], 1)
    return np.vstack([center[None], arc])


def render_donut(frame: np.ndarray, occluder: np.ndarray, H: np.ndarray,
                 center_m: np.ndarray, value: float, st: DonutStyle) -> None:
    """Composite one donut into `frame` (in place). `occluder` is float HxW in [0,1]."""
    h, w = frame.shape[:2]
    value = float(np.clip(value, 0.0, 1.0))
    R_out = st.outer_r + st.outline
    r_in = st.outer_r - st.ring_w
    r_in_out = max(0.0, r_in - st.outline)

    # projected footprint -> ROI
    big = to_image(H, _circle_pts(center_m, R_out, st.n_pts))
    pad = 4
    x0 = int(np.floor(big[:, 0].min())) - pad
    y0 = int(np.floor(big[:, 1].min())) - pad
    x1 = int(np.ceil(big[:, 0].max())) + pad
    y1 = int(np.ceil(big[:, 1].max())) + pad
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return
    rw, rh = x1 - x0, y1 - y0
    ss = st.ss

    def poly_mask(pts_m: np.ndarray) -> np.ndarray:
        p = (to_image(H, pts_m) - [x0, y0]) * ss
        m = np.zeros((rh * ss, rw * ss), np.uint8)
        cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
        return m.astype(np.float32) / 255.0

    def disk(r: float) -> np.ndarray:
        if r <= 0:
            return np.zeros((rh * ss, rw * ss), np.float32)
        return poly_mask(_circle_pts(center_m, r, st.n_pts))

    d_outline_out = disk(R_out)
    d_outer = disk(st.outer_r)
    d_inner = disk(r_in)
    d_outline_in = disk(r_in_out)

    a_outline = np.clip(d_outline_out - d_outline_in, 0, 1)   # whole black band
    a_ring = np.clip(d_outer - d_inner, 0, 1)                  # white/red band
    if value >= 0.999:
        a_red = a_ring
    elif value <= 0.001:
        a_red = np.zeros_like(a_ring)
    else:
        a0 = np.deg2rad(st.start_deg)
        wedge = poly_mask(_wedge_pts(center_m, R_out * 1.5, a0, a0 + 2 * np.pi * value, st.n_pts))
        a_red = a_ring * wedge

    # back to frame resolution (area filter = proper AA)
    def down(a):
        return cv2.resize(a, (rw, rh), interpolation=cv2.INTER_AREA)

    a_outline, a_ring, a_red = down(a_outline), down(a_ring), down(a_red)
    if st.soften > 0:
        k = (0, 0)
        a_outline = cv2.GaussianBlur(a_outline, k, st.soften)
        a_ring = cv2.GaussianBlur(a_ring, k, st.soften)
        a_red = cv2.GaussianBlur(a_red, k, st.soften)

    # colour layers: black -> white -> red
    col = np.empty((rh, rw, 3), np.float32)
    col[:] = BLACK
    col = col * (1 - a_ring[..., None]) + WHITE * a_ring[..., None]
    col = col * (1 - a_red[..., None]) + RED * a_red[..., None]
    alpha = a_outline * st.opacity

    # crop ROI to frame
    sx, sy = x0c - x0, y0c - y0
    ex, ey = sx + (x1c - x0c), sy + (y1c - y0c)
    col, alpha = col[sy:ey, sx:ex], alpha[sy:ey, sx:ex]
    bg = frame[y0c:y1c, x0c:x1c].astype(np.float32)

    # lighting: follow local court luminance (player shadows darken the paint)
    if st.shade > 0:
        gray = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # median filter removes thin painted court lines so they don't read as "light"
        k = 7 if min(gray.shape) > 7 else 3
        lum = cv2.medianBlur(gray, k).astype(np.float32)
        lum = cv2.GaussianBlur(lum, (0, 0), 2.0)
        ring_px = alpha > 0.5
        ref = np.median(lum[ring_px]) if ring_px.sum() > 20 else np.median(lum)
        # darken-only: shadows dim the paint, but surface colour changes don't brighten it
        gain = np.clip(lum / max(ref, 1.0), 0.35, 1.0)
        gain = 1 + st.shade * (gain - 1)
        col = col * gain[..., None]

    # occlusion by players (feet/legs stand "on top" of the painted ring)
    occ = occluder[y0c:y1c, x0c:x1c]
    alpha = alpha * (1 - occ)

    out = bg * (1 - alpha[..., None]) + col * alpha[..., None]
    frame[y0c:y1c, x0c:x1c] = np.clip(out, 0, 255).astype(np.uint8)


def build_occluder(dets: list[Detection], shape, thresh: float, feather: float) -> np.ndarray:
    occ = np.zeros(shape[:2], np.uint8)
    for d in dets:
        if d.score >= thresh:
            occ |= d.mask.astype(np.uint8)
    occ = cv2.dilate(occ, np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
    if feather > 0:
        occ = cv2.GaussianBlur(occ, (0, 0), feather)
    return np.clip(occ, 0, 1)


# ===========================================================================
# Per-frame SAM 3 cache
# ===========================================================================


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


# ===========================================================================
# Progress display
# ===========================================================================


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


# ===========================================================================
# Target selection & following
# ===========================================================================


@dataclass
class TrackedDet:
    obj_id: int
    mask: np.ndarray   # bool HxW
    score: float


def id_color(oid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(oid * 7919 + 13)
    hsv = np.uint8([[[rng.integers(0, 180), 200, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def id_preview(frame: np.ndarray, dets: list[TrackedDet], highlight=()) -> np.ndarray:
    vis = frame.copy()
    overlay = frame.copy()
    for d in dets:
        overlay[d.mask] = id_color(d.obj_id)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
    for d in dets:
        cnts, _ = cv2.findContours(d.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thick = 3 if d.obj_id in highlight else 1
        cv2.drawContours(vis, cnts, -1, id_color(d.obj_id), thick, cv2.LINE_AA)
        ys, xs = np.nonzero(d.mask)
        org = (int(np.median(xs)) - 12, max(25, int(ys.min()) - 10))
        label = str(d.obj_id)
        cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def pick_targets(frame, dets) -> list[int]:
    """Click two players; returns their SAM 3 IDs."""
    picked: list[int] = []
    win = "Click the 2 target players (Backspace=undo, Enter=done, Esc=quit)"

    def on_mouse(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN or len(picked) >= 2:
            return
        hits = [d for d in dets if d.mask[y, x]]
        if not hits:   # nearest mask centroid
            hits = sorted(dets, key=lambda d: np.hypot(*(np.argwhere(d.mask).mean(0) - (y, x))))
        if hits and hits[0].obj_id not in picked:
            picked.append(hits[0].obj_id)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = id_preview(frame, dets, highlight=picked)
        msg = f"Picked: {picked}" + ("  - press Enter" if len(picked) == 2 else "")
        cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(picked) == 2:
            break
        if k == 8 and picked:
            picked.pop()
        if k == 27:
            sys.exit("Aborted.")
    cv2.destroyWindow(win)
    return picked


class TargetFollower:
    """
    Follows one SAM 3 object ID. If the tracker drops it and later re-detects
    the same player under a NEW ID (after a heavy occlusion), the new ID is
    adopted when it appears close to the player's last known court position.
    """

    def __init__(self, obj_id: int, reacquire_m: float, newborn_frames: int):
        self.obj_id = obj_id
        self.reacquire_m = reacquire_m
        self.newborn_frames = newborn_frames
        self.track = PlayerTrack(str(obj_id))
        self.last_seen_pos = None

    def find(self, dets, by_id, first_seen, frame_idx, H_inv, taken) -> TrackedDet | None:
        d = by_id.get(self.obj_id)
        if d is not None or self.last_seen_pos is None:
            return d
        best, best_dist = None, self.reacquire_m
        for c in dets:
            if c.obj_id in taken or frame_idx - first_seen.get(c.obj_id, -10**9) > self.newborn_frames:
                continue
            gp = ground_point(c.mask)
            if gp is None:
                continue
            dist = float(np.linalg.norm(to_court(H_inv, gp) - self.last_seen_pos))
            if dist < best_dist:
                best, best_dist = c, dist
        if best is not None:
            tqdm.write(f"frame {frame_idx}: target {self.obj_id} re-identified as SAM 3 ID {best.obj_id}")
            self.obj_id = best.obj_id
        return best


# ===========================================================================
# Values
# ===========================================================================


def load_values_csv(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[int(r["frame"])] = (float(r["p1"]), float(r["p2"]))
    return rows


def values_at(idx, table, default):
    if not table:
        return default
    keys = sorted(table)
    if idx <= keys[0]:
        return table[keys[0]]
    if idx >= keys[-1]:
        return table[keys[-1]]
    i = int(np.searchsorted(keys, idx))
    k0, k1 = keys[i - 1], keys[i]
    t = (idx - k0) / (k1 - k0)
    return tuple(np.array(table[k0]) + t * (np.array(table[k1]) - np.array(table[k0])))


try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # greedy fallback
    linear_sum_assignment = None


# ===========================================================================
# Player identity tracker
# ===========================================================================
DS = 4  # mask downsampling for overlap computation


def torso_hist(frame_hsv: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Hue/saturation histogram of the torso (jersey colour)."""
    ys = np.nonzero(mask.any(1))[0]
    if len(ys) == 0:
        return None
    top, bot = ys.min(), ys.max()
    band = np.zeros_like(mask)
    y0, y1 = int(top + 0.2 * (bot - top)), int(top + 0.55 * (bot - top)) + 1
    band[y0:y1] = mask[y0:y1]
    if band.sum() < 20:
        band = mask
    h = cv2.calcHist([frame_hsv], [0, 1], band.astype(np.uint8), [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h, 1, 0, cv2.NORM_L1).flatten()


@dataclass
class PlayerObs:
    det_idx: int
    mask: np.ndarray
    small: np.ndarray        # downsampled mask
    pos: np.ndarray          # court metres
    hist: np.ndarray | None


@dataclass
class Track:
    tid: int
    pos: np.ndarray
    small: np.ndarray
    hist: np.ndarray | None
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    missing: int = 0


def small_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter / np.logical_or(a, b).sum())


def assign(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    pairs, used_r, used_c = [], set(), set()
    for flat in np.argsort(cost, axis=None):
        i, j = divmod(int(flat), cost.shape[1])
        if i not in used_r and j not in used_c:
            pairs.append((i, j))
            used_r.add(i)
            used_c.add(j)
    return pairs


class PlayerIDTracker:
    """
    Frame-to-frame identity assignment.

    Cost of matching a track to a detection combines:
      * mask overlap with the track's last mask (only if seen last frame)
      * court distance to the track's predicted position (gated)
      * jersey colour distance (Bhattacharyya)
    Unmatched detections start new IDs. Unmatched tracks are kept (coasting
    on their velocity) for up to max_age frames, so a briefly hidden player
    keeps their ID.
    """

    BIG = 1e6

    def __init__(self, gate_m=1.5, max_age=50):
        self.gate_m, self.max_age = gate_m, max_age
        self.tracks: list[Track] = []
        self.next_id = 1

    def _cost(self, t: Track, o: PlayerObs) -> float:
        pred = t.pos + t.vel * (t.missing + 1)
        gate = min(self.gate_m + 0.3 * t.missing, 4.0)
        d = float(np.linalg.norm(o.pos - pred))
        if d > gate:
            return self.BIG
        app = 0.5 if (t.hist is None or o.hist is None) else \
            float(cv2.compareHist(t.hist.astype(np.float32), o.hist.astype(np.float32),
                                  cv2.HISTCMP_BHATTACHARYYA))
        if t.missing == 0:
            return 0.45 * (1 - small_iou(t.small, o.small)) + 0.30 * d / gate + 0.25 * app
        return 0.60 * d / gate + 0.40 * app

    def step(self, obs: list[PlayerObs]) -> dict[int, int]:
        """Returns {det_idx: track id}."""
        cost = np.array([[self._cost(t, o) for o in obs] for t in self.tracks]).reshape(len(self.tracks), len(obs))
        matched_t, matched_o, out = set(), set(), {}
        for ti, oi in assign(cost):
            if cost[ti, oi] >= self.BIG / 2 or cost[ti, oi] > 0.75:
                continue
            t, o = self.tracks[ti], obs[oi]
            step = (o.pos - t.pos) / (t.missing + 1)
            t.vel = 0.6 * t.vel + 0.4 * step
            t.pos, t.small, t.missing = o.pos, o.small, 0
            if o.hist is not None:
                t.hist = o.hist if t.hist is None else 0.8 * t.hist + 0.2 * o.hist
            matched_t.add(ti)
            matched_o.add(oi)
            out[o.det_idx] = t.tid
        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.missing += 1
                t.vel *= 0.8
        self.tracks = [t for t in self.tracks if t.missing <= self.max_age]
        for oi, o in enumerate(obs):
            if oi not in matched_o:
                t = Track(self.next_id, o.pos, o.small, o.hist)
                self.next_id += 1
                self.tracks.append(t)
                out[o.det_idx] = t.tid
        return out


def mask_nms(dets, iou_thresh=0.7):
    """Drop duplicate masks of the same person (keep the higher score)."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i].score)
    smalls = {i: dets[i].mask[::DS, ::DS] for i in order}
    keep = []
    for i in order:
        if all(small_iou(smalls[i], smalls[k]) < iou_thresh for k in keep):
            keep.append(i)
    return sorted(keep)


def player_observations(frame, dets, H_inv, court, args) -> list[PlayerObs]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    L, W, m = court["L"], court["W"], args.court_margin
    obs = []
    for i in mask_nms(dets):
        d = dets[i]
        if d.score < args.det_thresh:
            continue
        gp = ground_point(d.mask)
        if gp is None:
            continue
        pos = to_court(H_inv, gp).astype(np.float64)
        if not (-m <= pos[0] <= L + m and -m <= pos[1] <= W + m):
            continue                      # bench / crowd
        obs.append(PlayerObs(i, d.mask, d.mask[::DS, ::DS], pos, torso_hist(hsv, d.mask)))
    return obs


# ===========================================================================
# Main
# ===========================================================================
def split_people(player_dets, ref_dets, ref_thresh, iou_thresh=0.5):
    """
    Resolve people found by both prompts. When a "player" mask and a
    "referee" mask cover the same person, the higher-scoring label wins.
    Returns (players, referees).
    """
    refs = [r for r in ref_dets if r.score >= ref_thresh]
    if not refs:
        return list(player_dets), []
    ps = [d.mask[::DS, ::DS] for d in player_dets]
    rs = [r.mask[::DS, ::DS] for r in refs]
    drop_p, drop_r = set(), set()
    for i, pm in enumerate(ps):
        for j, rm in enumerate(rs):
            if small_iou(pm, rm) > iou_thresh:
                if refs[j].score >= player_dets[i].score:
                    drop_p.add(i)
                else:
                    drop_r.add(j)
    return ([d for i, d in enumerate(player_dets) if i not in drop_p],
            [r for j, r in enumerate(refs) if j not in drop_r])


def draw_referees(vis: np.ndarray, refs) -> np.ndarray:
    """Grey overlay + 'REF' label (referees never get player IDs)."""
    if not refs:
        return vis
    overlay = vis.copy()
    for r in refs:
        overlay[r.mask] = (160, 160, 160)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
    for r in refs:
        cnts, _ = cv2.findContours(r.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (40, 40, 40), 2, cv2.LINE_AA)
        ys, xs = np.nonzero(r.mask)
        org = (int(np.median(xs)) - 22, max(25, int(ys.min()) - 10))
        cv2.putText(vis, "REF", org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(vis, "REF", org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--targets", type=int, nargs=2, metavar=("ID1", "ID2"))
    ap.add_argument("--pick", action="store_true", help="click the two players instead of giving --targets")
    ap.add_argument("--select-frame", type=int, default=0, help="clip frame for the ID preview / picker")
    ap.add_argument("--values", type=float, nargs=2, default=[0.25, 0.25], metavar=("V1", "V2"))
    ap.add_argument("--values-csv", help="CSV with columns frame,p1,p2")
    ap.add_argument("--court")
    ap.add_argument("--court-preset", default="left-half")
    ap.add_argument("--rules", choices=list(COURTS), default="fiba")
    ap.add_argument("--court-margin", type=float, default=1.0,
                    help="m outside the court lines still counted as on court")
    ap.add_argument("--prompt", default="basketball player")
    ap.add_argument("--det-thresh", type=float, default=0.4, help="min SAM 3 score to count as a player")
    ap.add_argument("--referee-prompt", default="referee",
                    help="SAM 3 prompt for referees (they occlude the rings but never get player IDs)")
    ap.add_argument("--referee-thresh", type=float, default=0.4, help="min SAM 3 score to count as a referee")
    ap.add_argument("--no-referees", action="store_true", help="don't segment referees")
    ap.add_argument("--occluder-thresh", type=float, default=0.3)
    ap.add_argument("--max-age", type=int, default=50,
                    help="frames a hidden player keeps their ID (default 50 = 2 s at 25 fps)")
    ap.add_argument("--gate", type=float, default=1.5, help="max movement (m) between frames for an ID match")
    ap.add_argument("--bf16-weights", action="store_true", help="half-precision SAM 3 weights (~1.7 GB)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--radius", type=float, default=0.9)
    ap.add_argument("--ring-width", type=float, default=0.28)
    ap.add_argument("--outline", type=float, default=0.045)
    ap.add_argument("--start-deg", type=float, default=90.0)
    ap.add_argument("--opacity", type=float, default=0.92)
    ap.add_argument("--shade", type=float, default=0.85)
    ap.add_argument("--smooth", type=float, default=0.55)
    ap.add_argument("--hold", type=int, default=8)
    ap.add_argument("--reacquire-m", type=float, default=1.5)
    ap.add_argument("--debug", action="store_true", help="draw all IDs and the court lines")
    args = ap.parse_args()

    court = COURTS[args.rules]
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = args.start
    n = total - start if total > 0 else 0
    if args.max_frames:
        n = min(n, args.max_frames) if n > 0 else args.max_frames
    if n <= 0:
        sys.exit("Empty clip: check --start / --max-frames.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, first = cap.read()
    if not ok:
        sys.exit("Cannot read the first frame of the clip.")
    h, w = first.shape[:2]

    # ---- 1. calibration (fixed camera) --------------------------------------
    H, _ = calibrate(first, args, court)
    H_inv = np.linalg.inv(H)

    # ---- 2. SAM 3, frame by frame (cached) -----------------------------------
    caches = {args.prompt: SegCache(args.input, args.prompt, first.shape, root=args.cache_dir,
                                    refresh=args.refresh_cache)}
    if not args.no_referees:
        caches[args.referee_prompt] = SegCache(args.input, args.referee_prompt, first.shape,
                                               root=args.cache_dir, refresh=args.refresh_cache)
    cache = caches[args.prompt]
    ref_cache = caches.get(args.referee_prompt) if not args.no_referees else None
    missing = {start + i: [p for p, c in caches.items() if not os.path.exists(c._path(start + i))]
               for i in range(n)}
    missing = {k: v for k, v in missing.items() if v}
    todo = sorted(missing)
    for p, c in caches.items():
        done = sum(os.path.exists(c._path(start + i)) for i in range(n))
        print(f"SAM 3 cache '{p}': {c.dir} ({done}/{n} clip frames already done)")
    if todo:
        segmenter = None
        progress = FrameProgress(len(todo))
        todo_set = set(todo)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            if start + i not in todo_set:
                continue
            if segmenter is None:
                progress.set_stage("Loading SAM 3")
                segmenter = Sam3PersonSegmenter(prompt=args.prompt, device=args.device,
                                                bf16_weights=args.bf16_weights)
            progress.set_stage("SAM 3 warm-up" if progress.n_sam == 0 else "SAM 3")
            t0 = time.perf_counter()
            results = segmenter.segment(frame, missing[start + i])
            for prompt, dets in results.items():
                caches[prompt].save(start + i, dets)
            progress.record_seg(sum(d.score >= args.det_thresh for d in results.get(args.prompt, [])),
                                time.perf_counter() - t0)
            progress.step()
        progress.close()
        del segmenter

    def load_people(idx):
        """(players, referees) for an absolute frame index, duplicates resolved."""
        players = cache.load(idx) or []
        refs = (ref_cache.load(idx) or []) if ref_cache else []
        return split_people(players, refs, args.referee_thresh)

    # ---- 3. identity tracking over the clip --------------------------------
    tracker = PlayerIDTracker(gate_m=args.gate, max_age=args.max_age)
    ids_per_frame: list[dict[int, int]] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in tqdm(range(n), desc="Tracking player IDs", unit="frame", dynamic_ncols=True):
        ok, frame = cap.read()
        if not ok:
            break
        dets, _ = load_people(start + i)
        ids_per_frame.append(tracker.step(player_observations(frame, dets, H_inv, court, args)))
    n = len(ids_per_frame)
    print(f"Assigned {tracker.next_id - 1} player IDs over {n} frames.")

    def tracked(idx_in_clip, dets):
        return [TrackedDet(tid, dets[di].mask, dets[di].score)
                for di, tid in sorted(ids_per_frame[idx_in_clip].items(), key=lambda kv: kv[1])]

    # ---- 4. choose targets ---------------------------------------------------
    sel = min(max(args.select_frame, 0), n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start + sel)
    _, sel_frame = cap.read()
    sel_dets, sel_refs = load_people(start + sel)
    sel_players = tracked(sel, sel_dets)
    if args.pick:
        targets = pick_targets(sel_frame, sel_players)
        print(f"Targets: {targets[0]} {targets[1]}  (next time: --targets {targets[0]} {targets[1]})")
    elif args.targets:
        targets = list(args.targets)
    else:
        prev = os.path.splitext(args.output)[0] + "_ids.jpg"
        vis = draw_referees(id_preview(sel_frame, sel_players), sel_refs)
        draw_court(vis, H, court)
        cv2.imwrite(prev, vis)
        print(f"Players on clip frame {sel}: IDs {[p.obj_id for p in sel_players]}"
              + (f", plus {len(sel_refs)} referee(s)" if ref_cache else ""))
        print(f"Preview written to {prev}. Re-run with --targets ID1 ID2 (or --pick).")
        return

    # ---- 5. render -----------------------------------------------------------
    style = DonutStyle(outer_r=args.radius, ring_w=args.ring_width, outline=args.outline,
                       start_deg=args.start_deg, opacity=args.opacity, shade=args.shade)
    table = load_values_csv(args.values_csv) if args.values_csv else None
    followers = [TargetFollower(t, args.reacquire_m, newborn_frames=int(fps)) for t in targets]
    first_seen: dict[int, int] = {}
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in tqdm(range(n), desc="Rendering", unit="frame", dynamic_ncols=True):
        ok, frame = cap.read()
        if not ok:
            break
        idx = start + i
        dets, refs = load_people(idx)
        players = tracked(i, dets)
        for p in players:
            first_seen.setdefault(p.obj_id, idx)
        by_id = {p.obj_id: p for p in players}
        occ = build_occluder(dets + refs, frame.shape, args.occluder_thresh, feather=0.8)
        vals = values_at(idx, table, tuple(args.values))
        taken = {f.obj_id for f in followers}
        for f in followers:
            d = f.find(players, by_id, first_seen, idx, H_inv, taken)
            pos = None
            if d is not None:
                gp = ground_point(d.mask)
                if gp is not None:
                    pos = to_court(H_inv, gp)
                    f.last_seen_pos = pos
            f.track.update(pos, args.smooth, args.hold)
        out = frame.copy()
        order = sorted(range(2), key=lambda k: -1e9 if followers[k].track.pos is None
                       else followers[k].track.pos[1])
        for k in order:
            if followers[k].track.pos is not None:
                render_donut(out, occ, H, followers[k].track.pos, vals[k], style)
        if args.debug:
            out = draw_referees(id_preview(out, players, highlight={f.obj_id for f in followers}), refs)
            draw_court(out, H, court)
        writer.write(out)
    cap.release()
    writer.release()
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()