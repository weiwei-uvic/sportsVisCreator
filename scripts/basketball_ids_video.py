#!/usr/bin/env python3
"""
basketball_ids_video.py
=======================
Make a video that shows the tracked ID of every person, from the SAM 3
segmentation cache written by basketball_donuts_frames.py. SAM 3 is NOT run
(and not needed): the script only reads the cache.

Identity tracking is the same code, with the same defaults, as
basketball_donuts_frames.py, so the IDs shown here are the IDs to pass to
its --targets. Use the same --start, --max-frames, court calibration and
tracking options as in that script, otherwise the numbering differs.

What is drawn
-------------
* every tracked player: coloured mask, outline and ID label
* referees: grey mask labelled REF (they never get IDs)
* optional: a short trail of each player's floor positions (--trail), which
  makes ID swaps easy to spot, the court lines (--court-lines), and the
  frame number (--frame-number)

Standalone: needs only OpenCV, NumPy and tqdm (SciPy optional), plus the cache.

Usage
-----
    python basketball_ids_video.py game.mp4 game_ids.mp4 --start 1500 --max-frames 300 \
        --court-preset left-half --court "x,y;x,y;x,y;x,y" --trail 25 --court-lines

    # also write every ID's court position per frame:
    python basketball_ids_video.py ... --csv game_ids.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from tqdm import tqdm


# ===========================================================================
# Cache (read only)
# ===========================================================================


@dataclass
class Detection:
    mask: np.ndarray  # bool HxW
    score: float


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


class ReadOnlyCache:
    """
    Reads the per-frame SAM 3 cache written by basketball_donuts_frames.py.
    Never writes or deletes anything. Refuses a cache made from a different
    video file.
    """

    def __init__(self, video_path: str, prompt: str, frame_shape, root: str | None = None):
        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-") or "prompt"
        root = root or os.path.splitext(video_path)[0] + ".sam3cache"
        self.dir = os.path.join(root, slug)
        self.shape = tuple(frame_shape[:2])
        meta_path = os.path.join(self.dir, "meta.json")
        if not os.path.exists(meta_path):
            sys.exit(f"No SAM 3 cache for prompt '{prompt}' at {self.dir}.\n"
                     "Run basketball_donuts_frames.py on this video first.")
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("fingerprint") != video_fingerprint(video_path) or \
                (meta.get("height"), meta.get("width")) != self.shape:
            sys.exit(f"The cache in {self.dir} was made from a different video file.")

    def _path(self, idx: int) -> str:
        return os.path.join(self.dir, f"frame_{idx:06d}.npz")

    def has(self, idx: int) -> bool:
        return os.path.exists(self._path(idx))

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


# ===========================================================================
# Player identity tracking (same as basketball_donuts_frames.py)
# ===========================================================================


try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # greedy fallback
    linear_sum_assignment = None


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


# ===========================================================================
# Drawing
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


def draw_trails(vis: np.ndarray, trails: dict, H: np.ndarray) -> None:
    """Recent floor positions of each ID, projected back into the image."""
    for tid, pts in trails.items():
        if len(pts) < 2:
            continue
        p = to_image(H, np.float32(list(pts)))
        col = id_color(tid)
        for k in range(1, len(p)):
            t = k / (len(p) - 1)                       # older segments fainter/thinner
            c = tuple(int(v * (0.35 + 0.65 * t)) for v in col)
            cv2.line(vis, tuple(np.int32(p[k - 1])), tuple(np.int32(p[k])), c,
                     1 + int(2 * t), cv2.LINE_AA)
        cv2.circle(vis, tuple(np.int32(p[-1])), 4, col, -1, cv2.LINE_AA)


def draw_frame_number(vis: np.ndarray, idx: int, n_players: int, n_refs: int) -> None:
    txt = f"frame {idx}   players {n_players}   refs {n_refs}"
    cv2.putText(vis, txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


# ===========================================================================
# Main
# ===========================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="the same video given to basketball_donuts_frames.py")
    ap.add_argument("output", help="output video with IDs")
    # must match basketball_donuts_frames.py for identical IDs
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--court")
    ap.add_argument("--court-preset", default="left-half")
    ap.add_argument("--rules", choices=list(COURTS), default="fiba")
    ap.add_argument("--court-margin", type=float, default=1.0)
    ap.add_argument("--prompt", default="basketball player")
    ap.add_argument("--det-thresh", type=float, default=0.4)
    ap.add_argument("--referee-prompt", default="referee")
    ap.add_argument("--referee-thresh", type=float, default=0.4)
    ap.add_argument("--no-referees", action="store_true")
    ap.add_argument("--max-age", type=int, default=50)
    ap.add_argument("--gate", type=float, default=1.5)
    ap.add_argument("--cache-dir")
    # display
    ap.add_argument("--trail", type=int, default=25, help="trail length in frames (0 = off)")
    ap.add_argument("--court-lines", action="store_true", help="draw the calibrated court lines")
    ap.add_argument("--frame-number", action="store_true", help="print frame number and counts")
    ap.add_argument("--highlight", type=int, nargs="*", default=[], metavar="ID",
                    help="IDs drawn with a thick outline (e.g. your --targets)")
    ap.add_argument("--csv", help="also write frame,id,x_m,y_m for every tracked player")
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

    H, _ = calibrate(first, args, court)
    H_inv = np.linalg.inv(H)

    cache = ReadOnlyCache(args.input, args.prompt, first.shape, args.cache_dir)
    ref_cache = None
    if not args.no_referees:
        ref_dir = os.path.join(args.cache_dir or os.path.splitext(args.input)[0] + ".sam3cache",
                               re.sub(r"[^a-z0-9]+", "-", args.referee_prompt.lower()).strip("-"))
        if os.path.exists(os.path.join(ref_dir, "meta.json")):
            ref_cache = ReadOnlyCache(args.input, args.referee_prompt, first.shape, args.cache_dir)
        else:
            print("No referee cache found: showing players only.")
    missing = [start + i for i in range(n) if not cache.has(start + i)]
    if missing:
        sys.exit(f"{len(missing)} of {n} clip frames are not in the cache (first missing: frame "
                 f"{missing[0]}). Run basketball_donuts_frames.py with the same --start/--max-frames first.")

    tracker = PlayerIDTracker(gate_m=args.gate, max_age=args.max_age)
    trails = defaultdict(lambda: deque(maxlen=max(args.trail, 1)))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    csv_f = open(args.csv, "w", newline="") if args.csv else None
    csv_w = csv.writer(csv_f) if csv_f else None
    if csv_w:
        csv_w.writerow(["frame", "id", "x_m", "y_m"])
    seen_ids = set()

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in tqdm(range(n), desc="Writing ID video", unit="frame", dynamic_ncols=True):
        ok, frame = cap.read()
        if not ok:
            break
        idx = start + i
        dets = cache.load(idx) or []
        refs = (ref_cache.load(idx) or []) if (ref_cache and ref_cache.has(idx)) else []
        dets, refs = split_people(dets, refs, args.referee_thresh)
        obs = player_observations(frame, dets, H_inv, court, args)
        ids = tracker.step(obs)
        pos_by_det = {o.det_idx: o.pos for o in obs}

        players = []
        for di, tid in sorted(ids.items(), key=lambda kv: kv[1]):
            players.append(TrackedDet(tid, dets[di].mask, dets[di].score))
            seen_ids.add(tid)
            pos = pos_by_det[di]
            if args.trail > 0:
                trails[tid].append((float(pos[0]), float(pos[1])))
            if csv_w:
                csv_w.writerow([idx, tid, f"{pos[0]:.3f}", f"{pos[1]:.3f}"])
        visible = {p.obj_id for p in players}
        for tid in list(trails):                  # drop trails of IDs no longer tracked
            if tid not in visible and all(t.tid != tid for t in tracker.tracks):
                del trails[tid]

        vis = id_preview(frame, players, highlight=set(args.highlight))
        vis = draw_referees(vis, refs)
        if args.court_lines:
            draw_court(vis, H, court)
        if args.trail > 0:
            draw_trails(vis, {t: p for t, p in trails.items() if t in visible}, H)
        if args.frame_number:
            draw_frame_number(vis, idx, len(players), len(refs))
        writer.write(vis)

    cap.release()
    writer.release()
    if csv_f:
        csv_f.close()
        print(f"Positions written to {args.csv}")
    print(f"{len(seen_ids)} IDs tracked: {sorted(seen_ids)}")
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
