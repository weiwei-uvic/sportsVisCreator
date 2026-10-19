#!/usr/bin/env python3
"""
basketball_donuts.py
====================
Segment and track every basketball player with SAM 3 (video mode), then paint
a perspective-correct donut gauge on the floor under TWO chosen players. The
rings are occluded by every player (and referee) standing on them, and follow
the broadcast camera as it pans and zooms.

Needs tennis_donuts.py in the same folder (it reuses its ring renderer).

Pipeline
--------
1. SAM 3 video predictor, text prompt "basketball player": detects all players
   on the first frame, keeps detecting newcomers, and tracks each one with a
   stable object ID through the clip. Results are cached per clip.
2. You pick the two target players by their SAM 3 IDs (a labelled preview
   image is written for this, or click them with --pick).
3. Court calibration: 4 court landmarks clicked or given on the first frame.
   After that, the camera is followed automatically. Floor/background
   features (players masked out) are tracked frame to frame and the
   court-to-image mapping is updated. If tracking is lost (camera cut or
   replay), drawing pauses and resumes when the original view comes back.
4. For each target, the floor point under the body centre is projected onto
   the court, smoothed, and the ring is drawn there, occluded by all masks.

Usage
-----
    # 1st run: tracks everyone with SAM 3, writes match_out_ids.jpg, then stops
    python basketball_donuts.py match.mp4 match_out.mp4 --court-preset left-half

    # look at match_out_ids.jpg, then pick the two players by ID (SAM 3 is not
    # run again, results come from the cache):
    python basketball_donuts.py match.mp4 match_out.mp4 --court-preset left-half \
        --court "412,288;1260,301;1700,860;95,840" --targets 3 7 --values 0.25 0.6

    # or click the two players instead of typing IDs:
    python basketball_donuts.py match.mp4 match_out.mp4 --pick ...

Court calibration
-----------------
Coordinates are in metres. Origin is the far-left corner of the court as seen
on screen, x runs along the court to the right and y runs towards the camera.
Click or list 4 points in this order (same as tennis): FAR-LEFT, FAR-RIGHT,
NEAR-RIGHT, NEAR-LEFT, for the rectangle chosen with --court-preset:

    full        whole court (4 corners)
    left-half   left baseline corners + both ends of the half-court line
    right-half  both ends of the half-court line + right baseline corners
    left-paint  4 corners of the left painted area (key)
    right-paint 4 corners of the right painted area (key)

A bigger rectangle gives a more accurate result. Use the half that is fully in
view on the first frame. --rules fiba (default, 28 x 15 m) or nba (28.65 x 15.24 m).

Advanced: give any 4+ landmarks yourself as "u,v=X,Y" pairs (pixel = metres):
    --court "412,288=0,0;1260,301=14,0;1700,860=14,15;95,840=0,15"

Clips
-----
SAM 3 video mode loads the whole clip into memory and tracks it in one pass.
Work on one possession or camera shot at a time with --start/--max-frames
(e.g. 10-30 s). Each clip gets its own cache.

Values
------
--values A B          fixed values for target 1 and target 2 (0..1)
--values-csv file     columns frame,p1,p2 (interpolated between rows)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tennis_donuts import (  # noqa: E402  (shared ring renderer & helpers)
    DonutStyle,
    PlayerTrack,
    build_occluder,
    click_court_corners,
    ground_point,
    render_donut,
    to_court,
    to_image,
    video_fingerprint,
)

# ===========================================================================
# Court
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
# Camera tracking (broadcast camera pans / zooms)
# ===========================================================================
class CameraTracker:
    """
    Keeps the court->image homography up to date while the camera moves.

    Corners on the floor and background (with players masked out) are tracked
    with optical flow from a keyframe. The keyframe-to-current homography
    (RANSAC) is chained onto the keyframe's court->image mapping. When too few
    points survive, the current frame becomes the new keyframe. When tracking
    fails (cut / replay), it tries every frame to re-match ORB features
    against the calibration frame and resumes when they match.
    """

    def __init__(self, H0, gray0, exclude0, static_boxes=(), min_inliers=40,
                 refresh_below=150, max_corners=800):
        self.static_boxes = static_boxes
        self.min_inliers = min_inliers
        self.refresh_below = refresh_below
        self.max_corners = max_corners
        self.orb = cv2.ORB_create(3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.H_ref = H0.copy()
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(gray0, self._feat_mask(exclude0))
        self.lost = False
        self._new_keyframe(H0, gray0, exclude0)

    def _feat_mask(self, exclude: np.ndarray) -> np.ndarray:
        m = np.where(exclude, 0, 255).astype(np.uint8)
        for x, y, w, h in self.static_boxes:          # score bugs / logos
            m[y:y + h, x:x + w] = 0
        return m

    def _new_keyframe(self, H, gray, exclude):
        pts = cv2.goodFeaturesToTrack(gray, self.max_corners, 0.01, 8, mask=self._feat_mask(exclude))
        pts = np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2)
        self.key_H, self.key_pts, self.cur_pts, self.prev_gray = H, pts, pts.copy(), gray

    def update(self, gray: np.ndarray, exclude: np.ndarray) -> np.ndarray | None:
        if self.lost or len(self.cur_pts) < self.min_inliers:
            return self._relocalize(gray, exclude)
        lk = dict(winSize=(21, 21), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.cur_pts.reshape(-1, 1, 2), None, **lk)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **lk)
        p1, p0b = p1.reshape(-1, 2), p0b.reshape(-1, 2)
        h, w = gray.shape
        ok = (st1.ravel() == 1) & (st2.ravel() == 1)
        ok &= np.linalg.norm(p0b - self.cur_pts, axis=1) < 1.0          # forward-backward check
        ok &= (p1[:, 0] >= 0) & (p1[:, 0] < w - 1) & (p1[:, 1] >= 0) & (p1[:, 1] < h - 1)
        xi, yi = np.clip(p1[:, 0].astype(int), 0, w - 1), np.clip(p1[:, 1].astype(int), 0, h - 1)
        ok &= ~exclude[yi, xi]                                           # point now under a player
        key, cur = self.key_pts[ok], p1[ok]
        if len(key) < self.min_inliers:
            self.lost = True
            return self._relocalize(gray, exclude)
        G, inl = cv2.findHomography(key, cur, cv2.RANSAC, 2.0)
        if G is None or inl.sum() < self.min_inliers:
            self.lost = True
            return self._relocalize(gray, exclude)
        inl = inl.ravel().astype(bool)
        H = G @ self.key_H
        H /= H[2, 2]
        self.key_pts, self.cur_pts, self.prev_gray = key[inl], cur[inl], gray
        if inl.sum() < self.refresh_below:
            self._new_keyframe(H, gray, exclude)
        return H

    def _relocalize(self, gray, exclude) -> np.ndarray | None:
        self.lost = True
        if self.ref_des is None:
            return None
        kp, des = self.orb.detectAndCompute(gray, self._feat_mask(exclude))
        if des is None or len(kp) < self.min_inliers:
            return None
        good = [m for m, n in (p for p in self.bf.knnMatch(self.ref_des, des, k=2) if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) < self.min_inliers:
            return None
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        G, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if G is None or inl.sum() < self.min_inliers:
            return None
        H = G @ self.H_ref
        H /= H[2, 2]
        self.lost = False
        self._new_keyframe(H, gray, exclude)
        return H


# ===========================================================================
# SAM 3 video tracking + cache
# ===========================================================================
@dataclass
class TrackedDet:
    obj_id: int
    mask: np.ndarray   # bool HxW
    score: float


class TrackCache:
    """
    Per-clip cache of SAM 3 video results: object IDs, scores and masks for
    every frame (masks cropped to their box and bit-packed). The clip is
    reused only if it was tracked to the end (tracking can't resume midway).
    """

    VERSION = 1

    def __init__(self, video_path, prompt, start, n_frames, frame_shape, root=None, refresh=False):
        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-") or "prompt"
        root = root or os.path.splitext(video_path)[0] + ".sam3track"
        self.dir = os.path.join(root, f"{slug}_f{start}-{start + n_frames - 1}")
        self.shape = tuple(frame_shape[:2])
        self.meta = {"version": self.VERSION, "video": os.path.basename(video_path),
                     "fingerprint": video_fingerprint(video_path), "prompt": prompt,
                     "start": start, "n_frames": n_frames,
                     "height": self.shape[0], "width": self.shape[1], "complete": False}
        meta_path = os.path.join(self.dir, "meta.json")
        old = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    old = json.load(f)
            except (OSError, ValueError):
                old = None
        keys = ("version", "fingerprint", "start", "n_frames", "height", "width")
        if old and not refresh and all(old.get(k) == self.meta[k] for k in keys):
            self.meta["complete"] = bool(old.get("complete"))
        elif os.path.isdir(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir, exist_ok=True)
        self._write_meta()

    @property
    def complete(self) -> bool:
        return self.meta["complete"]

    def _write_meta(self):
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    def mark_complete(self):
        self.meta["complete"] = True
        self._write_meta()

    def _path(self, idx):
        return os.path.join(self.dir, f"frame_{idx:06d}.npz")

    def save(self, idx: int, ids, scores, masks) -> None:
        keep_ids, keep_sc, boxes, chunks, offs = [], [], [], [], [0]
        for oid, sc, m in zip(ids, scores, masks):
            m = np.asarray(m).astype(bool)
            if m.ndim == 3:
                m = m[0]
            if m.shape != self.shape:
                m = cv2.resize(m.astype(np.uint8), self.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            ys, xs = np.nonzero(m)
            if len(ys) == 0:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            packed = np.packbits(m[y0:y1, x0:x1].ravel())
            keep_ids.append(int(oid))
            keep_sc.append(float(sc))
            boxes.append((y0, x0, y1, x1))
            chunks.append(packed)
            offs.append(offs[-1] + len(packed))
        tmp = self._path(idx) + ".tmp.npz"
        np.savez_compressed(tmp, ids=np.int64(keep_ids), scores=np.float32(keep_sc),
                            boxes=np.int32(boxes).reshape(-1, 4),
                            bits=np.concatenate(chunks) if chunks else np.zeros(0, np.uint8),
                            offsets=np.int64(offs))
        os.replace(tmp, self._path(idx))

    def load(self, idx: int) -> list[TrackedDet]:
        p = self._path(idx)
        if not os.path.exists(p):
            return []
        z = np.load(p)
        out = []
        for i in range(len(z["ids"])):
            y0, x0, y1, x1 = (int(v) for v in z["boxes"][i])
            n = (y1 - y0) * (x1 - x0)
            crop = np.unpackbits(z["bits"][z["offsets"][i]:z["offsets"][i + 1]], count=n)
            mask = np.zeros(self.shape, bool)
            mask[y0:y1, x0:x1] = crop.reshape(y1 - y0, x1 - x0).astype(bool)
            out.append(TrackedDet(int(z["ids"][i]), mask, float(z["scores"][i])))
        return out


class Spinner:
    """Live elapsed-time line for a blocking step."""

    def __init__(self, text):
        self.text, self._stop = text, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        i, frames = 0, "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        while not self._stop.wait(0.1):
            sys.stderr.write(f"\r{frames[i % 10]} {self.text} ({time.perf_counter() - self.t0:5.1f}s)")
            sys.stderr.flush()
            i += 1

    def __enter__(self):
        self.t0 = time.perf_counter()
        self._t.start()
        return self

    def __exit__(self, exc, *_):
        self._stop.set()
        self._t.join()
        sys.stderr.write(f"\r{'✔' if exc is None else '✘'} {self.text} "
                         f"({time.perf_counter() - self.t0:5.1f}s)\n")


def extract_clip(video_path, start, n, out_dir) -> None:
    """Write frames [start, start+n) as 00000.jpg ... (SAM 3 reads JPEG folders)."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in tqdm(range(n), desc="Extracting clip frames", unit="frame", dynamic_ncols=True):
        ok, fr = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()


def run_sam3_tracking(args, cache: TrackCache, start: int, n: int, whole_video: bool) -> None:
    from sam3.model_builder import build_sam3_video_predictor

    clip_dir = None
    if whole_video:
        resource = args.input
    else:
        clip_dir = os.path.join(cache.dir, "_frames")
        extract_clip(args.input, start, n, clip_dir)
        resource = clip_dir

    with Spinner("Loading SAM 3 video model (first run downloads the checkpoint)"):
        predictor = build_sam3_video_predictor()
    try:
        with Spinner(f"Loading {n} frames into SAM 3"):
            sid = predictor.handle_request(dict(
                type="start_session", resource_path=resource,
                offload_video_to_cpu=not args.no_offload))["session_id"]
        with Spinner(f"Detecting '{args.prompt}' on clip frame {args.prompt_frame}"):
            resp = predictor.handle_request(dict(
                type="add_prompt", session_id=sid, frame_index=args.prompt_frame,
                text=args.prompt, output_prob_thresh=args.track_thresh))
        n0 = len(resp["outputs"]["out_obj_ids"])
        print(f"SAM 3 found {n0} '{args.prompt}' instances on the prompt frame.")

        seen = set()
        bar = tqdm(total=n, desc="SAM 3 tracking", unit="frame", dynamic_ncols=True)
        direction = "forward" if args.prompt_frame == 0 else "both"
        for r in predictor.handle_stream_request(dict(
                type="propagate_in_video", session_id=sid, propagation_direction=direction,
                start_frame_index=args.prompt_frame, output_prob_thresh=args.track_thresh)):
            fi, o = r["frame_index"], r["outputs"]
            cache.save(start + fi, o["out_obj_ids"], o["out_probs"], o["out_binary_masks"])
            if fi not in seen:
                seen.add(fi)
                bar.update(1)
                bar.set_postfix_str(f"objects {len(o['out_obj_ids'])}")
        bar.close()
        predictor.handle_request(dict(type="close_session", session_id=sid))
    finally:
        if hasattr(predictor, "shutdown"):
            predictor.shutdown()
        if clip_dir:
            shutil.rmtree(clip_dir, ignore_errors=True)
    cache.mark_complete()


# ===========================================================================
# Target selection
# ===========================================================================
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


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    # what to draw
    ap.add_argument("--targets", type=int, nargs=2, metavar=("ID1", "ID2"),
                    help="SAM 3 object IDs of the two players (see the *_ids.jpg preview)")
    ap.add_argument("--pick", action="store_true", help="click the two players instead of giving --targets")
    ap.add_argument("--select-frame", type=int, default=0,
                    help="clip frame shown in the ID preview / picker (default: first)")
    ap.add_argument("--values", type=float, nargs=2, default=[0.25, 0.25], metavar=("V1", "V2"))
    ap.add_argument("--values-csv", help="CSV with columns frame,p1,p2")
    # court
    ap.add_argument("--court", help="4 pixel points 'x,y;x,y;x,y;x,y' for --court-preset, "
                    "or 4+ 'u,v=X,Y' pairs (pixels=metres)")
    ap.add_argument("--court-preset", default="left-half",
                    help="full | left-half | right-half | left-paint | right-paint")
    ap.add_argument("--rules", choices=list(COURTS), default="fiba")
    ap.add_argument("--ignore-box", action="append", default=[], metavar="x,y,w,h",
                    help="screen region to ignore for camera tracking (score bug, logo); repeatable")
    # SAM 3
    ap.add_argument("--prompt", default="basketball player")
    ap.add_argument("--prompt-frame", type=int, default=0, help="clip frame where the text prompt is applied")
    ap.add_argument("--track-thresh", type=float, default=0.5, help="SAM 3 output probability threshold")
    ap.add_argument("--no-offload", action="store_true",
                    help="keep video frames on the GPU (faster, needs much more GPU memory)")
    ap.add_argument("--cache-dir")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--start", type=int, default=0, help="first frame of the clip")
    ap.add_argument("--max-frames", type=int, default=0, help="clip length (0 = to the end)")
    # ring look & behaviour
    ap.add_argument("--radius", type=float, default=0.9, help="ring outer radius (m)")
    ap.add_argument("--ring-width", type=float, default=0.28)
    ap.add_argument("--outline", type=float, default=0.045)
    ap.add_argument("--start-deg", type=float, default=90.0, help="red arc start, clockwise from 12 o'clock")
    ap.add_argument("--opacity", type=float, default=0.92)
    ap.add_argument("--shade", type=float, default=0.85)
    ap.add_argument("--occluder-thresh", type=float, default=0.3)
    ap.add_argument("--smooth", type=float, default=0.55)
    ap.add_argument("--hold", type=int, default=8, help="frames to keep a ring when its player is lost")
    ap.add_argument("--reacquire-m", type=float, default=1.5,
                    help="max distance (m) for re-identifying a lost target under a new ID")
    ap.add_argument("--debug", action="store_true", help="draw all IDs and the tracked court lines")
    args = ap.parse_args()

    court = COURTS[args.rules]
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = args.start
    n = (total - start) if total > 0 else 0
    if args.max_frames:
        n = min(n, args.max_frames) if n > 0 else args.max_frames
    if n <= 0:
        sys.exit("Empty clip: check --start / --max-frames.")
    whole_video = start == 0 and n == total

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, first = cap.read()
    if not ok:
        sys.exit("Cannot read the first frame of the clip.")
    h, w = first.shape[:2]

    # ---- 1. SAM 3 tracking (cached) -------------------------------------------
    cache = TrackCache(args.input, args.prompt, start, n, first.shape, args.cache_dir, args.refresh_cache)
    print(f"SAM 3 cache: {cache.dir}")
    if cache.complete:
        print("Using cached SAM 3 tracking (SAM 3 is not run).")
    else:
        run_sam3_tracking(args, cache, start, n, whole_video)

    # ---- 2. Target selection ---------------------------------------------------
    sel = start + args.select_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, sel)
    ok, sel_frame = cap.read()
    sel_dets = cache.load(sel)
    if args.pick:
        targets = pick_targets(sel_frame, sel_dets)
        print(f"Targets: {targets[0]} {targets[1]}  (next time: --targets {targets[0]} {targets[1]})")
    elif args.targets:
        targets = list(args.targets)
    else:
        prev = os.path.splitext(args.output)[0] + "_ids.jpg"
        cv2.imwrite(prev, id_preview(sel_frame, sel_dets))
        ids = sorted(d.obj_id for d in sel_dets)
        print(f"\nTracked {len(ids)} people on clip frame {args.select_frame}: IDs {ids}")
        print(f"Preview with IDs written to {prev}")
        print("Re-run with --targets ID1 ID2 (or --pick). SAM 3 won't run again.")
        return

    # ---- 3. Court calibration on the first clip frame --------------------------
    H0, img_pts = calibrate(first, args, court)
    first_dets = cache.load(start)
    exclude0 = np.zeros((h, w), np.uint8)
    for d in first_dets:
        exclude0 |= d.mask.astype(np.uint8)
    exclude0 = cv2.dilate(exclude0, np.ones((15, 15), np.uint8))
    boxes = [tuple(int(v) for v in b.split(",")) for b in args.ignore_box]
    camera = CameraTracker(H0, cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), exclude0.astype(bool), boxes)

    # ---- 4. Render -------------------------------------------------------------
    style = DonutStyle(outer_r=args.radius, ring_w=args.ring_width, outline=args.outline,
                       start_deg=args.start_deg, opacity=args.opacity, shade=args.shade)
    table = load_values_csv(args.values_csv) if args.values_csv else None
    followers = [TargetFollower(t, args.reacquire_m, newborn_frames=int(fps)) for t in targets]
    first_seen: dict[int, int] = {}

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    bar = tqdm(total=n, desc="Rendering", unit="frame", dynamic_ncols=True)
    n_lost = 0
    for i in range(n):
        idx = start + i
        ok, frame = cap.read()
        if not ok:
            break
        dets = cache.load(idx)
        for d in dets:
            first_seen.setdefault(d.obj_id, idx)
        by_id = {d.obj_id: d for d in dets}
        exclude = np.zeros((h, w), np.uint8)
        for d in dets:
            exclude |= d.mask.astype(np.uint8)
        exclude = cv2.dilate(exclude, np.ones((15, 15), np.uint8)).astype(bool)

        H = H0 if i == 0 else camera.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), exclude)
        out = frame.copy()
        if H is None:
            n_lost += 1
            for f in followers:
                f.track.pos = None
        else:
            H_inv = np.linalg.inv(H)
            occ = build_occluder(dets, frame.shape, args.occluder_thresh, feather=0.8)
            vals = values_at(idx, table, tuple(args.values))
            taken = {f.obj_id for f in followers}
            for f in followers:
                d = f.find(dets, by_id, first_seen, idx, H_inv, taken)
                pos = None
                if d is not None:
                    gp = ground_point(d.mask)
                    if gp is not None:
                        pos = to_court(H_inv, gp)
                        f.last_seen_pos = pos
                f.track.update(pos, args.smooth, args.hold)
            # far rings first
            order = sorted(range(2), key=lambda k: -1e9 if followers[k].track.pos is None
                           else followers[k].track.pos[1])
            for k in order:
                if followers[k].track.pos is not None:
                    render_donut(out, occ, H, followers[k].track.pos, vals[k], style)
            if args.debug:
                out = id_preview(out, dets, highlight={f.obj_id for f in followers})
                draw_court(out, H, court)
        if args.debug and H is None:
            cv2.putText(out, "camera tracking lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2, cv2.LINE_AA)
        writer.write(out)
        bar.update(1)
        bar.set_postfix_str(f"people {len(dets)}, targets "
                            + "/".join(str(f.obj_id) for f in followers)
                            + (", camera LOST" if H is None else ""))
    bar.close()
    cap.release()
    writer.release()
    print(f"Done -> {args.output}")
    if n_lost:
        print(f"Camera tracking was lost on {n_lost} frames (cut/replay?); no rings were drawn there.")


if __name__ == "__main__":
    main()