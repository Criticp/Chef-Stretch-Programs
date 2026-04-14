"""
Shared bringup helpers — sweep and visual-tracking loops used by both
test_03b_track.py (CLI) and test_04_gui.py (tkinter GUI).

Design:
- Every loop accepts a `stop_event: threading.Event` so callers can cancel.
- Every loop accepts an optional `on_pose(color, detections, pan, tilt)`
  callback so the caller can log, render a preview, or push to a queue.
- No direct stdio or UI code here — this module is headless.

The tracker is a simple proportional controller driving head pan/tilt to
keep the highest-confidence target detection centered in the frame. Pixel
error is converted to radian error using the RealSense intrinsics
(focal length in pixels), so the gain is unit-clean: err_rad ≈ err_px / fx.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from fruit_finder.camera import CameraManager
from fruit_finder.detector import Detection, FoodDetector

logger = logging.getLogger(__name__)


# ----- type aliases --------------------------------------------------------

# Callback: (color_bgr, detections, real_pan_rad, real_tilt_rad)
PoseCallback = Optional[Callable[[np.ndarray, list, float, float], None]]


# ----- data ---------------------------------------------------------------


@dataclass
class TrackerParams:
    """Runtime tunables. Loaded from config.yaml `tracking:` section with defaults."""
    kp: float = 0.4
    target_conf_min: float = 0.40
    sweep_acquire_conf_min: float = 0.50
    lost_frames_timeout: int = 20
    max_step_rad: float = 0.25
    pan_sign: float = -1.0
    tilt_sign: float = -1.0

    @classmethod
    def from_config(cls, config: dict) -> "TrackerParams":
        trk = config.get("tracking", {}) or {}
        defaults = cls()
        return cls(
            kp=float(trk.get("kp", defaults.kp)),
            target_conf_min=float(trk.get("target_conf_min", defaults.target_conf_min)),
            sweep_acquire_conf_min=float(
                trk.get("sweep_acquire_conf_min", defaults.sweep_acquire_conf_min)
            ),
            lost_frames_timeout=int(
                trk.get("lost_frames_timeout", defaults.lost_frames_timeout)
            ),
            max_step_rad=float(trk.get("max_step_rad", defaults.max_step_rad)),
            pan_sign=float(trk.get("pan_sign", defaults.pan_sign)),
            tilt_sign=float(trk.get("tilt_sign", defaults.tilt_sign)),
        )


@dataclass
class HeadLimits:
    pan_lo: float
    pan_hi: float
    tilt_lo: float
    tilt_hi: float


# ----- helpers -------------------------------------------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def read_head_pose(robot) -> tuple[float, float]:
    """Best-effort read of (head_pan, head_tilt) in radians."""
    try:
        pan = float(robot.head.get_joint("head_pan").status["pos"])
    except Exception:
        pan = 0.0
    try:
        tilt = float(robot.head.get_joint("head_tilt").status["pos"])
    except Exception:
        tilt = 0.0
    return pan, tilt


def get_head_limits(robot, margin: float = 0.05) -> HeadLimits:
    """Read hard soft-motion limits for head_pan/head_tilt; fall back to safe defaults."""
    pan_lo, pan_hi = -1.5, 1.5
    tilt_lo, tilt_hi = -1.5, 0.3
    try:
        hard = robot.head.get_joint("head_pan").soft_motion_limits["hard"]
        pan_lo, pan_hi = float(hard[0]) + margin, float(hard[1]) - margin
    except Exception:
        pass
    try:
        hard = robot.head.get_joint("head_tilt").soft_motion_limits["hard"]
        tilt_lo, tilt_hi = float(hard[0]) + margin, float(hard[1]) - margin
    except Exception:
        pass
    return HeadLimits(pan_lo, pan_hi, tilt_lo, tilt_hi)


def drain_frames(cam: CameraManager, n: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Grab n frames and return the last good (color, depth) pair."""
    last_color: Optional[np.ndarray] = None
    last_depth: Optional[np.ndarray] = None
    for _ in range(n):
        color, depth = cam.get_frames()
        if color is not None and depth is not None:
            last_color, last_depth = color, depth
    return last_color, last_depth


def rad_per_pixel(cam: CameraManager) -> tuple[float, float]:
    """Radians per pixel (x, y) from the RealSense intrinsics; fallback to D435i defaults."""
    intr = cam.intrinsics
    if intr is None:
        # D435i color: roughly 69 x 42 degrees FOV at 640x480
        return (np.deg2rad(69.0) / 640.0, np.deg2rad(42.0) / 480.0)
    # For small angles, rad-per-pixel ≈ 1/fx and 1/fy.
    return (1.0 / float(intr.fx), 1.0 / float(intr.fy))


def pick_target(detections: list[Detection], conf_min: float) -> Optional[Detection]:
    """Return the highest-confidence is_target detection at or above conf_min."""
    hits = [d for d in detections if d.is_target and d.confidence >= conf_min]
    if not hits:
        return None
    return max(hits, key=lambda d: d.confidence)


def center_head(robot, wait: bool = True) -> None:
    """Best-effort: return head to (pan=0, tilt=0)."""
    try:
        robot.head.move_to("head_pan", 0.0)
        robot.head.move_to("head_tilt", 0.0)
        if wait:
            robot.wait_command()
    except Exception as exc:
        logger.warning("center_head failed: %s", exc)


# ----- loops ---------------------------------------------------------------


def sweep_until_target(
    robot,
    cam: CameraManager,
    detector: FoodDetector,
    config: dict,
    target_label: str,
    stop_event: threading.Event,
    on_pose: PoseCallback = None,
    warmup_frames: int = 6,
) -> Optional[tuple[Detection, float, float]]:
    """
    Pan the head through the configured search range, running the detector
    at each pose. Returns (detection, pan, tilt) as soon as a target-class
    detection fires above the acquire threshold, or None if stop_event is
    set or the full sweep completes with no target.
    """
    nav = config["navigation"]
    tp = TrackerParams.from_config(config)
    limits = get_head_limits(robot)

    pan_min = clamp(float(nav["search_pan_min_rad"]), limits.pan_lo, limits.pan_hi)
    pan_max = clamp(float(nav["search_pan_max_rad"]), limits.pan_lo, limits.pan_hi)
    pan_step = float(nav["search_pan_step_rad"])
    tilt_target = clamp(float(nav["search_tilt_rad"]), limits.tilt_lo, limits.tilt_hi)
    dwell = float(nav["search_pause_sec"])
    max_sweeps = int(nav.get("max_search_sweeps", 3))

    detector.set_target(target_label)
    logger.info(
        "sweep: target=%r pan=[%+.2f,%+.2f] step=%+.2f tilt=%+.2f dwell=%.2f max_sweeps=%d",
        target_label, pan_min, pan_max, pan_step, tilt_target, dwell, max_sweeps,
    )

    # Move to search tilt + center pan before the sweep.
    robot.head.move_to("head_pan", 0.0)
    robot.head.move_to("head_tilt", tilt_target)
    robot.wait_command()
    time.sleep(0.2)
    drain_frames(cam, warmup_frames)

    n_poses = max(2, int(round((pan_max - pan_min) / pan_step)) + 1)

    for sweep_n in range(max_sweeps):
        if stop_event.is_set():
            return None

        # Alternate direction so we don't jump from one end back to the other.
        if sweep_n % 2 == 0:
            angles = [pan_min + i * pan_step for i in range(n_poses)]
        else:
            angles = [pan_max - i * pan_step for i in range(n_poses)]

        for pan in angles:
            if stop_event.is_set():
                return None
            pan = clamp(pan, limits.pan_lo, limits.pan_hi)

            robot.head.move_to("head_pan", pan)
            robot.wait_command()
            time.sleep(dwell)

            color, _depth = drain_frames(cam, warmup_frames)
            if color is None:
                continue

            real_pan, real_tilt = read_head_pose(robot)
            detections = detector.detect(color)

            if on_pose is not None:
                try:
                    on_pose(color, detections, real_pan, real_tilt)
                except Exception as exc:
                    logger.warning("on_pose callback raised: %s", exc)

            target = pick_target(detections, tp.sweep_acquire_conf_min)
            if target is not None:
                logger.info(
                    "sweep: acquired %r at pan=%+.2f tilt=%+.2f conf=%.2f",
                    target.label, real_pan, real_tilt, target.confidence,
                )
                return (target, real_pan, real_tilt)

    logger.info("sweep: %d sweep(s) complete, no target above threshold", max_sweeps)
    return None


def track(
    robot,
    cam: CameraManager,
    detector: FoodDetector,
    config: dict,
    stop_event: threading.Event,
    on_pose: PoseCallback = None,
    params: Optional[TrackerParams] = None,
) -> str:
    """
    Closed-loop P controller: keep the highest-confidence target centered.

    Runs at detector rate. No wait_command() inside the loop — head commands
    chain continuously. Exits when stop_event is set or when the target is
    lost for more than `lost_frames_timeout` consecutive frames.

    Returns a short reason string: "stopped", "lost", or "error".
    """
    tp = params or TrackerParams.from_config(config)
    limits = get_head_limits(robot)
    rpp_x, rpp_y = rad_per_pixel(cam)
    logger.info(
        "track: kp=%.2f max_step=%.2f lost_timeout=%d pan_sign=%+d tilt_sign=%+d "
        "rad_per_px=(%.5f, %.5f)",
        tp.kp, tp.max_step_rad, tp.lost_frames_timeout, int(tp.pan_sign), int(tp.tilt_sign),
        rpp_x, rpp_y,
    )

    lost_frames = 0
    reason = "stopped"

    while not stop_event.is_set():
        color, _depth = cam.get_frames()
        if color is None:
            time.sleep(0.01)
            continue

        detections = detector.detect(color)
        real_pan, real_tilt = read_head_pose(robot)

        if on_pose is not None:
            try:
                on_pose(color, detections, real_pan, real_tilt)
            except Exception as exc:
                logger.warning("on_pose callback raised: %s", exc)

        target = pick_target(detections, tp.target_conf_min)

        if target is None:
            lost_frames += 1
            if lost_frames >= tp.lost_frames_timeout:
                logger.info("track: target lost for %d frames; exiting", lost_frames)
                reason = "lost"
                break
            continue

        lost_frames = 0

        h, w = color.shape[:2]
        cx, cy = target.center
        err_x_px = float(cx - w / 2.0)
        err_y_px = float(cy - h / 2.0)
        err_x_rad = err_x_px * rpp_x
        err_y_rad = err_y_px * rpp_y

        delta_pan = tp.pan_sign * tp.kp * err_x_rad
        delta_tilt = tp.tilt_sign * tp.kp * err_y_rad

        delta_pan = clamp(delta_pan, -tp.max_step_rad, tp.max_step_rad)
        delta_tilt = clamp(delta_tilt, -tp.max_step_rad, tp.max_step_rad)

        new_pan = clamp(real_pan + delta_pan, limits.pan_lo, limits.pan_hi)
        new_tilt = clamp(real_tilt + delta_tilt, limits.tilt_lo, limits.tilt_hi)

        # Only issue a move if we're actually asking for something new.
        if abs(new_pan - real_pan) > 1e-3:
            robot.head.move_to("head_pan", new_pan)
        if abs(new_tilt - real_tilt) > 1e-3:
            robot.head.move_to("head_tilt", new_tilt)
        # No wait_command() — continuous update.

    return reason
