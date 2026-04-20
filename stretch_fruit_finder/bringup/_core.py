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

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, ContextManager, Optional

import numpy as np

from fruit_finder.camera import CameraManager
from fruit_finder.detector import Detection, FoodDetector

logger = logging.getLogger(__name__)


def _lock_ctx(lock: Optional[threading.Lock]) -> ContextManager:
    """Return `lock` itself when provided, or a no-op context otherwise.

    Lets the sweep/track loops be written once and work either standalone
    (no lock needed) or alongside another thread that also touches the
    robot (lock required to serialise `push_command()` etc).
    """
    return lock if lock is not None else contextlib.nullcontext()


# ----- type aliases --------------------------------------------------------

# Callback: (color_bgr, detections, real_pan_rad, real_tilt_rad)
PoseCallback = Optional[Callable[[np.ndarray, list, float, float], None]]


# ----- data ---------------------------------------------------------------


@dataclass
class TrackerParams:
    """Runtime tunables. Loaded from config.yaml `tracking:` section with defaults."""
    kp: float = 0.6
    target_conf_min: float = 0.25
    sweep_acquire_conf_min: float = 0.50
    lost_frames_timeout: int = 30
    max_step_rad: float = 0.18
    deadband_rad: float = 0.05
    pan_sign: float = -1.0
    tilt_sign: float = -1.0
    auto_sign_flip: bool = True
    stable_frames_drain: int = 2
    wait_command_in_track: bool = False
    verbose_first_n: int = 10

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
            deadband_rad=float(trk.get("deadband_rad", defaults.deadband_rad)),
            pan_sign=float(trk.get("pan_sign", defaults.pan_sign)),
            tilt_sign=float(trk.get("tilt_sign", defaults.tilt_sign)),
            auto_sign_flip=bool(trk.get("auto_sign_flip", defaults.auto_sign_flip)),
            stable_frames_drain=int(
                trk.get("stable_frames_drain", defaults.stable_frames_drain)
            ),
            wait_command_in_track=bool(
                trk.get("wait_command_in_track", defaults.wait_command_in_track)
            ),
            verbose_first_n=int(trk.get("verbose_first_n", defaults.verbose_first_n)),
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


def center_head(
    robot,
    wait: bool = True,
    robot_lock: Optional[threading.Lock] = None,
) -> None:
    """Best-effort: return head to (pan=0, tilt=0)."""
    try:
        with _lock_ctx(robot_lock):
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
    robot_lock: Optional[threading.Lock] = None,
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
    # detector.target_label == "" means "any food" — in that case we'll
    # narrow to the specific acquired label once sweep finds something,
    # so track() stays locked on one item instead of jumping between them.
    any_food_mode = detector.target_label == ""
    logger.info(
        "sweep: target=%s pan=[%+.2f,%+.2f] step=%+.2f tilt=%+.2f dwell=%.2f max_sweeps=%d",
        "(any food)" if any_food_mode else repr(target_label),
        pan_min, pan_max, pan_step, tilt_target, dwell, max_sweeps,
    )

    # Move to search tilt + center pan before the sweep.
    with _lock_ctx(robot_lock):
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

            with _lock_ctx(robot_lock):
                robot.head.move_to("head_pan", pan)
                robot.wait_command()
            time.sleep(dwell)

            color, _depth = drain_frames(cam, warmup_frames)
            if color is None:
                continue

            with _lock_ctx(robot_lock):
                real_pan, real_tilt = read_head_pose(robot)
            detections = detector.detect(color)

            if on_pose is not None:
                try:
                    on_pose(color, detections, real_pan, real_tilt)
                except Exception as exc:
                    logger.warning("on_pose callback raised: %s", exc)

            target = pick_target(detections, tp.sweep_acquire_conf_min)
            if target is not None:
                if any_food_mode:
                    # Lock the detector onto this specific label for track().
                    detector.set_target(target.label)
                    logger.info(
                        "sweep: narrowed target from (any food) to %r",
                        target.label,
                    )
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
    robot_lock: Optional[threading.Lock] = None,
) -> str:
    """
    Closed-loop P controller: keep the highest-confidence target centered.

    Stop-and-go loop, same pattern as sweep:
      1) drain a few frames + detect (frame is stable post-motion)
      2) compute pixel error -> radian error via camera intrinsics
      3) if target is inside the deadband (|err| < deadband_rad on an axis),
         do nothing on that axis
      4) otherwise command a proportional correction, clamped to max_step_rad
      5) wait_command() for motion to complete before looping

    This is slower per iteration than a continuous controller but eliminates
    motion blur in the per-iteration frame, which is what actually matters
    on a CPU-bound detector.

    The first *commanded move* is a "probe" using 40% of kp and 50% of
    max_step. The next detection checks whether error shrank; if it grew,
    the offending sign is auto-flipped. After that, probing ends.

    Returns: "stopped", "lost", or "error".
    """
    tp = params or TrackerParams.from_config(config)
    limits = get_head_limits(robot)
    rpp_x, rpp_y = rad_per_pixel(cam)
    logger.info(
        "track: kp=%.2f max_step=%.2f deadband=%.3f lost_timeout=%d "
        "pan_sign=%+d tilt_sign=%+d rad_per_px=(%.5f, %.5f) "
        "auto_sign_flip=%s stable_drain=%d",
        tp.kp, tp.max_step_rad, tp.deadband_rad, tp.lost_frames_timeout,
        int(tp.pan_sign), int(tp.tilt_sign), rpp_x, rpp_y,
        tp.auto_sign_flip, tp.stable_frames_drain,
    )

    lost_frames = 0
    reason = "stopped"
    iteration = 0

    # Probe / auto-sign-flip state.
    probe_done = not tp.auto_sign_flip
    probe_err_before: tuple[float, float] | None = None
    probe_delta: tuple[float, float] = (0.0, 0.0)

    while not stop_event.is_set():
        # Stable frame: drain a few so we don't detect on a motion-blurred one.
        color, _depth = drain_frames(cam, max(1, tp.stable_frames_drain))
        if color is None:
            time.sleep(0.01)
            continue

        detections = detector.detect(color)
        with _lock_ctx(robot_lock):
            real_pan, real_tilt = read_head_pose(robot)

        if on_pose is not None:
            try:
                on_pose(color, detections, real_pan, real_tilt)
            except Exception as exc:
                logger.warning("on_pose callback raised: %s", exc)

        target = pick_target(detections, tp.target_conf_min)

        if target is None:
            lost_frames += 1
            if iteration < tp.verbose_first_n:
                logger.info(
                    "track[%02d]: NO TARGET (lost_frames=%d/%d)",
                    iteration, lost_frames, tp.lost_frames_timeout,
                )
            if lost_frames >= tp.lost_frames_timeout:
                logger.info("track: target lost for %d frames; exiting", lost_frames)
                reason = "lost"
                break
            iteration += 1
            continue

        lost_frames = 0

        h, w = color.shape[:2]
        cx, cy = target.center
        err_x_px = float(cx - w / 2.0)
        err_y_px = float(cy - h / 2.0)
        err_x_rad = err_x_px * rpp_x
        err_y_rad = err_y_px * rpp_y

        # --- sign verification (runs on the first detection AFTER a probe move) ---
        if not probe_done and probe_err_before is not None:
            eps = 0.02
            moved_x = abs(probe_delta[0]) > 1e-3
            moved_y = abs(probe_delta[1]) > 1e-3
            old_x, old_y = probe_err_before

            if moved_x and abs(old_x) > eps:
                if abs(err_x_rad) > abs(old_x) + eps:
                    tp.pan_sign = -tp.pan_sign
                    logger.warning(
                        "track: AUTO-FLIPPED pan_sign to %+d (err_x %+.3f -> %+.3f rad)",
                        int(tp.pan_sign), old_x, err_x_rad,
                    )
                else:
                    logger.info(
                        "track: pan_sign %+d OK (err_x %+.3f -> %+.3f rad)",
                        int(tp.pan_sign), old_x, err_x_rad,
                    )
            if moved_y and abs(old_y) > eps:
                if abs(err_y_rad) > abs(old_y) + eps:
                    tp.tilt_sign = -tp.tilt_sign
                    logger.warning(
                        "track: AUTO-FLIPPED tilt_sign to %+d (err_y %+.3f -> %+.3f rad)",
                        int(tp.tilt_sign), old_y, err_y_rad,
                    )
                else:
                    logger.info(
                        "track: tilt_sign %+d OK (err_y %+.3f -> %+.3f rad)",
                        int(tp.tilt_sign), old_y, err_y_rad,
                    )
            probe_done = True

        # --- deadband: skip axes that are already close to centered ---
        within_x = abs(err_x_rad) < tp.deadband_rad
        within_y = abs(err_y_rad) < tp.deadband_rad
        if within_x and within_y:
            if iteration < tp.verbose_first_n:
                logger.info(
                    "track[%02d] DEADBAND: pos=(%+.2f, %+.2f) err_rad=(%+.3f, %+.3f) conf=%.2f",
                    iteration, real_pan, real_tilt, err_x_rad, err_y_rad, target.confidence,
                )
            iteration += 1
            # Don't wait_command — we didn't issue a move.
            continue

        # --- compute correction ---
        probing_now = (not probe_done) and probe_err_before is None
        kp_eff = tp.kp * 0.4 if probing_now else tp.kp
        step_eff = tp.max_step_rad * 0.5 if probing_now else tp.max_step_rad

        delta_pan = 0.0 if within_x else tp.pan_sign * kp_eff * err_x_rad
        delta_tilt = 0.0 if within_y else tp.tilt_sign * kp_eff * err_y_rad
        delta_pan = clamp(delta_pan, -step_eff, step_eff)
        delta_tilt = clamp(delta_tilt, -step_eff, step_eff)

        new_pan = clamp(real_pan + delta_pan, limits.pan_lo, limits.pan_hi)
        new_tilt = clamp(real_tilt + delta_tilt, limits.tilt_lo, limits.tilt_hi)

        if iteration < tp.verbose_first_n:
            logger.info(
                "track[%02d]%s: pos=(%+.2f, %+.2f) err_rad=(%+.3f, %+.3f) "
                "delta=(%+.3f, %+.3f) new=(%+.2f, %+.2f) conf=%.2f",
                iteration, " PROBE" if probing_now else "",
                real_pan, real_tilt,
                err_x_rad, err_y_rad,
                delta_pan, delta_tilt,
                new_pan, new_tilt,
                target.confidence,
            )

        if probing_now:
            probe_err_before = (err_x_rad, err_y_rad)
            probe_delta = (delta_pan, delta_tilt)

        # --- command the move and optionally wait for it to finish ---
        moved_any = False
        with _lock_ctx(robot_lock):
            if abs(new_pan - real_pan) > 1e-3:
                robot.head.move_to("head_pan", new_pan)
                moved_any = True
            if abs(new_tilt - real_tilt) > 1e-3:
                robot.head.move_to("head_tilt", new_tilt)
                moved_any = True

            if moved_any and tp.wait_command_in_track:
                try:
                    robot.wait_command()
                except Exception as exc:
                    logger.warning("wait_command failed: %s", exc)

        iteration += 1

    return reason
