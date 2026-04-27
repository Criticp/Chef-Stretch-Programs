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
    # Last-seen-direction memory + targeted re-acquire on loss.
    trace_max_samples: int = 12
    trace_max_age_s: float = 1.5
    reacquire_extent_rad: float = 0.6
    reacquire_budget_s: float = 2.0
    dt_predict_s: float = 0.5
    reacquire_speed_rad_per_s: float = 0.3
    reacquire_accel_rad_per_s2: float = 0.8

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
            trace_max_samples=int(trk.get("trace_max_samples", defaults.trace_max_samples)),
            trace_max_age_s=float(trk.get("trace_max_age_s", defaults.trace_max_age_s)),
            reacquire_extent_rad=float(
                trk.get("reacquire_extent_rad", defaults.reacquire_extent_rad)
            ),
            reacquire_budget_s=float(
                trk.get("reacquire_budget_s", defaults.reacquire_budget_s)
            ),
            dt_predict_s=float(trk.get("dt_predict_s", defaults.dt_predict_s)),
            reacquire_speed_rad_per_s=float(
                trk.get("reacquire_speed_rad_per_s", defaults.reacquire_speed_rad_per_s)
            ),
            reacquire_accel_rad_per_s2=float(
                trk.get("reacquire_accel_rad_per_s2", defaults.reacquire_accel_rad_per_s2)
            ),
        )


@dataclass
class HeadLimits:
    pan_lo: float
    pan_hi: float
    tilt_lo: float
    tilt_hi: float


class TargetTrace:
    """Bounded history of recent target observations in head-frame angles.

    Each sample is (timestamp_s, abs_pan_rad, abs_tilt_rad) where the abs
    angles are the target's angular position in the head frame, derived from
    `head_pose + pan/tilt_sign * pixel_error_rad`. Used to estimate the
    target's angular velocity so we can slew the head in the right direction
    when the target slips off-screen.
    """

    def __init__(self, max_samples: int = 12, max_age_s: float = 1.5):
        self.max_samples = max(2, int(max_samples))
        self.max_age_s = float(max_age_s)
        self._buf: list[tuple[float, float, float]] = []

    def append(self, t: float, abs_pan: float, abs_tilt: float) -> None:
        self._buf.append((t, abs_pan, abs_tilt))
        cutoff = t - self.max_age_s
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.pop(0)
        while len(self._buf) > self.max_samples:
            self._buf.pop(0)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def last(self) -> Optional[tuple[float, float, float]]:
        return self._buf[-1] if self._buf else None


def estimate_target_velocity(trace: TargetTrace) -> tuple[float, float]:
    """Linear-regression slope of (abs_pan, abs_tilt) vs time over the trace.

    Returns (vp_rad_per_s, vt_rad_per_s). Returns (0, 0) if too few samples
    or too small a time span to make a meaningful estimate.
    """
    buf = trace._buf  # noqa: SLF001 — intentional internal access
    if len(buf) < 3:
        return (0.0, 0.0)
    ts = np.asarray([s[0] for s in buf], dtype=float)
    span = float(ts[-1] - ts[0])
    if span < 0.2:
        return (0.0, 0.0)
    # Centre time so the regression is well-conditioned.
    t = ts - ts.mean()
    pans = np.asarray([s[1] for s in buf], dtype=float)
    tilts = np.asarray([s[2] for s in buf], dtype=float)
    denom = float(np.dot(t, t))
    if denom <= 0.0:
        return (0.0, 0.0)
    vp = float(np.dot(t, pans - pans.mean()) / denom)
    vt = float(np.dot(t, tilts - tilts.mean()) / denom)
    return (vp, vt)


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


def _resolve_tilt_rows(nav: dict, limits: HeadLimits) -> list[float]:
    """Read `search_tilt_rows_rad` (preferred) or fall back to single `search_tilt_rad`."""
    raw = nav.get("search_tilt_rows_rad")
    if raw is None:
        raw = [nav.get("search_tilt_rad", -0.6)]
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    rows = [clamp(float(t), limits.tilt_lo, limits.tilt_hi) for t in raw]
    # De-duplicate while preserving order; if everything collapses, keep one.
    seen: list[float] = []
    for t in rows:
        if not seen or abs(t - seen[-1]) > 1e-3:
            seen.append(t)
    return seen or [clamp(-0.6, limits.tilt_lo, limits.tilt_hi)]


_HEAD_SPEED_API_LOGGED: dict[str, bool] = {}


def _move_head_joint(robot, joint_name: str, x_des: float, v_des: Optional[float], a_des: Optional[float]) -> None:
    """Move a head joint at a capped speed/accel.

    Stretch's head Dynamixels run at the joint's default profile velocity
    (~3 rad/s for head_pan) unless a slower one is explicitly applied.
    `robot.head.move_to(name, x)` is a thin wrapper that does NOT forward
    speed caps in every stretch_body version, so we go straight to the
    joint object — which uniformly supports `move_to(x, v_des=, a_des=)`
    on DynamixelHelloXL430 — and only fall back to setting the profile
    velocity registers separately if even that signature is missing.

    A loud warning is logged the first time we have to fall back so
    runaway speed isn't silently masked again.
    """
    if v_des is None and a_des is None:
        robot.head.move_to(joint_name, x_des)
        return

    try:
        joint = robot.head.get_joint(joint_name)
    except Exception as exc:
        logger.warning("get_joint(%r) failed (%s); using head.move_to without speed cap", joint_name, exc)
        robot.head.move_to(joint_name, x_des)
        return

    # Preferred: joint-level move_to with kwargs (most stretch_body versions).
    try:
        joint.move_to(x_des, v_des=v_des, a_des=a_des)
        if not _HEAD_SPEED_API_LOGGED.get("kwargs"):
            _HEAD_SPEED_API_LOGGED["kwargs"] = True
            logger.info("head speed cap applied via joint.move_to(v_des=, a_des=)")
        return
    except TypeError:
        pass
    except Exception as exc:
        logger.warning("joint.move_to(v_des) failed unexpectedly (%s); trying profile-register fallback", exc)

    # Fallback: write the dynamixel profile velocity/accel registers, then move.
    applied_any = False
    for attr, value in (
        ("set_motion_profile_velocity", v_des),
        ("set_motion_profile_acceleration", a_des),
    ):
        if value is None:
            continue
        fn = getattr(joint, attr, None)
        if fn is None:
            continue
        try:
            fn(value)
            applied_any = True
        except Exception as exc:
            logger.warning("joint.%s(%.2f) failed (%s)", attr, value, exc)
    try:
        joint.move_to(x_des)
        if applied_any and not _HEAD_SPEED_API_LOGGED.get("profile"):
            _HEAD_SPEED_API_LOGGED["profile"] = True
            logger.info("head speed cap applied via set_motion_profile_velocity/acceleration")
        elif not applied_any and not _HEAD_SPEED_API_LOGGED.get("none"):
            _HEAD_SPEED_API_LOGGED["none"] = True
            logger.warning(
                "no head speed-cap API available on this stretch_body build; "
                "head will run at default profile velocity"
            )
    except Exception as exc:
        logger.warning("joint.move_to(x) fallback failed (%s); using head.move_to", exc)
        robot.head.move_to(joint_name, x_des)


def _slew_and_detect(
    robot,
    cam: CameraManager,
    detector: FoodDetector,
    *,
    pan_target: float,
    conf_min: float,
    slew_speed_rad_per_s: float,
    slew_accel_rad_per_s2: float,
    pan_arrival_tol_rad: float,
    time_budget_s: float,
    stop_event: threading.Event,
    on_pose: PoseCallback,
    robot_lock: Optional[threading.Lock],
    tilt_target: Optional[float] = None,
    tilt_arrival_tol_rad: float = 0.05,
    poll_period_s: float = 0.03,
) -> Optional[tuple[Detection, float, float]]:
    """Slew head pan (and optionally tilt) non-blockingly while detecting.

    Returns (detection, pan, tilt) on first hit at/above `conf_min`, or None
    if the slew completes / stop_event fires / the time budget runs out.

    `slew_speed_rad_per_s` is passed to the joint as v_des so the head
    actually moves at that speed instead of the (much faster) default
    profile velocity. On a hit, both pan and (if commanded) tilt are
    commanded back to the live pose to halt the in-flight slew in place.
    """
    with _lock_ctx(robot_lock):
        start_pan, start_tilt = read_head_pose(robot)
        _move_head_joint(robot, "head_pan", pan_target,
                         v_des=slew_speed_rad_per_s, a_des=slew_accel_rad_per_s2)
        if tilt_target is not None:
            _move_head_joint(robot, "head_tilt", tilt_target,
                             v_des=slew_speed_rad_per_s, a_des=slew_accel_rad_per_s2)

    pan_distance = abs(pan_target - start_pan)
    tilt_distance = abs((tilt_target or start_tilt) - start_tilt)
    distance = max(pan_distance, tilt_distance)
    ideal_time = distance / max(0.05, slew_speed_rad_per_s)
    deadline = time.time() + max(time_budget_s, ideal_time * 1.8 + 0.5)

    last_log_pan: Optional[float] = None
    while not stop_event.is_set():
        if time.time() > deadline:
            return None

        color, _depth = cam.get_frames()
        if color is None:
            time.sleep(poll_period_s)
            continue

        with _lock_ctx(robot_lock):
            real_pan, real_tilt = read_head_pose(robot)

        detections = detector.detect(color)
        if on_pose is not None:
            try:
                on_pose(color, detections, real_pan, real_tilt)
            except Exception as exc:
                logger.warning("on_pose callback raised: %s", exc)

        target = pick_target(detections, conf_min)
        if target is not None:
            # Halt motion in place. Use the default (fast) profile here so
            # the joint stops promptly rather than coasting at slew speed.
            with _lock_ctx(robot_lock):
                robot.head.move_to("head_pan", real_pan)
                if tilt_target is not None:
                    robot.head.move_to("head_tilt", real_tilt)
            return (target, real_pan, real_tilt)

        if last_log_pan is None or abs(real_pan - last_log_pan) > 0.5:
            last_log_pan = real_pan
            logger.debug("slew: pan=%+.2f -> %+.2f", real_pan, pan_target)

        pan_arrived = abs(real_pan - pan_target) <= pan_arrival_tol_rad
        tilt_arrived = (
            tilt_target is None or abs(real_tilt - tilt_target) <= tilt_arrival_tol_rad
        )
        if pan_arrived and tilt_arrived:
            return None

        time.sleep(poll_period_s)

    return None


def reacquire_directional(
    robot,
    cam: CameraManager,
    detector: FoodDetector,
    tp: TrackerParams,
    limits: HeadLimits,
    *,
    last_abs_pan: float,
    last_abs_tilt: float,
    vel: tuple[float, float],
    stop_event: threading.Event,
    on_pose: PoseCallback,
    robot_lock: Optional[threading.Lock],
) -> Optional[tuple[Detection, float, float]]:
    """One-shot directional re-acquire after the tracker loses the target.

    If the target had measurable angular velocity when it was last seen,
    slew the head along that velocity vector for up to `reacquire_extent_rad`
    (or `reacquire_budget_s`, whichever ends first), running the detector in
    flight. If the target was effectively static, do a small symmetric scan
    around the last known pan position instead.

    Returns (detection, pan, tilt) on first hit at/above `target_conf_min`,
    or None if the move completes / budget runs out / stop_event fires.
    """
    vp, vt = vel
    speed = float(np.hypot(vp, vt))
    extent = max(0.1, tp.reacquire_extent_rad)
    moving = speed > 0.05  # rad/s — anything less is treated as "stationary"
    slew_speed = tp.reacquire_speed_rad_per_s
    slew_accel = tp.reacquire_accel_rad_per_s2

    if moving:
        # Predict where the target is "now" (dt_predict ahead of last sample),
        # then extend further along the same direction up to the full extent.
        dt = max(0.0, tp.dt_predict_s)
        pred_pan = last_abs_pan + vp * dt
        pred_tilt = last_abs_tilt + vt * dt
        ux = vp / speed
        uy = vt / speed
        end_pan = clamp(pred_pan + ux * extent, limits.pan_lo, limits.pan_hi)
        end_tilt = clamp(pred_tilt + uy * extent, limits.tilt_lo, limits.tilt_hi)

        logger.info(
            "reacquire(directional): vel=(%+.2f, %+.2f) rad/s last=(%+.2f, %+.2f) -> end=(%+.2f, %+.2f) speed=%.2f",
            vp, vt, last_abs_pan, last_abs_tilt, end_pan, end_tilt, slew_speed,
        )

        return _slew_and_detect(
            robot, cam, detector,
            pan_target=end_pan,
            tilt_target=end_tilt,
            conf_min=tp.target_conf_min,
            slew_speed_rad_per_s=slew_speed,
            slew_accel_rad_per_s2=slew_accel,
            pan_arrival_tol_rad=0.05,
            time_budget_s=tp.reacquire_budget_s,
            stop_event=stop_event,
            on_pose=on_pose,
            robot_lock=robot_lock,
        )

    # Stationary fallback: small symmetric scan around the last position.
    half = extent * 0.5
    left = clamp(last_abs_pan - half, limits.pan_lo, limits.pan_hi)
    right = clamp(last_abs_pan + half, limits.pan_lo, limits.pan_hi)
    tilt_hold = clamp(last_abs_tilt, limits.tilt_lo, limits.tilt_hi)

    logger.info(
        "reacquire(scan): no velocity, scanning pan [%+.2f, %+.2f] at tilt %+.2f speed=%.2f",
        left, right, tilt_hold, slew_speed,
    )

    half_budget = max(0.5, tp.reacquire_budget_s * 0.5)

    hit = _slew_and_detect(
        robot, cam, detector,
        pan_target=left,
        tilt_target=tilt_hold,
        conf_min=tp.target_conf_min,
        slew_speed_rad_per_s=slew_speed,
        slew_accel_rad_per_s2=slew_accel,
        pan_arrival_tol_rad=0.05,
        time_budget_s=half_budget,
        stop_event=stop_event,
        on_pose=on_pose,
        robot_lock=robot_lock,
    )
    if hit is not None or stop_event.is_set():
        return hit

    return _slew_and_detect(
        robot, cam, detector,
        pan_target=right,
        tilt_target=tilt_hold,
        conf_min=tp.target_conf_min,
        slew_speed_rad_per_s=slew_speed,
        slew_accel_rad_per_s2=slew_accel,
        pan_arrival_tol_rad=0.05,
        time_budget_s=half_budget,
        stop_event=stop_event,
        on_pose=on_pose,
        robot_lock=robot_lock,
    )


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
    Continuous-slew search across pan x multiple tilt rows.

    The head pan is commanded across `[search_pan_min_rad, search_pan_max_rad]`
    without waiting for arrival, and the detector runs in flight on every
    fresh camera frame. On any detection above `sweep_acquire_conf_min`,
    motion is halted in place and (detection, pan, tilt) is returned.

    Tilt rows come from `search_tilt_rows_rad` (list) — defaults to floor /
    mid / counter-and-hand height — so a single sweep covers fruit at
    multiple heights, not just on the floor. After every tilt row is
    visited, that counts as one "full sweep"; up to `max_search_sweeps`
    full sweeps are attempted before giving up. Pan direction alternates
    each row (boustrophedon) so we don't jump from one extreme back to the
    other between rows.

    Returns None if `stop_event` is set or all sweeps complete with no hit.
    """
    nav = config["navigation"]
    tp = TrackerParams.from_config(config)
    limits = get_head_limits(robot)

    pan_min = clamp(float(nav["search_pan_min_rad"]), limits.pan_lo, limits.pan_hi)
    pan_max = clamp(float(nav["search_pan_max_rad"]), limits.pan_lo, limits.pan_hi)
    tilt_rows = _resolve_tilt_rows(nav, limits)
    slew_speed = float(nav.get("search_slew_speed_rad_per_s", 0.3))
    slew_accel = float(nav.get("search_slew_accel_rad_per_s2", 0.8))
    max_sweeps = int(nav.get("max_search_sweeps", 3))

    detector.set_target(target_label)
    # detector.target_label == "" means "any food" — in that case we'll
    # narrow to the specific acquired label once sweep finds something,
    # so track() stays locked on one item instead of jumping between them.
    any_food_mode = detector.target_label == ""
    logger.info(
        "sweep: target=%s pan=[%+.2f,%+.2f] tilt_rows=%s slew=%.2f rad/s a=%.2f rad/s^2 max_sweeps=%d",
        "(any food)" if any_food_mode else repr(target_label),
        pan_min, pan_max,
        "[" + ", ".join(f"{t:+.2f}" for t in tilt_rows) + "]",
        slew_speed, slew_accel, max_sweeps,
    )

    pan_arrival_tol = 0.05  # rad; "stroke complete" tolerance
    # Time budget per stroke — generous fallback if move_to never reports arrival.
    stroke_budget_s = max(3.0, (pan_max - pan_min) / max(0.05, slew_speed) * 1.8 + 1.0)

    pan_extremes = (pan_min, pan_max)
    # Start each new full sweep at the same end so coverage is symmetric.
    for sweep_n in range(max_sweeps):
        if stop_event.is_set():
            return None

        for row_idx, tilt_target in enumerate(tilt_rows):
            if stop_event.is_set():
                return None

            # Boustrophedon: alternate pan direction across rows AND across sweeps.
            forward = ((sweep_n + row_idx) % 2) == 0
            stroke_start, stroke_end = (pan_extremes[0], pan_extremes[1]) if forward else (pan_extremes[1], pan_extremes[0])

            # Pre-position to the stroke start + this row's tilt before slewing.
            with _lock_ctx(robot_lock):
                robot.head.move_to("head_pan", stroke_start)
                robot.head.move_to("head_tilt", tilt_target)
                robot.wait_command()
            time.sleep(0.15)
            drain_frames(cam, warmup_frames)

            if stop_event.is_set():
                return None

            logger.info(
                "sweep[%d/%d row=%d tilt=%+.2f]: slewing pan %+.2f -> %+.2f",
                sweep_n + 1, max_sweeps, row_idx, tilt_target,
                stroke_start, stroke_end,
            )

            hit = _slew_and_detect(
                robot, cam, detector,
                pan_target=stroke_end,
                conf_min=tp.sweep_acquire_conf_min,
                slew_speed_rad_per_s=slew_speed,
                slew_accel_rad_per_s2=slew_accel,
                pan_arrival_tol_rad=pan_arrival_tol,
                time_budget_s=stroke_budget_s,
                stop_event=stop_event,
                on_pose=on_pose,
                robot_lock=robot_lock,
            )

            if hit is not None:
                target, real_pan, real_tilt = hit
                if any_food_mode:
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

    Loss handling: every successful detection appends the target's absolute
    head-frame angle to a `TargetTrace`. When the target is missing for
    `lost_frames_timeout` consecutive frames, a single-shot directional
    re-acquire slews the head along the target's last estimated velocity
    (or scans symmetrically around the last position if it was static)
    before declaring the target officially lost.

    Returns: "stopped", "lost", or "error".
    """
    tp = params or TrackerParams.from_config(config)
    limits = get_head_limits(robot)
    rpp_x, rpp_y = rad_per_pixel(cam)
    logger.info(
        "track: kp=%.2f max_step=%.2f deadband=%.3f lost_timeout=%d "
        "pan_sign=%+d tilt_sign=%+d rad_per_px=(%.5f, %.5f) "
        "auto_sign_flip=%s stable_drain=%d reacquire_extent=%.2f budget=%.2fs",
        tp.kp, tp.max_step_rad, tp.deadband_rad, tp.lost_frames_timeout,
        int(tp.pan_sign), int(tp.tilt_sign), rpp_x, rpp_y,
        tp.auto_sign_flip, tp.stable_frames_drain,
        tp.reacquire_extent_rad, tp.reacquire_budget_s,
    )

    lost_frames = 0
    reason = "stopped"
    iteration = 0

    # Probe / auto-sign-flip state.
    probe_done = not tp.auto_sign_flip
    probe_err_before: tuple[float, float] | None = None
    probe_delta: tuple[float, float] = (0.0, 0.0)

    # Target trace (last-seen direction memory) + one-shot re-acquire flag.
    trace = TargetTrace(tp.trace_max_samples, tp.trace_max_age_s)
    reacquire_attempted = False

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
                if not reacquire_attempted and trace.last is not None:
                    reacquire_attempted = True
                    vel = estimate_target_velocity(trace)
                    _, last_abs_pan, last_abs_tilt = trace.last
                    logger.info(
                        "track: target lost for %d frames; attempting directional re-acquire",
                        lost_frames,
                    )
                    hit = reacquire_directional(
                        robot, cam, detector, tp, limits,
                        last_abs_pan=last_abs_pan,
                        last_abs_tilt=last_abs_tilt,
                        vel=vel,
                        stop_event=stop_event,
                        on_pose=on_pose,
                        robot_lock=robot_lock,
                    )
                    if hit is not None:
                        logger.info(
                            "track: re-acquired %r at pan=%+.2f tilt=%+.2f conf=%.2f",
                            hit[0].label, hit[1], hit[2], hit[0].confidence,
                        )
                        lost_frames = 0
                        trace.clear()
                        iteration += 1
                        continue
                    logger.info("track: directional re-acquire failed; target lost")
                else:
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

        # Record where the target is in head-frame angles, for velocity
        # estimation if it slips off-screen later. Using the same sign
        # convention as the controller so the resulting velocity matches
        # the head-pan axis.
        trace.append(
            time.time(),
            real_pan + tp.pan_sign * err_x_rad,
            real_tilt + tp.tilt_sign * err_y_rad,
        )

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


# ----- hover above locked target -------------------------------------------


def _safe_call(label: str, fn) -> None:
    """Run fn(); log a warning on any exception so a single bad call doesn't
    abort the rest of a hover sequence."""
    try:
        fn()
    except Exception as exc:
        logger.warning("hover: %s failed: %s", label, exc)


def _read_joint_pos(robot, joint_attr: str) -> float:
    try:
        return float(getattr(robot, joint_attr).status.get("pos", 0.0))
    except Exception:
        return 0.0


def _halt_joint(robot, joint_attr: str, robot_lock: Optional[threading.Lock], v: float, a: float) -> None:
    """Stop the named stepper joint in place by commanding it to its current position.

    Acquires the lock so other threads see a coherent stopped state.
    Works for `robot.arm` and `robot.lift` (they share the same API).
    """
    with _lock_ctx(robot_lock):
        cur = _read_joint_pos(robot, joint_attr)
        _safe_call(f"{joint_attr}.halt",
                   lambda: getattr(robot, joint_attr).move_to(cur, v, a))
        _safe_call(f"push_command({joint_attr}.halt)", robot.push_command)


def _scan_joint_for_visibility(
    robot,
    joint_attr: str,
    cam: CameraManager,
    detector: FoodDetector,
    *,
    target_pos: float,
    scan_v: float,
    scan_a: float,
    halt_v: float,
    halt_a: float,
    target_conf_min: float,
    wait_for: str,
    arrival_tol: float = 0.005,
    transition_threshold: int = 3,
    poll_period_s: float = 0.03,
    time_budget_s: float = 30.0,
    stop_event: threading.Event,
    on_pose: PoseCallback,
    robot_lock: Optional[threading.Lock],
) -> str:
    """Slew the named stepper joint (arm or lift) at scan speed, watching the
    detector for a target-visibility transition.

    `wait_for` selects the state machine:
      - "reveal_after_occlusion": expects target visible at start. Returns
        "passed" when the target has been LOST for `transition_threshold`
        consecutive cycles AND THEN visible again for the same threshold.
      - "occlusion": returns "passed" when the target is lost for the
        threshold count.

    Returns one of:
      "passed"        — desired transition occurred; joint is halted in place.
      "max_reached"   — joint reached `target_pos` without the transition.
      "stopped"       — stop_event fired; joint halted.
      "timeout"       — `time_budget_s` expired; joint halted.
    """
    assert wait_for in ("reveal_after_occlusion", "occlusion"), wait_for

    with _lock_ctx(robot_lock):
        start_pos = _read_joint_pos(robot, joint_attr)
        _safe_call(f"{joint_attr}.move_to(scan)",
                   lambda: getattr(robot, joint_attr).move_to(target_pos, scan_v, scan_a))
        _safe_call(f"push_command({joint_attr}.scan)", robot.push_command)

    state = "INITIAL_VISIBLE" if wait_for == "reveal_after_occlusion" else "WAIT_OCCLUDE"
    consecutive_lost = 0
    consecutive_visible = 0

    direction = 1 if target_pos > start_pos else -1
    deadline = time.time() + time_budget_s

    logger.info(
        "scan(%s): %s %.3f -> %.3f (v=%.2f) state=%s",
        joint_attr, wait_for, start_pos, target_pos, scan_v, state,
    )

    while not stop_event.is_set():
        if time.time() > deadline:
            _halt_joint(robot, joint_attr, robot_lock, halt_v, halt_a)
            logger.warning("scan(%s): timed out after %.1fs in state %s",
                           joint_attr, time_budget_s, state)
            return "timeout"

        color, _depth = cam.get_frames()
        if color is None:
            time.sleep(poll_period_s)
            continue

        with _lock_ctx(robot_lock):
            real_pan, real_tilt = read_head_pose(robot)
            cur_pos = _read_joint_pos(robot, joint_attr)

        detections = detector.detect(color)
        if on_pose is not None:
            try:
                on_pose(color, detections, real_pan, real_tilt)
            except Exception as exc:
                logger.warning("on_pose callback raised: %s", exc)

        target_seen = pick_target(detections, target_conf_min) is not None

        if state == "INITIAL_VISIBLE":
            if not target_seen:
                consecutive_lost += 1
                if consecutive_lost >= transition_threshold:
                    logger.info("scan(%s): occluded at %.3f — waiting for reveal",
                                joint_attr, cur_pos)
                    state = "WAIT_REVEAL"
                    consecutive_lost = 0
            else:
                consecutive_lost = 0
        elif state == "WAIT_REVEAL":
            if target_seen:
                consecutive_visible += 1
                if consecutive_visible >= transition_threshold:
                    _halt_joint(robot, joint_attr, robot_lock, halt_v, halt_a)
                    logger.info("scan(%s): revealed at %.3f", joint_attr, cur_pos)
                    return "passed"
            else:
                consecutive_visible = 0
        elif state == "WAIT_OCCLUDE":
            if not target_seen:
                consecutive_lost += 1
                if consecutive_lost >= transition_threshold:
                    _halt_joint(robot, joint_attr, robot_lock, halt_v, halt_a)
                    logger.info("scan(%s): occluded at %.3f", joint_attr, cur_pos)
                    return "passed"
            else:
                consecutive_lost = 0

        # Direction-aware arrival check.
        if direction > 0 and cur_pos >= target_pos - arrival_tol:
            logger.info("scan(%s): reached target %.3f without transition (state=%s)",
                        joint_attr, cur_pos, state)
            return "max_reached"
        if direction < 0 and cur_pos <= target_pos + arrival_tol:
            logger.info("scan(%s): reached target %.3f without transition (state=%s)",
                        joint_attr, cur_pos, state)
            return "max_reached"

        time.sleep(poll_period_s)

    _halt_joint(robot, joint_attr, robot_lock, halt_v, halt_a)
    return "stopped"


def hover_above_target(
    robot,
    cam: CameraManager,
    detector: FoodDetector,
    config: dict,
    stop_event: threading.Event,
    arm_controller,
    on_pose: PoseCallback = None,
    robot_lock: Optional[threading.Lock] = None,
    wrist_yaw_target: float = 0.0,
) -> str:
    """Visual-servo hover: extend until the fruit is revealed past the
    gripper, then retract until it is occluded again.

    Caller is expected to have stopped the active track loop first so the
    head is held at the lock-on pose. The fruit's lateral position is
    determined visually by the gripper sweeping through the camera-to-fruit
    line of sight (rather than by deprojected depth) — much more accurate
    in the presence of detector/depth noise.

    Steps:
      1. Sample a single 3D position to pick the lift height (z + clearance)
         and to detect the wrong-side case.
      2. Wrong-side: run the existing position_above_unreachable fallback.
      3. Right-side:
         a. Pose the wrist outward (yaw=0, pitch=0, roll=0), gripper open.
         b. Move the lift to clearance height (blocking).
         c. Phase A: extend the arm slowly toward max_extension. Watch the
            detector: wait for the fruit to disappear (occluded), then for
            it to reappear (gripper passed it). Halt.
         d. Phase B: retract the arm slowly toward 0. Watch the detector:
            stop the moment the fruit becomes occluded again — that point
            is the precise hover target.

    Returns one of:
      "hovered"          — full extend+retract cycle completed.
      "no_target"        — no detection at the lock-on pose.
      "no_depth"         — invalid depth at the target pixel.
      "extend_no_pass"   — arm reached max_extension without the fruit being
                           passed; arm retracted to 0 before returning.
      "retract_no_cover" — arm got back to 0 without re-occluding the fruit.
      "stopped"          — stop_event fired mid-sequence.
      "error"            — frames unavailable.

    The head is never commanded by this function.
    """
    arm_kbd = config.get("arm_keyboard", {}) or {}
    lift_v = float(arm_kbd.get("lift_v_m_s", 0.5))
    lift_a = float(arm_kbd.get("lift_a_m_s2", 0.4))
    arm_v_halt = float(arm_kbd.get("arm_v_m_s", 0.5))
    arm_a_halt = float(arm_kbd.get("arm_a_m_s2", 0.4))
    wrist_v = float(arm_kbd.get("wrist_v_rad_s", 2.0))
    wrist_a = float(arm_kbd.get("wrist_a_rad_s2", 4.0))
    grip_v = float(arm_kbd.get("gripper_v", 5.0))
    grip_a = float(arm_kbd.get("gripper_a", 10.0))

    arm_cfg = config.get("arm", {}) or {}
    max_extension = float(arm_cfg.get("max_extension_m", 0.5))
    max_lift = float(arm_cfg.get("max_lift_m", 1.05))
    stow_lift = float(arm_cfg.get("stow_lift_m", 0.6))
    # Hover uses a slightly-closed gripper (narrow but not pinched) for
    # clean occlusion during the visual servo. Falls back to 25 if not set.
    grip_pct = float(arm_cfg.get("hover_gripper_pct", 25.0))

    trk_cfg = config.get("tracking", {}) or {}
    target_conf_min = float(trk_cfg.get("target_conf_min", 0.25))

    # Slow scan envelope so the detector at ~5 Hz has multiple frames per
    # cm of motion. Hard-coded here (not in config) to keep the visual
    # servo predictable; promote to config if we want to tune later.
    scan_v_arm = 0.05
    scan_a_arm = 0.2
    scan_v_lift = 0.05
    scan_a_lift = 0.2

    if stop_event.is_set():
        return "stopped"

    color, depth = drain_frames(cam, 6)
    if color is None or depth is None:
        logger.warning("hover: no frames available")
        return "error"

    with _lock_ctx(robot_lock):
        real_pan, real_tilt = read_head_pose(robot)

    detections = detector.detect(color)
    if on_pose is not None:
        try:
            on_pose(color, detections, real_pan, real_tilt)
        except Exception as exc:
            logger.warning("on_pose callback raised: %s", exc)

    target = pick_target(detections, conf_min=0.20)
    if target is None:
        logger.info("hover: no target in current frame; aborting")
        return "no_target"

    cx, cy = target.center
    target_3d = cam.pixel_to_robot_frame(cx, cy, depth, real_pan, real_tilt)
    if target_3d is None:
        logger.info("hover: invalid depth at target center pixel; aborting")
        return "no_depth"

    x_fwd, y_left, z_up = float(target_3d[0]), float(target_3d[1]), float(target_3d[2])
    logger.info(
        "hover: target_3d=[x=%+.2f, y=%+.2f, z=%+.2f] head=(pan=%+.2f, tilt=%+.2f) conf=%.2f",
        x_fwd, y_left, z_up, real_pan, real_tilt, target.confidence,
    )

    # Wrong-side fallback: arm only extends right; if the fruit is on the
    # left there's nothing to scan toward. Use the existing pose helper.
    if y_left > -0.05:
        logger.info(
            "hover: fruit not on the right (y_left=%+.2f); retracting arm and lifting to clearance",
            y_left,
        )
        commands = arm_controller.position_above_unreachable(z_up)
        cmd_by_joint = {c.joint: c.value for c in commands}
        lift_target_unreachable = float(cmd_by_joint.get("lift", stow_lift))
        with _lock_ctx(robot_lock):
            _safe_call("wrist_yaw",
                       lambda: robot.end_of_arm.move_to("wrist_yaw", wrist_yaw_target, wrist_v, wrist_a))
            _safe_call("wrist_pitch",
                       lambda: robot.end_of_arm.move_to("wrist_pitch", 0.0, wrist_v, wrist_a))
            _safe_call("wrist_roll",
                       lambda: robot.end_of_arm.move_to("wrist_roll", 0.0, wrist_v, wrist_a))
            _safe_call("gripper.set",
                       lambda: robot.end_of_arm.move_to("stretch_gripper", grip_pct, grip_v, grip_a))
            _safe_call("arm.retract",
                       lambda: robot.arm.move_to(0.0, arm_v_halt, arm_a_halt))
            _safe_call("push_command", robot.push_command)
            _safe_call("wait_command", robot.wait_command)
            _safe_call("lift.move_to",
                       lambda: robot.lift.move_to(lift_target_unreachable, lift_v, lift_a))
            _safe_call("push_command(lift)", robot.push_command)
            _safe_call("wait_command(lift)", robot.wait_command)
        return "hovered"

    # Right-side: full visual-servo path. Lift first to find the right
    # height, then arm extension to find the right lateral position.

    # 1. Pose wrist (yaw honors the GUI Gripper-orientation toggle) + apply
    #    slightly-closed gripper.
    if stop_event.is_set():
        return "stopped"
    with _lock_ctx(robot_lock):
        _safe_call("wrist_yaw",
                   lambda: robot.end_of_arm.move_to("wrist_yaw", wrist_yaw_target, wrist_v, wrist_a))
        _safe_call("wrist_pitch",
                   lambda: robot.end_of_arm.move_to("wrist_pitch", 0.0, wrist_v, wrist_a))
        _safe_call("wrist_roll",
                   lambda: robot.end_of_arm.move_to("wrist_roll", 0.0, wrist_v, wrist_a))
        _safe_call("gripper.set",
                   lambda: robot.end_of_arm.move_to("stretch_gripper", grip_pct, grip_v, grip_a))

    if stop_event.is_set():
        return "stopped"

    # 2. Lift visual servo: lift until the gripper occludes the fruit.
    #    The lift starts from wherever the arm currently is (typically
    #    stow_lift after a Stow press) and rises to find the height at
    #    which the gripper crosses the camera-to-fruit line of sight.
    logger.info("hover: phase 0 — lift visual servo to find correct height")
    lift_result = _scan_joint_for_visibility(
        robot, "lift", cam, detector,
        target_pos=max_lift,
        scan_v=scan_v_lift, scan_a=scan_a_lift,
        halt_v=lift_v, halt_a=lift_a,
        target_conf_min=target_conf_min,
        wait_for="occlusion",
        stop_event=stop_event,
        on_pose=on_pose,
        robot_lock=robot_lock,
    )
    if lift_result == "stopped":
        return "stopped"
    if lift_result != "passed":
        logger.warning("hover: lift phase ended as %r — lowering back to stow_lift", lift_result)
        with _lock_ctx(robot_lock):
            _safe_call("lift.recover",
                       lambda: robot.lift.move_to(stow_lift, lift_v, lift_a))
            _safe_call("push_command(lift recover)", robot.push_command)
            _safe_call("wait_command(lift recover)", robot.wait_command)
        return "lift_no_cover"

    if stop_event.is_set():
        return "stopped"

    # 3. Phase A — extend until the fruit reappears past the gripper.
    logger.info("hover: phase A — extending to find reveal-after-occlusion")
    extend_result = _scan_joint_for_visibility(
        robot, "arm", cam, detector,
        target_pos=max_extension,
        scan_v=scan_v_arm, scan_a=scan_a_arm,
        halt_v=arm_v_halt, halt_a=arm_a_halt,
        target_conf_min=target_conf_min,
        wait_for="reveal_after_occlusion",
        stop_event=stop_event,
        on_pose=on_pose,
        robot_lock=robot_lock,
    )
    if extend_result == "stopped":
        return "stopped"
    if extend_result != "passed":
        logger.warning("hover: extend phase ended as %r — retracting", extend_result)
        with _lock_ctx(robot_lock):
            _safe_call("arm.retract",
                       lambda: robot.arm.move_to(0.0, arm_v_halt, arm_a_halt))
            _safe_call("push_command", robot.push_command)
            _safe_call("wait_command", robot.wait_command)
        return "extend_no_pass"

    if stop_event.is_set():
        return "stopped"

    # 4. Phase B — retract until the fruit becomes occluded again.
    logger.info("hover: phase B — retracting to find first occlusion")
    retract_result = _scan_joint_for_visibility(
        robot, "arm", cam, detector,
        target_pos=0.0,
        scan_v=scan_v_arm, scan_a=scan_a_arm,
        halt_v=arm_v_halt, halt_a=arm_a_halt,
        target_conf_min=target_conf_min,
        wait_for="occlusion",
        stop_event=stop_event,
        on_pose=on_pose,
        robot_lock=robot_lock,
    )
    if retract_result == "stopped":
        return "stopped"
    if retract_result != "passed":
        logger.warning("hover: retract phase ended as %r", retract_result)
        return "retract_no_cover"

    final_lift = _read_joint_pos(robot, "lift")
    final_arm = _read_joint_pos(robot, "arm")
    logger.info("hover: complete (lift=%.3f arm=%.3f, gripper-occlusion locked)",
                final_lift, final_arm)
    return "hovered"


def hover_hold_and_watch(
    robot,
    cam: CameraManager,
    detector: FoodDetector,
    config: dict,
    stop_event: threading.Event,
    target_conf_min: float,
    on_pose: PoseCallback = None,
    robot_lock: Optional[threading.Lock] = None,
    poll_period_s: float = 0.05,
) -> str:
    """Hold head still after a hover, but keep detecting; return when the
    target reappears so the caller can re-enter active tracking.

    No head motion at all — this is the user's "freeze the head" mode.
    Returns "target_visible" when a target detection appears with
    confidence >= target_conf_min, or "stopped" when stop_event fires.
    """
    logger.info(
        "hover_hold: passive watch (target_conf_min=%.2f) — head will not move until target reappears",
        target_conf_min,
    )
    while not stop_event.is_set():
        color, _depth = cam.get_frames()
        if color is None:
            time.sleep(poll_period_s)
            continue

        with _lock_ctx(robot_lock):
            real_pan, real_tilt = read_head_pose(robot)
        detections = detector.detect(color)

        if on_pose is not None:
            try:
                on_pose(color, detections, real_pan, real_tilt)
            except Exception as exc:
                logger.warning("on_pose callback raised: %s", exc)

        if pick_target(detections, target_conf_min) is not None:
            logger.info("hover_hold: target reappeared; resuming active track")
            return "target_visible"

        time.sleep(poll_period_s)

    return "stopped"
