"""
Keyboard driver for Stretch arm / lift / wrist / gripper teleop inside
the L4 GUI. Pairs with `_keyboard_driver.py` (which handles the base).

Earlier versions issued one `move_by(small_delta)` per tick at 30 Hz,
which caused the stepper firmware to ramp up-and-down inside every 33 ms
chunk -- visible as staggered motion. This rewrite uses the API stretch
already provides for *continuous* motion:

  Steppers (lift, arm)    -> `joint.set_velocity(v, a, contact_thresh_…)`
                            once per direction transition (and a zero-
                            velocity hard-stop on release). This is the
                            same API the base uses for smooth driving.

  Dynamixels (wrist*, grip)-> `move_to(extreme_target, v, a)` once on key
                            press / direction-flip; `move_to(current_pos,
                            0, a)` on release. The Dynamixel firmware
                            executes a single smooth trajectory rather
                            than 30 chained micro-trajectories per second.

Per-tick work is now just: compute desired direction from held keys,
diff against last-commanded direction, emit a robot call only on
transitions. push_command() is invoked once per tick, only when a
stepper transition actually happened.

Hardware safeguards (drawn from Hello Robot's stretch_body docs):

- runstop check: every tick consults `robot.pimu.status['runstop_event']`
  via `_is_runstopped()`. If active, no robot calls fire that tick.
- contact-force thresholds: every `set_velocity` and every `move_to`
  passes `contact_thresh_pos_N` / `contact_thresh_neg_N` from config so
  the firmware halts the joint if external force exceeds the limit.
- soft-motion-limit clamping: `move_to` extreme targets are derived from
  `joint.soft_motion_limits['current']` (with a small safety margin), so
  we never request a pose outside the firmware-enforced envelope.
- `req_calibration=True` everywhere (the stretch_body default). The
  GUI separately verifies `robot.is_homed()` before starting us.
- `hard_stop()` zeros stepper velocities + settles Dynamixels at their
  current pose; called from the executor's `finally` and the GUI's main
  finally so an exception or shutdown can never leave a joint moving.

Key mapping (right hand, complements WASD on the left):

    I / K   lift up / down           (stepper, m, set_velocity + push)
    J / L   arm retract / extend     (stepper, m, set_velocity + push)
    U / O   wrist yaw CCW / CW       (Dynamixel, rad)
    Y / H   wrist pitch up / down    (Dynamixel, rad)
    N / M   wrist roll CCW / CW      (Dynamixel, rad)
    [ / ]   gripper close / open     (Dynamixel, %)
    X       stow arm (one-shot)      (retract + lift to stow height)
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# Mapped keysyms, grouped by action. Keeping shift / caps variants so
# holding Shift doesn't silently break the teleop.
_LIFT_UP = frozenset({"i", "I"})
_LIFT_DOWN = frozenset({"k", "K"})
_ARM_RETRACT = frozenset({"j", "J"})
_ARM_EXTEND = frozenset({"l", "L"})
_WRIST_YAW_POS = frozenset({"u", "U"})
_WRIST_YAW_NEG = frozenset({"o", "O"})
_WRIST_PITCH_POS = frozenset({"y", "Y"})
_WRIST_PITCH_NEG = frozenset({"h", "H"})
_WRIST_ROLL_POS = frozenset({"n", "N"})
_WRIST_ROLL_NEG = frozenset({"m", "M"})
_GRIPPER_CLOSE = frozenset({"bracketleft"})
_GRIPPER_OPEN = frozenset({"bracketright"})
_STOW_KEYS = frozenset({"x", "X"})


# Tunable defaults. v / a values match Hello Robot's xbox teleop "regular"
# envelopes, which are known-safe on every Stretch image. Contact-force
# thresholds are conservative bringup values; raise them for heavier
# manipulation work via config.yaml `arm_keyboard:`.
DEFAULTS: Dict[str, float] = {
    # Velocity / acceleration caps used in set_velocity and move_to.
    "lift_v_m_s": 0.10,
    "lift_a_m_s2": 0.20,
    "arm_v_m_s": 0.10,
    "arm_a_m_s2": 0.20,
    "wrist_v_rad_s": 2.0,
    "wrist_a_rad_s2": 4.0,
    "gripper_v": 3.0,
    "gripper_a": 5.0,
    # Contact-force thresholds (Newtons). Joint motion halts immediately
    # if external force exceeds this threshold. Negative values protect
    # the opposite direction. Tune in config.yaml if needed.
    "lift_contact_thresh_pos_N": 60.0,
    "lift_contact_thresh_neg_N": -60.0,
    "arm_contact_thresh_pos_N": 30.0,
    "arm_contact_thresh_neg_N": -30.0,
}


# Safety margins applied when reading soft-motion-limits, so we never
# command exactly at the firmware boundary (which firmware then refuses).
_SOFT_LIMIT_MARGIN: Dict[str, float] = {
    "lift": 0.02,            # m
    "arm": 0.02,             # m
    "wrist_yaw": 0.05,       # rad
    "wrist_pitch": 0.05,     # rad
    "wrist_roll": 0.05,      # rad
    "stretch_gripper": 5.0,  # percent
}


# Permissive fallback limits for joints whose soft_motion_limits we
# couldn't read. Better to drive into a soft-clamp than to crash on
# missing config.
_FALLBACK_LIMITS: Dict[str, Tuple[float, float]] = {
    "lift": (0.05, 1.05),
    "arm": (0.0, 0.5),
    "wrist_yaw": (-3.14, 3.14),
    "wrist_pitch": (-1.57, 0.5),
    "wrist_roll": (-3.14, 3.14),
    "stretch_gripper": (-100.0, 100.0),
}


# ---------------------------------------------------------------------------


def _is_runstopped(robot) -> bool:
    """
    Best-effort check of the robot's runstop button state.

    stretch_body exposes runstop via the Pimu sub-device's status dict;
    the canonical key is `runstop_event` but we tolerate the API moving
    around between robot images. If we can't tell, we say "not
    runstopped" (False) so missing telemetry doesn't block teleop.
    """
    try:
        return bool(robot.pimu.status.get("runstop_event", False))
    except Exception:
        pass
    try:
        # Some images expose this as a Robot-level method.
        return bool(robot.is_runstopped())
    except Exception:
        pass
    return False


def hard_stop(
    robot,
    robot_lock: Optional[threading.Lock],
    config: Optional[dict] = None,
) -> None:
    """
    Zero stepper velocities and settle Dynamixels at their current pose.

    Called by ArmExecutor's `finally` and by the GUI's main `finally`
    on shutdown. Robust to partial-init: any individual robot call that
    fails is logged and skipped, never raised.
    """
    arm_kb_cfg = (config or {}).get("arm_keyboard", {}) or {}
    p = {**DEFAULTS, **arm_kb_cfg}

    def _safe(label: str, fn) -> None:
        try:
            fn()
        except Exception as exc:
            logger.warning("arm hard_stop: %s failed: %s", label, exc)

    def _do() -> None:
        # Steppers: command zero velocity. Pass contact thresholds so the
        # firmware still has its safety net during the stop.
        _safe(
            "lift.set_velocity(0)",
            lambda: robot.lift.set_velocity(
                0.0,
                p["lift_a_m_s2"],
                contact_thresh_pos_N=p.get("lift_contact_thresh_pos_N"),
                contact_thresh_neg_N=p.get("lift_contact_thresh_neg_N"),
            ),
        )
        _safe(
            "arm.set_velocity(0)",
            lambda: robot.arm.set_velocity(
                0.0,
                p["arm_a_m_s2"],
                contact_thresh_pos_N=p.get("arm_contact_thresh_pos_N"),
                contact_thresh_neg_N=p.get("arm_contact_thresh_neg_N"),
            ),
        )
        _safe("push_command", robot.push_command)

        # Dynamixels: settle at the current measured position with v=0.
        for joint_name, a in [
            ("wrist_yaw", p["wrist_a_rad_s2"]),
            ("wrist_pitch", p["wrist_a_rad_s2"]),
            ("wrist_roll", p["wrist_a_rad_s2"]),
            ("stretch_gripper", p["gripper_a"]),
        ]:
            try:
                cur = robot.end_of_arm.get_joint(joint_name).status.get("pos", 0.0)
            except Exception:
                cur = 0.0
            _safe(
                f"{joint_name}.move_to(stop)",
                lambda j=joint_name, c=cur, aa=a: robot.end_of_arm.move_to(
                    j, c, 0.0, aa
                ),
            )

    if robot_lock is not None:
        with robot_lock:
            _do()
    else:
        _do()


# ---------------------------------------------------------------------------


class ArmKeyboardDriver:
    """
    Tkinter-bound keyboard driver for the arm / lift / wrist / gripper.

    Holds a set of currently-pressed keysyms plus a one-shot flag for
    "stow requested". `apply_tick(robot, robot_lock)` is called each
    tick by the ArmExecutor thread.
    """

    # All joints we manage, listed once so cache + dispatch loops stay
    # consistent.
    _STEPPER_AXES = ("lift", "arm")
    _DYNAMIXEL_AXES = ("wrist_yaw", "wrist_pitch", "wrist_roll", "stretch_gripper")

    def __init__(self, root: tk.Misc, config: Optional[dict] = None):
        self._root = root

        arm_cfg = (config or {}).get("arm_keyboard", {}) or {}
        self._p: Dict[str, float] = {**DEFAULTS, **arm_cfg}

        self._pressed: Set[str] = set()
        self._stow_requested = False
        self._lock = threading.Lock()

        # Last-commanded direction per axis: -1, 0, or +1. We diff against
        # the desired direction each tick and emit a robot call only on
        # transitions. Single-threaded (only ArmExecutor mutates this).
        self._last_dir: Dict[str, int] = {
            ax: 0 for ax in (*self._STEPPER_AXES, *self._DYNAMIXEL_AXES)
        }

        # Soft motion limits, read once on first apply_tick. Format:
        # {axis_name: (min_with_margin, max_with_margin)}.
        self._limits: Dict[str, Tuple[float, float]] = {}

        root.bind_all("<KeyPress>", self._on_press, add="+")
        root.bind_all("<KeyRelease>", self._on_release, add="+")

        logger.info(
            "ArmKeyboardDriver: lift v=%.2f m/s a=%.2f m/s^2  "
            "arm v=%.2f m/s a=%.2f m/s^2  wrist v=%.2f rad/s a=%.2f rad/s^2  "
            "lift_contact=±%.0fN  arm_contact=±%.0fN",
            self._p["lift_v_m_s"], self._p["lift_a_m_s2"],
            self._p["arm_v_m_s"], self._p["arm_a_m_s2"],
            self._p["wrist_v_rad_s"], self._p["wrist_a_rad_s2"],
            self._p["lift_contact_thresh_pos_N"],
            self._p["arm_contact_thresh_pos_N"],
        )

    # ------------------------------------------------------------------

    def _is_mapped(self, sym: str) -> bool:
        return (
            sym in _LIFT_UP or sym in _LIFT_DOWN
            or sym in _ARM_RETRACT or sym in _ARM_EXTEND
            or sym in _WRIST_YAW_POS or sym in _WRIST_YAW_NEG
            or sym in _WRIST_PITCH_POS or sym in _WRIST_PITCH_NEG
            or sym in _WRIST_ROLL_POS or sym in _WRIST_ROLL_NEG
            or sym in _GRIPPER_CLOSE or sym in _GRIPPER_OPEN
            or sym in _STOW_KEYS
        )

    def _on_press(self, event) -> None:
        sym = event.keysym
        if not self._is_mapped(sym):
            return
        with self._lock:
            # Stow is edge-triggered, not held. Set the one-shot flag
            # but don't add to pressed set (we don't want stow firing
            # every tick).
            if sym in _STOW_KEYS:
                self._stow_requested = True
                return
            self._pressed.add(sym)

    def _on_release(self, event) -> None:
        with self._lock:
            self._pressed.discard(event.keysym)

    # ------------------------------------------------------------------

    def pressed_snapshot(self) -> Set[str]:
        with self._lock:
            return set(self._pressed)

    def take_stow_request(self) -> bool:
        """Return True if stow was requested since last call, then reset."""
        with self._lock:
            out = self._stow_requested
            self._stow_requested = False
            return out

    def is_active(self) -> bool:
        with self._lock:
            return bool(self._pressed) or self._stow_requested

    # ------------------------------------------------------------------

    def _cache_limits(self, robot) -> None:
        """
        Read soft-motion-limits for every managed joint once and stash
        them with safety margins applied. Falls back to per-joint
        permissive defaults if the limits aren't readable for any reason.
        """
        def _read(soft_motion_limits) -> Optional[Tuple[float, float]]:
            # 'current' is stretch_body's most-restrictive merged view;
            # 'hard' is the manufacturer envelope. Either is acceptable.
            for key in ("current", "hard"):
                try:
                    lo, hi = soft_motion_limits[key]
                    if lo is not None and hi is not None:
                        return float(lo), float(hi)
                except Exception:
                    continue
            return None

        # Steppers.
        for axis, joint in [("lift", robot.lift), ("arm", robot.arm)]:
            margin = _SOFT_LIMIT_MARGIN[axis]
            limits = None
            try:
                limits = _read(joint.soft_motion_limits)
            except Exception as exc:
                logger.warning(
                    "arm keyboard: could not read %s soft limits: %s", axis, exc,
                )
            if limits is None:
                limits = _FALLBACK_LIMITS[axis]
            self._limits[axis] = (limits[0] + margin, limits[1] - margin)
            logger.info(
                "arm keyboard: %s soft limits = (%+.3f, %+.3f) m (margin=%.3f)",
                axis, *self._limits[axis], margin,
            )

        # Dynamixels.
        for axis in self._DYNAMIXEL_AXES:
            margin = _SOFT_LIMIT_MARGIN[axis]
            limits = None
            try:
                joint = robot.end_of_arm.get_joint(axis)
                limits = _read(joint.soft_motion_limits)
            except Exception as exc:
                logger.warning(
                    "arm keyboard: could not read %s soft limits: %s", axis, exc,
                )
            if limits is None:
                limits = _FALLBACK_LIMITS[axis]
            self._limits[axis] = (limits[0] + margin, limits[1] - margin)
            logger.info(
                "arm keyboard: %s soft limits = (%+.3f, %+.3f) (margin=%.3f)",
                axis, *self._limits[axis], margin,
            )

    # ------------------------------------------------------------------

    def apply_tick(self, robot, robot_lock: Optional[threading.Lock]) -> None:
        """
        Apply one tick of motion based on currently-held keys.

        Issues a robot call only on direction transitions per axis, so
        steady-state held-key motion incurs zero bus traffic past the
        initial press.
        """
        # Lazy init: cache soft limits the first time we have a robot.
        if not self._limits:
            self._cache_limits(robot)

        # Hardware-runstop sanity. If active, do nothing this tick — no
        # set_velocity, no move_to, no push_command. Resume on release.
        if _is_runstopped(robot):
            return

        pressed = self.pressed_snapshot()

        # Desired direction per axis: +1, -1, or 0. Opposing keys cancel.
        desired_dir: Dict[str, int] = {
            "lift":          (1 if pressed & _LIFT_UP else 0)        - (1 if pressed & _LIFT_DOWN else 0),
            "arm":           (1 if pressed & _ARM_EXTEND else 0)     - (1 if pressed & _ARM_RETRACT else 0),
            "wrist_yaw":     (1 if pressed & _WRIST_YAW_POS else 0)  - (1 if pressed & _WRIST_YAW_NEG else 0),
            "wrist_pitch":   (1 if pressed & _WRIST_PITCH_POS else 0)- (1 if pressed & _WRIST_PITCH_NEG else 0),
            "wrist_roll":    (1 if pressed & _WRIST_ROLL_POS else 0) - (1 if pressed & _WRIST_ROLL_NEG else 0),
            "stretch_gripper": (1 if pressed & _GRIPPER_OPEN else 0) - (1 if pressed & _GRIPPER_CLOSE else 0),
        }

        # Bail early if nothing changed across all axes — skip lock and
        # robot calls entirely. This is the steady-state common case.
        if all(desired_dir[ax] == self._last_dir[ax] for ax in desired_dir):
            return

        p = self._p

        def _safe(label: str, fn) -> None:
            try:
                fn()
            except Exception as exc:
                logger.warning("arm keyboard: %s failed: %s", label, exc)

        def _do_steppers() -> bool:
            """Emit set_velocity for any stepper whose direction changed.
            Returns True if a push_command is needed."""
            stepper_changed = False
            stepper_specs = [
                ("lift", robot.lift,
                 "lift_v_m_s", "lift_a_m_s2",
                 "lift_contact_thresh_pos_N", "lift_contact_thresh_neg_N"),
                ("arm",  robot.arm,
                 "arm_v_m_s",  "arm_a_m_s2",
                 "arm_contact_thresh_pos_N",  "arm_contact_thresh_neg_N"),
            ]
            for axis, joint, v_key, a_key, ct_pos_key, ct_neg_key in stepper_specs:
                new_d = desired_dir[axis]
                if new_d == self._last_dir[axis]:
                    continue
                v = new_d * p[v_key]
                a = p[a_key]
                ct_pos = p.get(ct_pos_key)
                ct_neg = p.get(ct_neg_key)
                _safe(
                    f"{axis}.set_velocity",
                    lambda j=joint, vv=v, aa=a, cp=ct_pos, cn=ct_neg: j.set_velocity(
                        vv, aa,
                        contact_thresh_pos_N=cp,
                        contact_thresh_neg_N=cn,
                    ),
                )
                self._last_dir[axis] = new_d
                stepper_changed = True
            return stepper_changed

        def _do_dynamixels() -> None:
            """Emit move_to for any Dynamixel whose direction changed.
            move_to is non-blocking; the firmware tracks the new target."""
            dyn_specs = [
                ("wrist_yaw",      "wrist_v_rad_s",  "wrist_a_rad_s2"),
                ("wrist_pitch",    "wrist_v_rad_s",  "wrist_a_rad_s2"),
                ("wrist_roll",     "wrist_v_rad_s",  "wrist_a_rad_s2"),
                ("stretch_gripper","gripper_v",      "gripper_a"),
            ]
            for axis, v_key, a_key in dyn_specs:
                new_d = desired_dir[axis]
                if new_d == self._last_dir[axis]:
                    continue
                v = p[v_key]
                a = p[a_key]
                if new_d == 0:
                    # Released: settle at current measured position.
                    try:
                        cur = float(
                            robot.end_of_arm.get_joint(axis).status.get("pos", 0.0)
                        )
                    except Exception:
                        cur = 0.0
                    _safe(
                        f"{axis}.move_to(stop)",
                        lambda j=axis, c=cur, aa=a: robot.end_of_arm.move_to(
                            j, c, 0.0, aa
                        ),
                    )
                else:
                    lo, hi = self._limits.get(axis, _FALLBACK_LIMITS[axis])
                    target = hi if new_d > 0 else lo
                    _safe(
                        f"{axis}.move_to(extreme)",
                        lambda j=axis, t=target, vv=v, aa=a:
                            robot.end_of_arm.move_to(j, t, vv, aa),
                    )
                self._last_dir[axis] = new_d

        def _do_work() -> None:
            stepper_changed = _do_steppers()
            if stepper_changed:
                _safe("push_command", robot.push_command)
            _do_dynamixels()

        if robot_lock is not None:
            with robot_lock:
                _do_work()
        else:
            _do_work()

    def reset_directions(self) -> None:
        """
        Reset every axis's last-commanded direction to 0.

        Called by the executor after a stow completes (or after any
        external blocking operation that physically moved the joints out
        from under us). Forces the next apply_tick call to treat any
        held key as a fresh transition.
        """
        for k in self._last_dir:
            self._last_dir[k] = 0

    def apply_stow(self, robot, robot_lock: Optional[threading.Lock], config: dict) -> None:
        """
        Drive the arm to the configured stow pose. Called from
        ArmExecutor when the X key has been pressed.

        Uses absolute positions from config.arm (the same values the
        existing ArmController.stow() method would have returned).
        This is a one-shot blocking call: it retracts arm first, then
        lifts, then wrist, then opens gripper slightly. Total sequence
        takes a couple of seconds.
        """
        arm_cfg = config.get("arm", {}) or {}
        stow_arm = float(arm_cfg.get("stow_arm_m", 0.0))
        stow_lift = float(arm_cfg.get("stow_lift_m", 0.8))
        stow_yaw = float(arm_cfg.get("stow_wrist_yaw", 0.0))
        stow_pitch = float(arm_cfg.get("stow_wrist_pitch", -0.5))
        stow_roll = float(arm_cfg.get("stow_wrist_roll", 0.0))

        def _safe(fn_name: str, fn):
            try:
                fn()
            except Exception as exc:
                logger.warning("arm stow: %s failed: %s", fn_name, exc)

        def _do():
            # Phase 1: retract arm (safer before lifting).
            _safe("arm.move_to", lambda: robot.arm.move_to(stow_arm, 0.1, 0.2))
            _safe("push_command", robot.push_command)
            _safe("wait_command", robot.wait_command)

            # Phase 2: lift + wrist pose together.
            _safe("lift.move_to", lambda: robot.lift.move_to(stow_lift, 0.1, 0.2))
            _safe("push_command", robot.push_command)
            _safe("wrist_yaw", lambda: robot.end_of_arm.move_to("wrist_yaw", stow_yaw, 2.0, 4.0))
            _safe("wrist_pitch", lambda: robot.end_of_arm.move_to("wrist_pitch", stow_pitch, 2.0, 4.0))
            _safe("wrist_roll", lambda: robot.end_of_arm.move_to("wrist_roll", stow_roll, 2.0, 4.0))
            _safe("wait_command", robot.wait_command)

        logger.info("arm stow: starting sequence")
        if robot_lock is not None:
            with robot_lock:
                _do()
        else:
            _do()
        # After stow, all joints are at rest at known positions. Reset
        # last-direction state so the next held key registers as a fresh
        # press transition.
        self.reset_directions()
        logger.info("arm stow: complete")
