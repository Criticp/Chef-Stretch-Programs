"""
Keyboard driver for Stretch arm / lift / wrist / gripper teleop inside
the L4 GUI. Pairs with `_keyboard_driver.py` (which handles the base).

Layering note: this module is the *teleop input + hardware dispatch*
layer. It deliberately does NOT replace
`stretch_fruit_finder.fruit_finder.arm_controller.ArmController`, which
is the *pose-math* layer that returns `MotorCommand` lists for things
like "position the gripper above this 3D point" or "stow". That math
layer is queued for use by L5b (the auto-position-arm-above-detected-
orange button); the two layers will compose, not duplicate.

Control strategy: each tick, for every held direction key, we issue a
small `move_by(delta, v, a)` and one `push_command()` for the steppers.
This matches Hello Robot's own `stretch_xbox_controller_teleop.py` —
their reference, known-good pattern. An earlier rewrite tried to use
`set_velocity()` with one command per direction transition, which left
the steppers idle between ticks and made the arm unresponsive on this
robot. Reverted; per-tick refresh is the standard and works.

Tuning relative to the original L5a code (which felt staggered):
- Larger per-tick delta (0.02 m vs 0.005 m) so the stepper has runway
  to build up speed before each chunk decelerates.
- Bigger v / a caps (matching Hello Robot's "regular mode" envelope)
  so successive moves blend rather than stop-and-start.

Hardware safeguards drawn from Hello Robot's stretch_body docs:

- runstop check: every tick consults `robot.pimu.status['runstop_event']`
  via `_is_runstopped()`. If active, no robot calls fire that tick.
- contact-force thresholds: optional, off by default. When set in
  config.yaml `arm_keyboard:`, every move_by passes
  `contact_thresh_pos_N` / `contact_thresh_neg_N` so the firmware
  halts the joint if external force exceeds the limit. Defaults to
  None (use stretch_body's tuned per-joint defaults) so we don't
  spuriously trip on normal gravity load.
- soft-motion-limit clamping: stretch_body's move_by clamps internally
  against `soft_motion_limits['current']`, so we don't have to.
- `req_calibration=True` everywhere (the stretch_body default). The
  GUI separately verifies `robot.is_homed()` before starting us.
- `hard_stop()` zeros stepper velocities and settles Dynamixels at
  their current pose; called from the executor's `finally` and the
  GUI's main `finally` so an exception or shutdown can never leave a
  joint moving.

Key mapping (right hand, complements WASD on the left):

    I / K   lift up / down           (stepper, m, push needed)
    J / L   arm retract / extend     (stepper, m, push needed)
    U / O   wrist yaw CCW / CW       (Dynamixel, rad, no push)
    Y / H   wrist pitch up / down    (Dynamixel, rad, no push)
    N / M   wrist roll CCW / CW      (Dynamixel, rad, no push)
    [ / ]   gripper close / open     (Dynamixel, %, no push)
    X       stow arm (one-shot)      (retract + lift to stow height)
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from typing import Dict, Optional, Set

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


# Tunable defaults. Per-tick deltas + v/a caps tuned for smooth motion
# on a 30 Hz tick. Contact thresholds default to None (firmware uses
# its own tuned per-joint defaults); set them in config.yaml
# arm_keyboard: only if you need extra-conservative force limits.
DEFAULTS: Dict[str, Optional[float]] = {
    # Per-tick movement deltas. Bigger = snappier, smoother (fewer
    # accel/decel transitions) but less fine-grained control.
    "lift_m_per_tick": 0.02,
    "arm_m_per_tick": 0.02,
    "wrist_rad_per_tick": 0.06,
    "gripper_pct_per_tick": 10.0,
    # Velocity / acceleration caps passed to move_by. Match Hello
    # Robot's regular-mode envelope from stretch_xbox_controller_teleop.
    "lift_v_m_s": 0.5,
    "lift_a_m_s2": 0.4,
    "arm_v_m_s": 0.5,
    "arm_a_m_s2": 0.4,
    "wrist_v_rad_s": 10.0,
    "wrist_a_rad_s2": 15.0,
    "gripper_v": 10.0,
    "gripper_a": 15.0,
    # Contact-force thresholds (Newtons). Optional; firmware defaults
    # are used when these are None. Only override if you've measured
    # the resting load on a given joint and want a tighter envelope.
    "lift_contact_thresh_pos_N": None,
    "lift_contact_thresh_neg_N": None,
    "arm_contact_thresh_pos_N": None,
    "arm_contact_thresh_neg_N": None,
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


def _kw_thresholds(p: Dict[str, Optional[float]], pos_key: str, neg_key: str) -> Dict[str, float]:
    """
    Build the contact-threshold kwargs for a move_by / set_velocity call.

    Returns an empty dict if both thresholds are None, so we don't
    pass `contact_thresh_pos_N=None` (which some stretch_body builds
    treat as "disable contact protection" rather than "use default").
    """
    kw: Dict[str, float] = {}
    pos = p.get(pos_key)
    neg = p.get(neg_key)
    if pos is not None:
        kw["contact_thresh_pos_N"] = float(pos)
    if neg is not None:
        kw["contact_thresh_neg_N"] = float(neg)
    return kw


def hard_stop(
    robot,
    robot_lock: Optional[threading.Lock],
    config: Optional[dict] = None,
) -> None:
    """
    Bring all teleop-managed joints to rest.

    Called by ArmExecutor's `finally` and by the GUI's main `finally`
    on shutdown. Robust to partial-init: any individual robot call that
    fails is logged and skipped, never raised.

    For steppers we issue a tiny move_by(0, v, a) + push to ensure the
    last command in flight is settled. For Dynamixels we re-issue
    move_to(current_pos, 0, a) so any in-flight motion stops.
    """
    arm_kb_cfg = (config or {}).get("arm_keyboard", {}) or {}
    p: Dict[str, Optional[float]] = {**DEFAULTS, **arm_kb_cfg}

    def _safe(label: str, fn) -> None:
        try:
            fn()
        except Exception as exc:
            logger.warning("arm hard_stop: %s failed: %s", label, exc)

    def _do() -> None:
        # Steppers: zero-displacement move_by re-issues the position
        # target as "right where you are", which the firmware treats as
        # "stop". Pass current contact thresholds (or empty kwargs).
        _safe(
            "lift.move_by(0)",
            lambda: robot.lift.move_by(
                0.0, p["lift_v_m_s"], p["lift_a_m_s2"],
                **_kw_thresholds(p, "lift_contact_thresh_pos_N",
                                 "lift_contact_thresh_neg_N"),
            ),
        )
        _safe(
            "arm.move_by(0)",
            lambda: robot.arm.move_by(
                0.0, p["arm_v_m_s"], p["arm_a_m_s2"],
                **_kw_thresholds(p, "arm_contact_thresh_pos_N",
                                 "arm_contact_thresh_neg_N"),
            ),
        )
        _safe("push_command", robot.push_command)

        # Dynamixels: settle at current measured position with v=0.
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

    # Autorepeat fake-release suppression window — see _keyboard_driver.py
    # for the rationale. X11 generates alternating KeyPress/KeyRelease at
    # the autorepeat rate while a key is held; we defer real releases by
    # this many ms and cancel them if a KeyPress for the same key arrives
    # in that window.
    _AUTOREPEAT_DEBOUNCE_MS = 30

    def __init__(self, root: tk.Misc, config: Optional[dict] = None):
        self._root = root

        # config.arm_keyboard overrides defaults.
        arm_cfg = (config or {}).get("arm_keyboard", {}) or {}
        self._p: Dict[str, Optional[float]] = {**DEFAULTS, **arm_cfg}

        self._pressed: Set[str] = set()
        self._pending_releases: Dict[str, str] = {}  # keysym -> after-id
        self._stow_requested = False
        self._lock = threading.Lock()

        root.bind_all("<KeyPress>", self._on_press, add="+")
        root.bind_all("<KeyRelease>", self._on_release, add="+")

        ct_lift = self._p.get("lift_contact_thresh_pos_N")
        ct_arm = self._p.get("arm_contact_thresh_pos_N")
        logger.info(
            "ArmKeyboardDriver: lift=%.3f m/tick (v=%.2f a=%.2f)  "
            "arm=%.3f m/tick (v=%.2f a=%.2f)  wrist=%.3f rad/tick  "
            "gripper=%.1f pct/tick  contact: lift=%s arm=%s",
            self._p["lift_m_per_tick"], self._p["lift_v_m_s"], self._p["lift_a_m_s2"],
            self._p["arm_m_per_tick"], self._p["arm_v_m_s"], self._p["arm_a_m_s2"],
            self._p["wrist_rad_per_tick"], self._p["gripper_pct_per_tick"],
            f"+/-{ct_lift}N" if ct_lift is not None else "firmware default",
            f"+/-{ct_arm}N" if ct_arm is not None else "firmware default",
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
        # Cancel any pending release for the same key — autorepeat fake.
        pending = self._pending_releases.pop(sym, None)
        if pending is not None:
            try:
                self._root.after_cancel(pending)
            except Exception:
                pass
        with self._lock:
            # Stow is edge-triggered, not held. Set the one-shot flag
            # but don't add to pressed set (we don't want stow firing
            # every tick).
            if sym in _STOW_KEYS:
                self._stow_requested = True
                return
            self._pressed.add(sym)

    def _on_release(self, event) -> None:
        sym = event.keysym
        if not self._is_mapped(sym):
            return
        # Stow has no held state to clear; nothing to defer either.
        if sym in _STOW_KEYS:
            return
        existing = self._pending_releases.pop(sym, None)
        if existing is not None:
            try:
                self._root.after_cancel(existing)
            except Exception:
                pass
        self._pending_releases[sym] = self._root.after(
            self._AUTOREPEAT_DEBOUNCE_MS, lambda s=sym: self._real_release(s)
        )

    def _real_release(self, sym: str) -> None:
        self._pending_releases.pop(sym, None)
        with self._lock:
            self._pressed.discard(sym)

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

    def apply_tick(self, robot, robot_lock: Optional[threading.Lock]) -> None:
        """
        Apply one tick of motion based on currently-held keys.

        Caller is responsible for rate (ArmExecutor calls at 30 Hz).
        Robot method calls happen under `robot_lock` if provided.
        """
        # Hardware-runstop sanity. If active, do nothing this tick — no
        # move_by, no push_command. Resume on release.
        if _is_runstopped(robot):
            return

        pressed = self.pressed_snapshot()
        if not pressed:
            return  # nothing held, nothing to do

        p = self._p

        # Resolve axis intentions: +1, -1, or 0.
        lift_dir = (1 if pressed & _LIFT_UP else 0) - (1 if pressed & _LIFT_DOWN else 0)
        arm_dir = (1 if pressed & _ARM_EXTEND else 0) - (1 if pressed & _ARM_RETRACT else 0)
        yaw_dir = (1 if pressed & _WRIST_YAW_POS else 0) - (1 if pressed & _WRIST_YAW_NEG else 0)
        pitch_dir = (1 if pressed & _WRIST_PITCH_POS else 0) - (1 if pressed & _WRIST_PITCH_NEG else 0)
        roll_dir = (1 if pressed & _WRIST_ROLL_POS else 0) - (1 if pressed & _WRIST_ROLL_NEG else 0)
        grip_dir = (1 if pressed & _GRIPPER_OPEN else 0) - (1 if pressed & _GRIPPER_CLOSE else 0)

        # Skip lock entirely if every axis is idle (e.g. only stow key
        # was held — already consumed by the executor before this call).
        if not any((lift_dir, arm_dir, yaw_dir, pitch_dir, roll_dir, grip_dir)):
            return

        def _safe(fn_name: str, fn):
            try:
                fn()
            except Exception as exc:
                logger.warning("arm keyboard: %s failed: %s", fn_name, exc)

        def _do_work():
            push_needed = False

            # Stepper joints (lift + arm) -- need push_command at end.
            if lift_dir != 0:
                _safe("lift.move_by", lambda: robot.lift.move_by(
                    lift_dir * p["lift_m_per_tick"],
                    p["lift_v_m_s"],
                    p["lift_a_m_s2"],
                    **_kw_thresholds(p, "lift_contact_thresh_pos_N",
                                     "lift_contact_thresh_neg_N"),
                ))
                push_needed = True

            if arm_dir != 0:
                _safe("arm.move_by", lambda: robot.arm.move_by(
                    arm_dir * p["arm_m_per_tick"],
                    p["arm_v_m_s"],
                    p["arm_a_m_s2"],
                    **_kw_thresholds(p, "arm_contact_thresh_pos_N",
                                     "arm_contact_thresh_neg_N"),
                ))
                push_needed = True

            # Dynamixel joints (wrist + gripper) -- no push needed.
            if yaw_dir != 0:
                _safe("wrist_yaw.move_by", lambda: robot.end_of_arm.move_by(
                    "wrist_yaw",
                    yaw_dir * p["wrist_rad_per_tick"],
                    p["wrist_v_rad_s"],
                    p["wrist_a_rad_s2"],
                ))
            if pitch_dir != 0:
                _safe("wrist_pitch.move_by", lambda: robot.end_of_arm.move_by(
                    "wrist_pitch",
                    pitch_dir * p["wrist_rad_per_tick"],
                    p["wrist_v_rad_s"],
                    p["wrist_a_rad_s2"],
                ))
            if roll_dir != 0:
                _safe("wrist_roll.move_by", lambda: robot.end_of_arm.move_by(
                    "wrist_roll",
                    roll_dir * p["wrist_rad_per_tick"],
                    p["wrist_v_rad_s"],
                    p["wrist_a_rad_s2"],
                ))
            if grip_dir != 0:
                _safe("gripper.move_by", lambda: robot.end_of_arm.move_by(
                    "stretch_gripper",
                    grip_dir * p["gripper_pct_per_tick"],
                    p["gripper_v"],
                    p["gripper_a"],
                ))

            if push_needed:
                _safe("push_command", robot.push_command)

        if robot_lock is not None:
            with robot_lock:
                _do_work()
        else:
            _do_work()

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
        logger.info("arm stow: complete")
