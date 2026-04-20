"""
Keyboard driver for Stretch arm / lift / wrist / gripper teleop inside
the L4 GUI. Pairs with `_keyboard_driver.py` (which handles the base).

Mirrors Hello Robot's own `stretch_xbox_controller_teleop.py` approach:
each tick, if a key is held, we issue a small incremental
`move_by(...)` on the corresponding joint. stretch_body smooths the
trajectory internally. One `push_command()` per tick flushes stepper
(arm / lift) moves; wrist and gripper are Dynamixel-backed so they
don't need a push.

Key mapping (right hand, complements WASD on the left):

    I / K   lift up / down           (stepper, m, push needed)
    J / L   arm retract / extend     (stepper, m, push needed)
    U / O   wrist yaw CCW / CW       (Dynamixel, rad, no push)
    Y / H   wrist pitch up / down    (Dynamixel, rad, no push)
    N / M   wrist roll CCW / CW      (Dynamixel, rad, no push)
    [ / ]   gripper close / open     (Dynamixel, %, no push)
    X       stow arm (one-shot)      (retract + lift to stow height)

Held = keep moving; released = stop. Opposing keys cancel by not
queuing any move. A stow request fires once per X-keypress.

Per-tick deltas (at 30 Hz):
    lift / arm           0.005 m/tick  -> ~0.15 m/s
    wrist joints         0.03 rad/tick -> ~0.9 rad/s
    gripper              10 pct/tick   -> full open/close in ~30 ticks

These are conservative starting values; tune via config later if needed.
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


# Per-tick step sizes. These are "how much the joint moves per tick at
# 30 Hz when the key is held". Bigger = snappier but less fine control.
DEFAULTS = {
    "lift_m_per_tick": 0.005,
    "arm_m_per_tick": 0.005,
    "wrist_rad_per_tick": 0.03,
    "gripper_pct_per_tick": 10.0,
    # Velocity / acceleration caps passed to move_by. Conservative.
    "lift_v_m_s": 0.12,
    "lift_a_m_s2": 0.20,
    "arm_v_m_s": 0.12,
    "arm_a_m_s2": 0.20,
    "wrist_v_rad_s": 2.0,
    "wrist_a_rad_s2": 4.0,
    "gripper_v": 3.0,
    "gripper_a": 5.0,
}


class ArmKeyboardDriver:
    """
    Tkinter-bound keyboard driver for the arm / lift / wrist / gripper.

    Holds a set of currently-pressed keysyms plus a one-shot flag for
    "stow requested". `apply_tick(robot, robot_lock)` is called each
    tick by the ArmExecutor thread.
    """

    def __init__(self, root: tk.Misc, config: Optional[dict] = None):
        self._root = root

        # config.arm.keyboard could override step sizes later; for now
        # pick up defaults and allow per-key override if anyone adds them.
        arm_cfg = (config or {}).get("arm_keyboard", {}) or {}
        self._p: Dict[str, float] = {**DEFAULTS, **arm_cfg}

        self._pressed: Set[str] = set()
        self._stow_requested = False
        self._lock = threading.Lock()

        root.bind_all("<KeyPress>", self._on_press, add="+")
        root.bind_all("<KeyRelease>", self._on_release, add="+")

        logger.info(
            "ArmKeyboardDriver: lift=%.3f m/tick  arm=%.3f m/tick  "
            "wrist=%.3f rad/tick  gripper=%.1f pct/tick",
            self._p["lift_m_per_tick"], self._p["arm_m_per_tick"],
            self._p["wrist_rad_per_tick"], self._p["gripper_pct_per_tick"],
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

    def apply_tick(self, robot, robot_lock: Optional[threading.Lock]) -> None:
        """
        Apply one tick of motion based on currently-held keys.

        Caller is responsible for rate (ArmExecutor calls at 30 Hz).
        Robot method calls happen under `robot_lock` if provided.
        """
        pressed = self.pressed_snapshot()
        if not pressed and not self.take_stow_request():
            return  # nothing held, nothing to do

        p = self._p

        # Resolve axis intentions: +1, -1, or 0.
        lift_dir = (1 if pressed & _LIFT_UP else 0) - (1 if pressed & _LIFT_DOWN else 0)
        arm_dir = (1 if pressed & _ARM_EXTEND else 0) - (1 if pressed & _ARM_RETRACT else 0)
        yaw_dir = (1 if pressed & _WRIST_YAW_POS else 0) - (1 if pressed & _WRIST_YAW_NEG else 0)
        pitch_dir = (1 if pressed & _WRIST_PITCH_POS else 0) - (1 if pressed & _WRIST_PITCH_NEG else 0)
        roll_dir = (1 if pressed & _WRIST_ROLL_POS else 0) - (1 if pressed & _WRIST_ROLL_NEG else 0)
        grip_dir = (1 if pressed & _GRIPPER_OPEN else 0) - (1 if pressed & _GRIPPER_CLOSE else 0)

        def _safe(fn_name: str, fn):
            try:
                fn()
            except Exception as exc:
                logger.warning("arm keyboard: %s failed: %s", fn_name, exc)

        def _do_work():
            pushed_needed = False

            # Stepper joints (lift + arm) -- need push_command at end.
            if lift_dir != 0:
                _safe("lift.move_by", lambda: robot.lift.move_by(
                    lift_dir * p["lift_m_per_tick"],
                    p["lift_v_m_s"],
                    p["lift_a_m_s2"],
                ))
                pushed_needed = True

            if arm_dir != 0:
                _safe("arm.move_by", lambda: robot.arm.move_by(
                    arm_dir * p["arm_m_per_tick"],
                    p["arm_v_m_s"],
                    p["arm_a_m_s2"],
                ))
                pushed_needed = True

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

            if pushed_needed:
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
