"""
Gamepad executor — drives the Stretch base from an Xbox-style gamepad.

Pairs with `fruit_finder.gamepad.GamepadThread`, which reads the USB
joystick and publishes state into a shared `GamepadState`. This module
consumes that state and commands the robot base via stretch_body's
velocity API:

    robot.base.set_velocity(v_trans_m_s, v_rot_rad_s)
    robot.push_command()

Stick mapping (matches Hello Robot's own xbox teleop):
    left stick Y  (up/down)   -> base translate   (up = forward)
    left stick X  (left/right)-> base rotate      (left = CCW / left turn)

The head is left untouched — callers run a separate tracker thread that
owns head motion. A `robot_lock` is required so `push_command()` here
doesn't race with head commands elsewhere.

Stop behavior:
- On `stop_event.set()`, the run loop exits after commanding zero
  velocities + push, so the base halts cleanly.
- If the gamepad "stop" button (config `gamepad.stop_button`, default 6 =
  Back) is pressed, we set the stop_event too and let the caller handle
  the rest of the shutdown (re-center head, robot.stop(), etc).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from fruit_finder.gamepad import GamepadState

logger = logging.getLogger(__name__)


class GamepadExecutor(threading.Thread):
    """Thread that turns GamepadState into base velocity commands."""

    def __init__(
        self,
        robot,
        gamepad_state: GamepadState,
        robot_lock: threading.Lock,
        config: dict,
        stop_event: threading.Event,
        rate_hz: float = 30.0,
    ):
        super().__init__(daemon=True, name="GamepadExecutor")

        self.robot = robot
        self.state = gamepad_state
        self.robot_lock = robot_lock
        self.stop_event = stop_event

        gp_cfg = config.get("gamepad", {}) or {}
        self.max_trans_m_s = float(gp_cfg.get("base_translate_speed", 0.1))
        self.max_rot_rad_s = float(gp_cfg.get("base_rotate_speed", 0.15))

        # Button ids come from the same config section as the GamepadThread.
        self.stop_button = int(gp_cfg.get("stop_button", 6))

        self.rate_hz = float(rate_hz)
        self._period = 1.0 / self.rate_hz

        # Track last-sent so we only touch the bus when something changes.
        self._last_v: tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------

    def _command_velocity(self, v_trans: float, v_rot: float) -> None:
        """Send v_trans + v_rot to the base under the shared lock."""
        try:
            with self.robot_lock:
                self.robot.base.set_velocity(v_trans, v_rot)
                self.robot.push_command()
        except Exception as exc:
            logger.warning("gamepad: set_velocity failed: %s", exc)

    def _halt(self) -> None:
        """Ensure the base is stopped. Safe to call multiple times."""
        self._command_velocity(0.0, 0.0)
        self._last_v = (0.0, 0.0)

    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info(
            "GamepadExecutor started: max_trans=%.2f m/s  max_rot=%.2f rad/s  rate=%.0fHz",
            self.max_trans_m_s, self.max_rot_rad_s, self.rate_hz,
        )

        try:
            while not self.stop_event.is_set():
                start = time.time()
                snap = self.state.get_copy()

                # Stop button -> request global shutdown.
                if snap.stop_pressed:
                    logger.warning("gamepad: STOP button pressed; setting stop_event")
                    self.stop_event.set()
                    break

                # Only drive the base when the gamepad is connected; otherwise
                # hold position so a disconnect doesn't leave us running.
                if not snap.connected:
                    if self._last_v != (0.0, 0.0):
                        self._halt()
                    time.sleep(self._period)
                    continue

                # Stick -> velocity mapping.
                # Pygame left_y is positive-DOWN, so negate for forward-positive.
                # Pygame left_x is positive-RIGHT, so negate so stick-left = CCW.
                v_trans = -snap.left_y * self.max_trans_m_s
                v_rot = -snap.left_x * self.max_rot_rad_s

                # Only hit the bus when the command meaningfully changed.
                # Tiny changes get dropped to keep USB traffic low.
                dv_trans = abs(v_trans - self._last_v[0])
                dv_rot = abs(v_rot - self._last_v[1])
                changed = dv_trans > 0.005 or dv_rot > 0.01
                # Always force a zero-velocity push once after stick returns to
                # center, so the base actually comes to a stop.
                returning_to_zero = (
                    (v_trans == 0.0 and v_rot == 0.0)
                    and self._last_v != (0.0, 0.0)
                )

                if changed or returning_to_zero:
                    self._command_velocity(v_trans, v_rot)
                    self._last_v = (v_trans, v_rot)

                # Hold loop rate.
                elapsed = time.time() - start
                if elapsed < self._period:
                    time.sleep(self._period - elapsed)
        finally:
            self._halt()
            logger.info("GamepadExecutor stopped")
