"""
Background tick loop that applies ArmKeyboardDriver intent to the robot.

Separate from GamepadExecutor (which owns the base) so arm and base can
be commanded independently without stepping on each other's command
queue. Uses the same shared robot_lock so `push_command()` on the base
doesn't race with `push_command()` on the arm.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from _arm_keyboard_driver import ArmKeyboardDriver

logger = logging.getLogger(__name__)


class ArmExecutor(threading.Thread):
    def __init__(
        self,
        robot,
        driver: ArmKeyboardDriver,
        robot_lock: threading.Lock,
        config: dict,
        stop_event: threading.Event,
        rate_hz: float = 30.0,
    ):
        super().__init__(daemon=True, name="ArmExecutor")
        self.robot = robot
        self.driver = driver
        self.robot_lock = robot_lock
        self.config = config
        self.stop_event = stop_event
        self.rate_hz = float(rate_hz)
        self._period = 1.0 / self.rate_hz

    def run(self) -> None:
        logger.info("ArmExecutor started at %.0f Hz", self.rate_hz)
        try:
            while not self.stop_event.is_set():
                start = time.time()

                # Stow is one-shot and blocking (it has its own waits
                # internally). Handle it before the per-tick move_by
                # pulse so the two don't interleave.
                if self.driver.take_stow_request():
                    try:
                        self.driver.apply_stow(
                            self.robot, self.robot_lock, self.config
                        )
                    except Exception as exc:
                        logger.warning("arm stow raised: %s", exc)

                # Normal per-tick held-key moves.
                try:
                    self.driver.apply_tick(self.robot, self.robot_lock)
                except Exception as exc:
                    logger.warning("arm tick raised: %s", exc)

                elapsed = time.time() - start
                if elapsed < self._period:
                    time.sleep(self._period - elapsed)
        finally:
            logger.info("ArmExecutor stopped")
