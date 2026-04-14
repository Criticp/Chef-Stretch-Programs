"""
Gamepad module — Pygame-based gamepad input with AUTO/MANUAL mode toggle.

Runs in its own thread. Reads axes and buttons from an Xbox-style controller.
Provides thread-safe access to the current gamepad state.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GamepadState:
    """Thread-safe container for the current gamepad readings."""
    # Axes (after deadzone filtering).
    left_x: float = 0.0       # Left stick horizontal: negative=left, positive=right
    left_y: float = 0.0       # Left stick vertical:   negative=up,   positive=down
    right_x: float = 0.0      # Right stick horizontal
    right_y: float = 0.0      # Right stick vertical

    # Edge-detected button presses (True for one poll cycle, then cleared).
    toggle_pressed: bool = False     # Start button — toggle AUTO/MANUAL
    arm_pressed: bool = False        # A button — trigger arm positioning
    stow_pressed: bool = False       # B button — stow arm
    search_pressed: bool = False     # Y button — start search from MANUAL
    stop_pressed: bool = False       # Back button — emergency stop

    # Current mode.
    mode: str = "AUTO"               # "AUTO" or "MANUAL"

    connected: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_copy(self) -> "GamepadState":
        """Return a snapshot of the current state (without the lock)."""
        with self.lock:
            return GamepadState(
                left_x=self.left_x,
                left_y=self.left_y,
                right_x=self.right_x,
                right_y=self.right_y,
                toggle_pressed=self.toggle_pressed,
                arm_pressed=self.arm_pressed,
                stow_pressed=self.stow_pressed,
                search_pressed=self.search_pressed,
                stop_pressed=self.stop_pressed,
                mode=self.mode,
                connected=self.connected,
            )

    def clear_buttons(self) -> None:
        """Clear edge-detected button presses after they've been consumed."""
        with self.lock:
            self.toggle_pressed = False
            self.arm_pressed = False
            self.stow_pressed = False
            self.search_pressed = False
            self.stop_pressed = False


class GamepadThread(threading.Thread):
    """
    Background thread that polls a gamepad via Pygame.

    Pygame is initialized inside the thread because its event handling
    is not fully thread-safe when initialized from another thread.
    """

    def __init__(self, config: dict, state: GamepadState):
        super().__init__(daemon=True, name="GamepadThread")
        gp_cfg = config["gamepad"]
        self._toggle_btn = gp_cfg["toggle_button"]
        self._arm_btn = gp_cfg["arm_button"]
        self._stow_btn = gp_cfg["stow_button"]
        self._search_btn = gp_cfg["search_button"]
        self._stop_btn = gp_cfg["stop_button"]
        self._deadzone = gp_cfg["deadzone"]

        self._state = state
        self._stop_event = threading.Event()

        # Track previous button states for edge detection.
        self._prev_buttons: dict = {}

    def stop(self) -> None:
        self._stop_event.set()

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self._deadzone:
            return 0.0
        return value

    def _button_edge(self, joystick, btn_id: int) -> bool:
        """Return True only on the rising edge (pressed this frame, not last)."""
        try:
            current = joystick.get_button(btn_id)
        except Exception:
            return False
        prev = self._prev_buttons.get(btn_id, 0)
        self._prev_buttons[btn_id] = current
        return current == 1 and prev == 0

    def run(self) -> None:
        # Import pygame here so it's initialized in this thread.
        try:
            import pygame
        except ImportError:
            logger.error("pygame not installed — gamepad support disabled")
            return

        pygame.init()
        pygame.joystick.init()

        joystick: Optional[object] = None

        logger.info("GamepadThread started — waiting for controller...")

        while not self._stop_event.is_set():
            try:
                pygame.event.pump()
            except Exception:
                time.sleep(0.1)
                continue

            # Try to connect to a joystick if we don't have one.
            if joystick is None:
                if pygame.joystick.get_count() > 0:
                    joystick = pygame.joystick.Joystick(0)
                    joystick.init()
                    with self._state.lock:
                        self._state.connected = True
                    logger.info("Gamepad connected: %s", joystick.get_name())
                else:
                    time.sleep(0.5)
                    continue

            # Read axes.
            try:
                lx = self._apply_deadzone(joystick.get_axis(0))
                ly = self._apply_deadzone(joystick.get_axis(1))
                rx = self._apply_deadzone(joystick.get_axis(3))
                ry = self._apply_deadzone(joystick.get_axis(4))
            except Exception:
                lx = ly = rx = ry = 0.0

            # Edge-detect buttons.
            toggle = self._button_edge(joystick, self._toggle_btn)
            arm = self._button_edge(joystick, self._arm_btn)
            stow = self._button_edge(joystick, self._stow_btn)
            search = self._button_edge(joystick, self._search_btn)
            stop = self._button_edge(joystick, self._stop_btn)

            # Update shared state.
            with self._state.lock:
                self._state.left_x = lx
                self._state.left_y = ly
                self._state.right_x = rx
                self._state.right_y = ry

                if toggle:
                    # Toggle mode.
                    self._state.mode = (
                        "MANUAL" if self._state.mode == "AUTO" else "AUTO"
                    )
                    self._state.toggle_pressed = True
                    logger.info("Gamepad mode toggled to %s", self._state.mode)

                if arm:
                    self._state.arm_pressed = True
                if stow:
                    self._state.stow_pressed = True
                if search:
                    self._state.search_pressed = True
                if stop:
                    self._state.stop_pressed = True

            time.sleep(0.02)  # ~50 Hz

        # Cleanup.
        try:
            import pygame
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
        logger.info("GamepadThread stopped")
