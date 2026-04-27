"""
Keyboard driver for Stretch base teleop from the L4 GUI.

Uses tkinter key events only — no extra libraries, no pygame, no evdev,
no raw device access. Works anywhere tkinter works (local HDMI or
X-forwarded SSH).

Key mapping (matches WASD + arrow keys):

    W or Up     : translate forward
    S or Down   : translate backward
    A or Left   : rotate CCW  (turn left)
    D or Right  : rotate CW   (turn right)

Hold a key to keep moving; release to stop. Multiple keys compose —
W+A drives forward-and-left simultaneously. Per-axis behaviour is
digital (either full-speed or zero), unlike a gamepad stick.

Usage:

    kb = KeyboardDriver(root, config)
    # later, anywhere in the app:
    v_trans, v_rot = kb.velocity()

Max speeds come from the same `gamepad.base_translate_speed` and
`gamepad.base_rotate_speed` config values the GamepadExecutor uses,
so keyboard and gamepad honour the same speed limit.

Safety: the driver has no idea about the robot. It only reports what
the user is asking for. The GamepadExecutor consumes this and commands
the base under the shared robot_lock.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from typing import Tuple

logger = logging.getLogger(__name__)


# Tkinter keysyms are case-sensitive. 'w' and 'W' are different events,
# so we listen for both to handle Caps Lock / Shift being held.
_FORWARD_KEYS = frozenset({"w", "W", "Up"})
_BACKWARD_KEYS = frozenset({"s", "S", "Down"})
_LEFT_KEYS = frozenset({"a", "A", "Left"})
_RIGHT_KEYS = frozenset({"d", "D", "Right"})


class KeyboardDriver:
    """
    Tkinter-key-bound base velocity source.

    Registers KeyPress / KeyRelease handlers on the given tkinter root
    so any widget focus inside the window forwards keystrokes to us.
    Call `velocity()` to read the current (v_trans, v_rot) based on
    which keys are pressed right now.
    """

    # Autorepeat fake-release suppression window. On X11/Linux, holding a
    # key generates alternating KeyPress + KeyRelease events at the OS
    # autorepeat rate (~30 Hz), which would otherwise toggle the held-key
    # set off for ~1 ms each cycle and stutter the motion. We defer real
    # releases by this many ms; if a KeyPress arrives in that window we
    # treat it as autorepeat and cancel the release.
    _AUTOREPEAT_DEBOUNCE_MS = 30

    def __init__(self, root: tk.Misc, config: dict):
        self._root = root
        gp_cfg = config.get("gamepad", {}) or {}
        self.max_trans_m_s = float(gp_cfg.get("base_translate_speed", 0.1))
        self.max_rot_rad_s = float(gp_cfg.get("base_rotate_speed", 0.15))

        self._pressed: set[str] = set()
        self._pending_releases: dict[str, str] = {}  # keysym -> after-id
        self._lock = threading.Lock()

        # `bind_all` so the keys are picked up no matter which tkinter
        # widget currently has keyboard focus, as long as the window
        # itself is focused.
        root.bind_all("<KeyPress>", self._on_press, add="+")
        root.bind_all("<KeyRelease>", self._on_release, add="+")

        logger.info(
            "KeyboardDriver: W/S=translate (%.2f m/s), A/D=rotate (%.2f rad/s); "
            "arrow keys equivalent",
            self.max_trans_m_s, self.max_rot_rad_s,
        )

    # ------------------------------------------------------------------

    def _on_press(self, event) -> None:
        sym = event.keysym
        # Only track the symbols we care about — avoids hoarding random keys.
        if not (
            sym in _FORWARD_KEYS
            or sym in _BACKWARD_KEYS
            or sym in _LEFT_KEYS
            or sym in _RIGHT_KEYS
        ):
            return
        # Cancel any pending release for the same key — that release was
        # an X11 autorepeat fake, not a real key-up.
        pending = self._pending_releases.pop(sym, None)
        if pending is not None:
            try:
                self._root.after_cancel(pending)
            except Exception:
                pass
        with self._lock:
            self._pressed.add(sym)

    def _on_release(self, event) -> None:
        sym = event.keysym
        # Defer the actual clearance. If a KeyPress for the same key
        # arrives in the debounce window, this will be cancelled.
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

    def velocity(self) -> Tuple[float, float]:
        """
        Return (v_trans_m_s, v_rot_rad_s) implied by the currently-held keys.

        If both W and S are held, translate cancels to zero. Same for A/D.
        """
        with self._lock:
            pressed = set(self._pressed)

        v_trans = 0.0
        if pressed & _FORWARD_KEYS:
            v_trans += self.max_trans_m_s
        if pressed & _BACKWARD_KEYS:
            v_trans -= self.max_trans_m_s

        v_rot = 0.0
        if pressed & _LEFT_KEYS:
            v_rot += self.max_rot_rad_s
        if pressed & _RIGHT_KEYS:
            v_rot -= self.max_rot_rad_s

        return (v_trans, v_rot)

    def is_active(self) -> bool:
        """True if any mapped movement key is currently held."""
        with self._lock:
            return bool(self._pressed)

    def pressed_snapshot(self) -> set[str]:
        """Thread-safe copy of the currently-held key symbols."""
        with self._lock:
            return set(self._pressed)
