"""
Level 4 bringup test — basic tkinter GUI on the robot's local HDMI display.

A minimal window that shows the live camera feed with detection overlays
and gives you three buttons: set target, start search-and-track, stop.
Runs the exact same sweep/track loops as Level 3b, just wrapped in a
tkinter main loop so you can see what the robot sees while it works.

Why tkinter: requirements.txt pins opencv-python-headless, which has no
cv2.imshow. Tkinter ships with every Python install and Pillow is already
a requirement, so this adds zero new deps.

Run from the package root on the robot NUC, at the local console (so the
window opens on whatever monitor is plugged into the NUC's HDMI):

    cd ~/chef_ai/Chef-Stretch-Programs/stretch_fruit_finder
    python3 bringup/test_04_gui.py

Or over SSH with X-forwarding if you prefer (ssh -Y) — but the primary
target is a monitor plugged into the robot itself.

The GUI layout:
    +--------------------------+---------------------+
    |                          |  Target: [apple]    |
    |                          |  [Set target]       |
    |    live camera feed      |  [Start search]     |
    |    with bbox overlay     |  [Stop]             |
    |                          |                     |
    |                          |  State: IDLE        |
    +--------------------------+  Head: pan=+0.00    |
    |  log: ...                |        tilt=+0.00   |
    |  log: ...                |  FPS: 0             |
    +--------------------------+---------------------+

State machine: IDLE -> SEARCHING -> TRACKING -> IDLE (loops back on lost
or stop). The worker thread runs the sweep and track loops from _core.py
and publishes the latest (frame, detections, pan, tilt) via a queue. The
tkinter main thread polls the queue every ~33 ms and updates the canvas
and log.

Exit cleanly by closing the window or pressing the Stop button then Quit.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

# pygame (used by the gamepad reader) wants a display by default. Set the
# dummy driver before any pygame import so the GUI works over SSH without
# X-forwarding. If $DISPLAY is already set for the local HDMI session we
# leave it alone.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2
import numpy as np
import yaml
from PIL import Image, ImageTk

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fruit_finder.camera import CameraManager  # noqa: E402
from fruit_finder.detector import COCO_FOOD_NAMES, Detection, FoodDetector  # noqa: E402
from fruit_finder.gamepad import GamepadState, GamepadThread  # noqa: E402
import _arm_exec  # noqa: E402
import _arm_keyboard_driver  # noqa: E402
import _core  # noqa: E402
import _gamepad_exec  # noqa: E402
import _keyboard_driver  # noqa: E402


# Shown in the target dropdown to select any-food mode.
ANY_FOOD_OPTION = "(any food)"


# ----- worker -------------------------------------------------------------


class Worker(threading.Thread):
    """
    Background thread that runs the sweep/track state machine.

    Communication:
    - `command_q`: main thread drops ('start', target_label) or ('stop',)
      tuples to control the worker.
    - `event_q`: worker drops ('frame', color, detections, pan, tilt) and
      ('state', state_name) tuples for the main thread to render.
    - `stop_event`: set to exit the thread entirely on shutdown.
    """

    def __init__(
        self,
        robot,
        cam: CameraManager,
        detector: FoodDetector,
        config: dict,
        command_q: queue.Queue,
        event_q: queue.Queue,
        shutdown_event: threading.Event,
        robot_lock: Optional[threading.Lock] = None,
    ):
        super().__init__(daemon=True, name="FruitFinderWorker")
        self.robot = robot
        self.cam = cam
        self.detector = detector
        self.config = config
        self.command_q = command_q
        self.event_q = event_q
        self.shutdown_event = shutdown_event
        self.robot_lock = robot_lock
        self._stop_current = threading.Event()  # stops the current sweep/track loop

    def stop_current(self) -> None:
        self._stop_current.set()

    def _publish_state(self, name: str) -> None:
        try:
            self.event_q.put_nowait(("state", name))
        except queue.Full:
            pass

    def _make_on_pose(self):
        def on_pose(color, detections, pan, tilt):
            # Drop oldest frame if the GUI is falling behind.
            try:
                self.event_q.put_nowait(("frame", color, detections, pan, tilt))
            except queue.Full:
                try:
                    self.event_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.event_q.put_nowait(("frame", color, detections, pan, tilt))
                except queue.Full:
                    pass
        return on_pose

    def _idle_preview(self) -> None:
        """Publish a live preview frame (no detection) while IDLE."""
        color, _depth = self.cam.get_frames()
        if color is None:
            return
        pan, tilt = _core.read_head_pose(self.robot)
        try:
            self.event_q.put_nowait(("frame", color, [], pan, tilt))
        except queue.Full:
            pass

    def run(self) -> None:
        self._publish_state("IDLE")
        while not self.shutdown_event.is_set():
            # Drain any pending commands; most recent wins.
            cmd = None
            while True:
                try:
                    cmd = self.command_q.get_nowait()
                except queue.Empty:
                    break

            if cmd is None:
                self._idle_preview()
                time.sleep(0.05)
                continue

            if cmd[0] == "stop":
                self._stop_current.set()
                continue

            if cmd[0] == "start":
                target_label = cmd[1]
                self._stop_current.clear()
                self._publish_state("SEARCHING")

                try:
                    result = _core.sweep_until_target(
                        self.robot,
                        self.cam,
                        self.detector,
                        self.config,
                        target_label,
                        self._stop_current,
                        on_pose=self._make_on_pose(),
                        robot_lock=self.robot_lock,
                    )
                except Exception as exc:
                    self.event_q.put(("log", f"ERROR during sweep: {exc}"))
                    self._publish_state("IDLE")
                    _core.center_head(self.robot, robot_lock=self.robot_lock)
                    continue

                if self._stop_current.is_set():
                    self.event_q.put(("log", "Search stopped by user"))
                    self._publish_state("IDLE")
                    _core.center_head(self.robot, robot_lock=self.robot_lock)
                    continue

                if result is None:
                    self.event_q.put(("log", f"Target {target_label!r} not found"))
                    self._publish_state("IDLE")
                    _core.center_head(self.robot, robot_lock=self.robot_lock)
                    continue

                target_det, acq_pan, acq_tilt = result
                self.event_q.put(("acquired_label", target_det.label))
                self.event_q.put(
                    ("log", f"ACQUIRED {target_det.label!r} at pan={acq_pan:+.2f}, "
                     f"tilt={acq_tilt:+.2f} conf={target_det.confidence:.2f}")
                )
                self._publish_state("TRACKING")

                try:
                    reason = _core.track(
                        self.robot,
                        self.cam,
                        self.detector,
                        self.config,
                        self._stop_current,
                        on_pose=self._make_on_pose(),
                        robot_lock=self.robot_lock,
                    )
                except Exception as exc:
                    self.event_q.put(("log", f"ERROR during track: {exc}"))
                    reason = "error"

                self.event_q.put(("log", f"Track ended: {reason}"))
                self._publish_state("IDLE")
                _core.center_head(self.robot, robot_lock=self.robot_lock)


# ----- GUI ----------------------------------------------------------------


class FruitFinderGUI:
    def __init__(
        self,
        root: tk.Tk,
        config: dict,
        robot,
        cam: CameraManager,
        detector: FoodDetector,
        robot_lock: Optional[threading.Lock] = None,
        gp_state: Optional[GamepadState] = None,
        shutdown_event: Optional[threading.Event] = None,
    ):
        self.root = root
        self.config = config
        self.robot = robot
        self.cam = cam
        self.detector = detector
        self.robot_lock = robot_lock
        self.gp_state = gp_state

        gui_cfg = config.get("gui", {}) or {}
        self.update_interval_ms = int(gui_cfg.get("update_interval_ms", 33))
        self.log_max_lines = int(gui_cfg.get("log_max_lines", 200))
        title = str(gui_cfg.get("window_title", "Stretch Fruit Finder"))

        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Queues for worker communication.
        self.command_q: queue.Queue = queue.Queue()
        self.event_q: queue.Queue = queue.Queue(maxsize=8)
        # Either accept a shutdown_event from main() (so gamepad exec and
        # other threads outside the GUI share it) or create a fresh one.
        self.shutdown_event = shutdown_event or threading.Event()

        # Worker thread.
        self.worker = Worker(
            robot, cam, detector, config,
            self.command_q, self.event_q, self.shutdown_event,
            robot_lock=self.robot_lock,
        )
        self.worker.start()

        # State.
        self._state_name = "IDLE"
        self._last_frame_time = 0.0
        self._fps_ema = 0.0
        self._current_photo = None  # keep reference or tkinter garbage-collects
        self._last_pan = 0.0
        self._last_tilt = 0.0
        self._closing = False  # guards _on_close from double-firing

        # Keyboard driver binds WASD / arrow keys to a velocity state the
        # GamepadExecutor reads alongside the gamepad stick. Always active
        # — a plugged-in keyboard is the baseline input. Exposed as
        # `self.keyboard` so main() can pass its velocity method to the
        # GamepadExecutor after the GUI is constructed.
        self.keyboard = _keyboard_driver.KeyboardDriver(self.root, config)

        # Arm keyboard driver binds I/K/J/L/U/O/Y/H/N/M/[/] to
        # arm / lift / wrist / gripper. ArmExecutor (started in main())
        # consumes the driver state at 30 Hz.
        self.arm_keyboard = _arm_keyboard_driver.ArmKeyboardDriver(
            self.root, config
        )

        self._build_layout()
        self._log(
            "GUI ready. Set a target and press Start search. "
            "W/A/S/D (or arrows) drive the base; I/K/J/L drive lift+arm; "
            "U/O = wrist yaw; Y/H = wrist pitch; N/M = wrist roll; "
            "[ / ] = gripper close/open; X = stow arm."
        )
        self.root.after(self.update_interval_ms, self._poll_events)
        self.root.after(250, self._poll_slow)

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left column: camera feed + log underneath.
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(left, width=640, height=480, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.create_text(
            320, 240,
            text="Waiting for camera...",
            fill="white",
            font=("Helvetica", 14),
            tags="placeholder",
        )
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        log_frame = ttk.LabelFrame(left, text="Log", padding=4)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        left.rowconfigure(1, weight=0)
        self.log_text = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text["yscrollcommand"] = log_scroll.set
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        # Right column: controls + status.
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="ns")

        target_frame = ttk.LabelFrame(right, text="Target", padding=6)
        target_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(target_frame, text="What to find:").grid(row=0, column=0, sticky="w")

        # Build the dropdown from the detector's known food labels, with
        # "(any food)" at the top as an explicit option. Staying in sync
        # with COCO_FOOD_NAMES means adding a new food class to the
        # detector automatically adds it to this list.
        target_options = [ANY_FOOD_OPTION] + sorted(COCO_FOOD_NAMES.values())
        self.target_var = tk.StringVar(value="apple")
        self.target_combo = ttk.Combobox(
            target_frame,
            textvariable=self.target_var,
            values=target_options,
            state="readonly",
            width=18,
        )
        self.target_combo.grid(row=1, column=0, sticky="ew", pady=(2, 4))
        self.set_target_btn = ttk.Button(
            target_frame, text="Set target", command=self._on_set_target
        )
        self.set_target_btn.grid(row=2, column=0, sticky="ew")

        btn_frame = ttk.Frame(right)
        btn_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.start_btn = ttk.Button(
            btn_frame, text="Start search", command=self._on_start
        )
        self.start_btn.grid(row=0, column=0, sticky="ew", pady=2)
        self.stop_btn = ttk.Button(
            btn_frame, text="Stop", command=self._on_stop
        )
        self.stop_btn.grid(row=1, column=0, sticky="ew", pady=2)
        self.stow_btn = ttk.Button(
            btn_frame, text="Stow arm", command=self._on_stow_arm
        )
        self.stow_btn.grid(row=2, column=0, sticky="ew", pady=2)
        self.quit_btn = ttk.Button(
            btn_frame, text="Quit", command=self._on_close
        )
        self.quit_btn.grid(row=3, column=0, sticky="ew", pady=2)
        btn_frame.columnconfigure(0, weight=1)

        status_frame = ttk.LabelFrame(right, text="Status", padding=6)
        status_frame.grid(row=2, column=0, sticky="ew")
        self.state_label = ttk.Label(status_frame, text="State: IDLE")
        self.state_label.grid(row=0, column=0, sticky="w")
        self.pose_label = ttk.Label(status_frame, text="Head: pan=+0.00  tilt=+0.00")
        self.pose_label.grid(row=1, column=0, sticky="w")
        self.fps_label = ttk.Label(status_frame, text="FPS: 0.0")
        self.fps_label.grid(row=2, column=0, sticky="w")
        self.target_status_label = ttk.Label(status_frame, text="Target: apple")
        self.target_status_label.grid(row=3, column=0, sticky="w")
        self.locked_status_label = ttk.Label(status_frame, text="Locked: —")
        self.locked_status_label.grid(row=4, column=0, sticky="w")
        gp_initial = "Gamepad: —" if self.gp_state is None else "Gamepad: searching..."
        self.gamepad_status_label = ttk.Label(status_frame, text=gp_initial)
        self.gamepad_status_label.grid(row=5, column=0, sticky="w")
        self.keyboard_status_label = ttk.Label(status_frame, text="Keyboard: idle")
        self.keyboard_status_label.grid(row=6, column=0, sticky="w")
        self.arm_status_label = ttk.Label(status_frame, text="Arm: idle")
        self.arm_status_label.grid(row=7, column=0, sticky="w")
        self.arm_pose_label = ttk.Label(status_frame, text="Lift/Arm: — / —")
        self.arm_pose_label.grid(row=8, column=0, sticky="w")

    # ----- events ---------------------------------------------------------

    def _ui_value_to_target(self, value: str) -> tuple[str, str]:
        """
        Translate a dropdown value into (detector_arg, display_text).

        ANY_FOOD_OPTION maps to "" (the detector's any-food sentinel).
        A specific label maps to itself, lowercased.
        """
        v = (value or "").strip()
        if v == ANY_FOOD_OPTION or v == "":
            return ("", ANY_FOOD_OPTION)
        return (v.lower(), v.lower())

    def _on_set_target(self) -> None:
        detector_arg, display = self._ui_value_to_target(self.target_var.get())
        self.detector.set_target(detector_arg)
        self.target_status_label.configure(text=f"Target: {display}")
        self.locked_status_label.configure(text="Locked: —")
        self._log(f"Target set to {display}")

    def _on_start(self) -> None:
        detector_arg, display = self._ui_value_to_target(self.target_var.get())
        self.detector.set_target(detector_arg)
        self.target_status_label.configure(text=f"Target: {display}")
        self.locked_status_label.configure(text="Locked: —")
        self.command_q.put(("start", detector_arg))
        self._log(f"Starting search for {display}")

    def _on_stow_arm(self) -> None:
        # The ArmKeyboardDriver's stow flag is edge-triggered; ArmExecutor
        # will pick it up on its next tick (~33 ms).
        self.arm_keyboard._stow_requested = True  # set directly for click UX
        self._log("Stow arm requested")

    def _on_stop(self) -> None:
        self.command_q.put(("stop",))
        self.worker.stop_current()
        self._log("Stop requested")

    def _on_close(self) -> None:
        # Idempotent — both the Quit button and the gamepad Back button
        # route through here via shutdown_event, so guard against re-entry.
        if self._closing:
            return
        self._closing = True
        self._log("Shutting down...")
        self.shutdown_event.set()
        self.worker.stop_current()
        try:
            self.root.after(150, self.root.destroy)
        except Exception:
            self.root.destroy()

    def _poll_slow(self) -> None:
        """
        Slow tick (~4 Hz): updates the gamepad-status line and closes the
        GUI if something else (Ctrl+C, gamepad Back button, GamepadExecutor
        error) has already set shutdown_event.
        """
        if self._closing:
            return

        # External shutdown trigger — e.g. gamepad Back button was pressed.
        if self.shutdown_event.is_set():
            self._on_close()
            return

        # Refresh the gamepad status line.
        if self.gp_state is not None:
            snap = self.gp_state.get_copy()
            if not snap.connected:
                text = "Gamepad: not connected"
            else:
                driving = abs(snap.left_x) > 0.01 or abs(snap.left_y) > 0.01
                if driving:
                    text = (
                        f"Gamepad: driving "
                        f"(stick x={snap.left_x:+.2f}, y={snap.left_y:+.2f})"
                    )
                else:
                    text = "Gamepad: connected (idle)"
            self.gamepad_status_label.configure(text=text)

        # Refresh the keyboard status line (base teleop).
        pressed = self.keyboard.pressed_snapshot()
        if pressed:
            pretty = {
                "w": "W", "W": "W", "Up": "↑",
                "s": "S", "S": "S", "Down": "↓",
                "a": "A", "A": "A", "Left": "←",
                "d": "D", "D": "D", "Right": "→",
            }
            labels = sorted({pretty.get(k, k) for k in pressed})
            self.keyboard_status_label.configure(
                text=f"Keyboard: {'+'.join(labels)}"
            )
        else:
            self.keyboard_status_label.configure(text="Keyboard: idle")

        # Refresh the arm keyboard status line (I/K/J/L/U/O/Y/H/N/M/[/]).
        arm_pressed = self.arm_keyboard.pressed_snapshot()
        if arm_pressed:
            pretty = {
                "i": "I", "I": "I", "k": "K", "K": "K",
                "j": "J", "J": "J", "l": "L", "L": "L",
                "u": "U", "U": "U", "o": "O", "O": "O",
                "y": "Y", "Y": "Y", "h": "H", "H": "H",
                "n": "N", "N": "N", "m": "M", "M": "M",
                "bracketleft": "[", "bracketright": "]",
            }
            labels = sorted({pretty.get(k, k) for k in arm_pressed})
            self.arm_status_label.configure(
                text=f"Arm: {'+'.join(labels)}"
            )
        else:
            self.arm_status_label.configure(text="Arm: idle")

        # Live lift / arm positions, best-effort read under the lock.
        if self.robot_lock is not None:
            try:
                with self.robot_lock:
                    lift_pos = float(self.robot.lift.status.get("pos", 0.0))
                    arm_pos = float(self.robot.arm.status.get("pos", 0.0))
                self.arm_pose_label.configure(
                    text=f"Lift/Arm: {lift_pos:+.2f} m / {arm_pos:+.2f} m"
                )
            except Exception:
                # Don't spam the log if a read fails transiently.
                pass

        self.root.after(250, self._poll_slow)

    # ----- queue poll -----------------------------------------------------

    def _poll_events(self) -> None:
        drained = 0
        while drained < 16:
            try:
                evt = self.event_q.get_nowait()
            except queue.Empty:
                break
            drained += 1

            kind = evt[0]
            if kind == "frame":
                _, color, detections, pan, tilt = evt
                self._render_frame(color, detections, pan, tilt)
            elif kind == "state":
                self._state_name = evt[1]
                self.state_label.configure(text=f"State: {self._state_name}")
                if self._state_name == "IDLE":
                    self.locked_status_label.configure(text="Locked: —")
            elif kind == "acquired_label":
                self.locked_status_label.configure(text=f"Locked: {evt[1]}")
            elif kind == "log":
                self._log(evt[1])

        if not self.shutdown_event.is_set():
            self.root.after(self.update_interval_ms, self._poll_events)

    def _render_frame(
        self,
        color: np.ndarray,
        detections: list[Detection],
        pan: float,
        tilt: float,
    ) -> None:
        now = time.time()
        if self._last_frame_time:
            dt = now - self._last_frame_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps_ema = 0.9 * self._fps_ema + 0.1 * inst_fps
        self._last_frame_time = now

        self._last_pan = pan
        self._last_tilt = tilt

        overlay = self._draw_overlay(color, detections, pan, tilt)

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._current_photo = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.canvas_image_id, image=self._current_photo)
        self.canvas.delete("placeholder")

        self.pose_label.configure(text=f"Head: pan={pan:+.2f}  tilt={tilt:+.2f}")
        self.fps_label.configure(text=f"FPS: {self._fps_ema:.1f}")

    def _draw_overlay(
        self,
        color: np.ndarray,
        detections: list[Detection],
        pan: float,
        tilt: float,
    ) -> np.ndarray:
        out = color.copy()
        h, w = out.shape[:2]

        # Crosshair.
        cv2.line(out, (w // 2 - 15, h // 2), (w // 2 + 15, h // 2), (80, 80, 80), 1)
        cv2.line(out, (w // 2, h // 2 - 15), (w // 2, h // 2 + 15), (80, 80, 80), 1)

        # Header bar.
        header = f"{self._state_name}  pan={pan:+.2f}  tilt={tilt:+.2f}  dets={len(detections)}"
        cv2.rectangle(out, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(
            out, header, (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        for det in detections:
            if det.is_target:
                box_color = (0, 0, 255)
            elif det.is_food:
                box_color = (0, 255, 0)
            elif det.is_kitchen:
                box_color = (255, 128, 0)
            else:
                box_color = (160, 160, 160)
            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), box_color, 2)

            label_text = f"{det.label} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                out,
                (det.x1, det.y1 - th - 6),
                (det.x1 + tw + 4, det.y1),
                box_color,
                -1,
            )
            cv2.putText(
                out, label_text, (det.x1 + 2, det.y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

            if det.is_target:
                cx, cy = det.center
                cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)
                cv2.line(out, (w // 2, h // 2), (cx, cy), (0, 0, 255), 1)

        return out

    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        # Trim old lines.
        end_index = self.log_text.index("end-1c")
        line_count = int(end_index.split(".")[0])
        if line_count > self.log_max_lines:
            trim_to = line_count - self.log_max_lines
            self.log_text.delete("1.0", f"{trim_to}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


# ----- main ---------------------------------------------------------------


def main() -> int:
    # force=True because ultralytics.YOLO reconfigures the root logger when
    # it loads, suppressing our INFO output. force=True re-establishes it.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    config_path = _PKG_ROOT / "config.yaml"
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    print("Starting CameraManager...")
    cam = CameraManager(config)
    try:
        cam.start()
    except Exception as exc:
        print(f"ERROR: camera failed to start: {exc}", file=sys.stderr)
        return 1

    print("Loading FoodDetector...")
    detector = FoodDetector(config)
    try:
        detector.load()
    except Exception as exc:
        print(f"ERROR: detector load failed: {exc}", file=sys.stderr)
        cam.stop()
        return 1
    # Ultralytics silently lowers the root logger; re-raise it so our INFO logs print.
    logging.getLogger().setLevel(logging.INFO)

    try:
        import stretch_body.robot as sb_robot
    except Exception as exc:
        print(f"ERROR: cannot import stretch_body: {exc}", file=sys.stderr)
        cam.stop()
        return 1

    robot = sb_robot.Robot()
    started = False

    # Shared between GUI worker, gamepad reader, gamepad executor.
    # Any thread that touches the robot object acquires robot_lock first;
    # shutdown_event is the global "stop everything and clean up" signal.
    robot_lock = threading.Lock()
    shutdown_event = threading.Event()

    gp_state = GamepadState()
    gp_thread: Optional[GamepadThread] = None
    gp_exec: Optional[_gamepad_exec.GamepadExecutor] = None
    arm_exec: Optional[_arm_exec.ArmExecutor] = None

    try:
        started = robot.startup()
        if not started:
            print("ERROR: robot.startup() returned False", file=sys.stderr)
            cam.stop()
            return 1
        if not robot.is_homed():
            print(
                "ERROR: robot is not homed. Run `stretch_robot_home.py` first.",
                file=sys.stderr,
            )
            robot.stop()
            cam.stop()
            return 1

        # Start the gamepad reader thread. It only READS input; it doesn't
        # touch the robot on its own, so order relative to the executor
        # doesn't matter.
        gp_thread = GamepadThread(config, gp_state)
        gp_thread.start()

        # Build the GUI before starting the GamepadExecutor, because the
        # executor needs to pull keyboard velocity from the KeyboardDriver
        # that lives inside the GUI.
        root = tk.Tk()
        app = FruitFinderGUI(
            root, config, robot, cam, detector,
            robot_lock=robot_lock,
            gp_state=gp_state,
            shutdown_event=shutdown_event,
        )

        gp_exec = _gamepad_exec.GamepadExecutor(
            robot, gp_state, robot_lock, config, shutdown_event,
            extra_velocity_source=app.keyboard.velocity,
        )
        gp_exec.start()

        # Arm executor consumes the arm-keyboard driver that lives inside
        # the GUI. Needs the same shared robot_lock so its push_command
        # doesn't race with the base's.
        arm_exec = _arm_exec.ArmExecutor(
            robot, app.arm_keyboard, robot_lock, config, shutdown_event
        )
        arm_exec.start()

        root.mainloop()
    finally:
        # Signal every thread to wind down, then join.
        shutdown_event.set()
        if arm_exec is not None:
            arm_exec.join(timeout=2.0)
        if gp_exec is not None:
            gp_exec.join(timeout=2.0)
        if gp_thread is not None:
            gp_thread.stop()
            gp_thread.join(timeout=2.0)

        if started:
            # In case the gamepad executor didn't get a chance to stop the
            # base, zero it explicitly before we let go of the robot.
            try:
                with robot_lock:
                    robot.base.set_velocity(0.0, 0.0)
                    robot.push_command()
            except Exception:
                pass
            _core.center_head(robot, robot_lock=robot_lock)
            try:
                robot.stop()
            except Exception:
                pass
        cam.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
