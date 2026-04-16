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
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

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
from fruit_finder.detector import Detection, FoodDetector  # noqa: E402
import _core  # noqa: E402


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
    ):
        super().__init__(daemon=True, name="FruitFinderWorker")
        self.robot = robot
        self.cam = cam
        self.detector = detector
        self.config = config
        self.command_q = command_q
        self.event_q = event_q
        self.shutdown_event = shutdown_event
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
                    )
                except Exception as exc:
                    self.event_q.put(("log", f"ERROR during sweep: {exc}"))
                    self._publish_state("IDLE")
                    _core.center_head(self.robot)
                    continue

                if self._stop_current.is_set():
                    self.event_q.put(("log", "Search stopped by user"))
                    self._publish_state("IDLE")
                    _core.center_head(self.robot)
                    continue

                if result is None:
                    self.event_q.put(("log", f"Target {target_label!r} not found"))
                    self._publish_state("IDLE")
                    _core.center_head(self.robot)
                    continue

                target_det, acq_pan, acq_tilt = result
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
                    )
                except Exception as exc:
                    self.event_q.put(("log", f"ERROR during track: {exc}"))
                    reason = "error"

                self.event_q.put(("log", f"Track ended: {reason}"))
                self._publish_state("IDLE")
                _core.center_head(self.robot)


# ----- GUI ----------------------------------------------------------------


class FruitFinderGUI:
    def __init__(
        self,
        root: tk.Tk,
        config: dict,
        robot,
        cam: CameraManager,
        detector: FoodDetector,
    ):
        self.root = root
        self.config = config
        self.robot = robot
        self.cam = cam
        self.detector = detector

        gui_cfg = config.get("gui", {}) or {}
        self.update_interval_ms = int(gui_cfg.get("update_interval_ms", 33))
        self.log_max_lines = int(gui_cfg.get("log_max_lines", 200))
        title = str(gui_cfg.get("window_title", "Stretch Fruit Finder"))

        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Queues for worker communication.
        self.command_q: queue.Queue = queue.Queue()
        self.event_q: queue.Queue = queue.Queue(maxsize=8)
        self.shutdown_event = threading.Event()

        # Worker thread.
        self.worker = Worker(
            robot, cam, detector, config,
            self.command_q, self.event_q, self.shutdown_event,
        )
        self.worker.start()

        # State.
        self._state_name = "IDLE"
        self._last_frame_time = 0.0
        self._fps_ema = 0.0
        self._current_photo = None  # keep reference or tkinter garbage-collects
        self._last_pan = 0.0
        self._last_tilt = 0.0

        self._build_layout()
        self._log("GUI ready. Set a target and press Start search.")
        self.root.after(self.update_interval_ms, self._poll_events)

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
        ttk.Label(target_frame, text="Food label:").grid(row=0, column=0, sticky="w")
        self.target_var = tk.StringVar(value="apple")
        self.target_entry = ttk.Entry(target_frame, textvariable=self.target_var, width=18)
        self.target_entry.grid(row=1, column=0, sticky="ew", pady=(2, 4))
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
        self.quit_btn = ttk.Button(
            btn_frame, text="Quit", command=self._on_close
        )
        self.quit_btn.grid(row=2, column=0, sticky="ew", pady=2)
        btn_frame.columnconfigure(0, weight=1)

        status_frame = ttk.LabelFrame(right, text="Status", padding=6)
        status_frame.grid(row=2, column=0, sticky="ew")
        self.state_label = ttk.Label(status_frame, text="State: IDLE")
        self.state_label.grid(row=0, column=0, sticky="w")
        self.pose_label = ttk.Label(status_frame, text="Head: pan=+0.00  tilt=+0.00")
        self.pose_label.grid(row=1, column=0, sticky="w")
        self.fps_label = ttk.Label(status_frame, text="FPS: 0.0")
        self.fps_label.grid(row=2, column=0, sticky="w")
        self.target_label = ttk.Label(status_frame, text="Target: apple")
        self.target_label.grid(row=3, column=0, sticky="w")

    # ----- events ---------------------------------------------------------

    def _on_set_target(self) -> None:
        target = self.target_var.get().strip().lower()
        if not target:
            self._log("Target cannot be empty.")
            return
        self.detector.set_target(target)
        self.target_label.configure(text=f"Target: {target}")
        self._log(f"Target set to {target!r}")

    def _on_start(self) -> None:
        target = self.target_var.get().strip().lower()
        if not target:
            self._log("Set a target first.")
            return
        self.detector.set_target(target)
        self.command_q.put(("start", target))
        self._log(f"Starting search for {target!r}")

    def _on_stop(self) -> None:
        self.command_q.put(("stop",))
        self.worker.stop_current()
        self._log("Stop requested")

    def _on_close(self) -> None:
        self._log("Shutting down...")
        self.shutdown_event.set()
        self.worker.stop_current()
        try:
            self.root.after(150, self.root.destroy)
        except Exception:
            self.root.destroy()

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

        root = tk.Tk()
        app = FruitFinderGUI(root, config, robot, cam, detector)
        root.mainloop()
    finally:
        if started:
            _core.center_head(robot)
            try:
                robot.stop()
            except Exception:
                pass
        cam.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
