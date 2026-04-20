"""
Level 3c bringup test — drive the base with a gamepad while the head
tracker keeps its lock on a food target.

Combines L3b (sweep-then-track, head-only) with base teleop from an
Xbox-style gamepad. Threads:

    main         orchestrates + runs sweep then track loops
    GamepadThread reads joystick -> populates GamepadState
    GamepadExecutor reads GamepadState -> commands base velocity

All robot method calls in all threads are serialised through a single
threading.Lock. Head commands (Dynamixel, immediate) and base velocity
commands (stepper, pushed) don't share motion state, but they do share
the command queue that `robot.push_command()` flushes, so the lock is
prudent.

Controls while running:
    left stick Y  : base translate (forward/back)
    left stick X  : base rotate    (turn CCW / CW)
    back button   : emergency stop -> re-center head + halt base + exit
    Ctrl+C        : same as back button

Run from the package root on the robot NUC, with the gamepad's USB
dongle plugged in:

    cd ~/chef_ai/Chef-Stretch-Programs/stretch_fruit_finder
    python3 bringup/test_03c_gamepad_track.py --target apple
    python3 bringup/test_03c_gamepad_track.py --target any

Optional flags:
    --target LABEL          What to find ("any" for any-food mode).
    --no-tracker            Skip sweep/track; gamepad-only base teleop.
    --translate-speed M_S   Override max base translate speed (m/s).
    --rotate-speed RAD_S    Override max base rotate speed (rad/s).
    --config PATH           Override config.yaml location.

Exit codes:
    0  clean exit (Ctrl+C, stop button, or tracker said "lost")
    1  startup / homing / camera / detector failure
    2  unexpected error during run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

# SDL tries to open a display when pygame.init() runs, even for joystick use.
# The gamepad thread init pygame lazily — set the dummy driver here so it
# works fine over SSH without X-forwarding. If $DISPLAY IS set, respect it.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import yaml  # noqa: E402

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fruit_finder.camera import CameraManager  # noqa: E402
from fruit_finder.detector import FoodDetector  # noqa: E402
from fruit_finder.gamepad import GamepadState, GamepadThread  # noqa: E402
import _core  # noqa: E402
import _gamepad_exec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=str, default="apple",
                        help="Food label to find, or 'any' for any-food mode.")
    parser.add_argument("--no-tracker", action="store_true",
                        help="Skip sweep/track; gamepad-only base teleop.")
    parser.add_argument("--translate-speed", type=float, default=None,
                        help="Override gamepad.base_translate_speed (m/s).")
    parser.add_argument("--rotate-speed", type=float, default=None,
                        help="Override gamepad.base_rotate_speed (rad/s).")
    parser.add_argument("--config", type=Path, default=_PKG_ROOT / "config.yaml")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # CLI overrides for gamepad speeds write into the config dict so both
    # the GamepadExecutor and any diagnostic logs see the same numbers.
    if args.translate_speed is not None or args.rotate_speed is not None:
        config.setdefault("gamepad", {})
        if args.translate_speed is not None:
            config["gamepad"]["base_translate_speed"] = float(args.translate_speed)
        if args.rotate_speed is not None:
            config["gamepad"]["base_rotate_speed"] = float(args.rotate_speed)

    # --- camera ---
    print("Starting CameraManager...")
    cam = CameraManager(config)
    try:
        cam.start()
    except Exception as exc:
        print(f"ERROR: camera failed to start: {exc}", file=sys.stderr)
        return 1

    # --- detector (skip if we're gamepad-only) ---
    detector = FoodDetector(config)
    if not args.no_tracker:
        print("Loading FoodDetector...")
        try:
            detector.load()
        except Exception as exc:
            print(f"ERROR: detector load failed: {exc}", file=sys.stderr)
            cam.stop()
            return 1
        logging.getLogger().setLevel(logging.INFO)

    # --- stretch_body ---
    try:
        import stretch_body.robot as sb_robot
    except Exception as exc:
        print(f"ERROR: cannot import stretch_body: {exc}", file=sys.stderr)
        cam.stop()
        return 1

    robot = sb_robot.Robot()
    started = False
    exit_code = 0

    # Shared between tracker thread + gamepad thread + main.
    stop_event = threading.Event()
    robot_lock = threading.Lock()

    # Gamepad plumbing.
    gp_state = GamepadState()
    gp_thread: GamepadThread | None = None
    gp_exec: _gamepad_exec.GamepadExecutor | None = None
    tracker_thread: threading.Thread | None = None

    try:
        print("Connecting to stretch_body.robot.Robot()...")
        started = robot.startup()
        if not started:
            print("ERROR: robot.startup() returned False", file=sys.stderr)
            return 1
        if not robot.is_homed():
            print(
                "ERROR: robot is not homed. Run `stretch_robot_home.py` first.",
                file=sys.stderr,
            )
            return 1

        # Start the gamepad reader + executor. These run regardless of
        # whether the tracker is enabled.
        gp_thread = GamepadThread(config, gp_state)
        gp_thread.start()
        gp_exec = _gamepad_exec.GamepadExecutor(
            robot, gp_state, robot_lock, config, stop_event
        )
        gp_exec.start()

        # Give the joystick a moment to be discovered before logging status.
        time.sleep(0.3)
        if gp_state.get_copy().connected:
            print("Gamepad connected.")
        else:
            print(
                "Gamepad not yet connected — left stick drives base once it is. "
                "(Base stays idle until a pad is found.)"
            )

        def tracker_body() -> None:
            """Runs sweep then track in this background thread."""
            try:
                if args.target.strip().lower() in ("any", "(any)", "any food", ""):
                    initial = ""
                else:
                    initial = args.target.strip().lower()

                print(
                    f"Tracker: searching for "
                    f"{'(any food)' if initial == '' else repr(initial)}..."
                )
                result = _core.sweep_until_target(
                    robot, cam, detector, config, initial, stop_event,
                    robot_lock=robot_lock,
                )
                if result is None:
                    if stop_event.is_set():
                        print("Tracker: interrupted during sweep.")
                    else:
                        print(
                            f"Tracker: sweep finished without finding "
                            f"{initial!r}. Base teleop still active."
                        )
                    return

                target_det, acq_pan, acq_tilt = result
                print(
                    f"Tracker: ACQUIRED {target_det.label!r} at "
                    f"pan={acq_pan:+.2f} tilt={acq_tilt:+.2f} "
                    f"conf={target_det.confidence:.2f}"
                )
                print("Tracker: entering track mode.")
                reason = _core.track(
                    robot, cam, detector, config, stop_event,
                    robot_lock=robot_lock,
                )
                print(f"Tracker: track loop exited ({reason}).")
            except Exception as exc:
                print(f"Tracker: error — {exc}", file=sys.stderr)

        if not args.no_tracker:
            tracker_thread = threading.Thread(
                target=tracker_body, name="TrackerThread", daemon=True
            )
            tracker_thread.start()
        else:
            print("Tracker disabled (--no-tracker). Gamepad-only base teleop.")

        # Main thread idles until stop_event fires or Ctrl+C.
        print("Ready. Drive with left stick, press Back (or Ctrl+C) to stop.")
        while not stop_event.is_set():
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nCtrl+C; shutting down.", file=sys.stderr)
        stop_event.set()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        stop_event.set()
        exit_code = 2
    finally:
        # Signal everyone to wind down and wait for threads to notice.
        stop_event.set()

        if tracker_thread is not None:
            tracker_thread.join(timeout=3.0)
        if gp_exec is not None:
            gp_exec.join(timeout=2.0)
        if gp_thread is not None:
            gp_thread.stop()
            gp_thread.join(timeout=2.0)

        if started:
            # Stop base explicitly in case the gamepad exec didn't get a
            # chance to do so, then re-center the head.
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

    if exit_code == 0:
        print("Level 3c gamepad+track test: OK")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
