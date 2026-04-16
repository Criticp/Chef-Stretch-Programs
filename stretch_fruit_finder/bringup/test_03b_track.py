"""
Level 3b bringup test — sweep until target, then visual tracking.

First bringup test that runs a continuous perception-to-motor loop.

Phase 1: the head sweeps through the configured search range (same as
Level 3) and exits as soon as the detector sees a target-class detection
above the acquire threshold.

Phase 2: closed-loop tracking. At detector rate, compute the target's
pixel offset from the frame center, convert to radians via the camera
intrinsics, and command a proportional correction to head pan/tilt. The
head should visibly follow the target if you move it.

Exit conditions:
- Ctrl+C
- Target lost for `tracking.lost_frames_timeout` consecutive frames
- Full sweep found nothing

Safety rails (same as Level 3):
- Refuses to run if robot isn't homed
- Clamps to joint soft limits
- try/finally re-centers head and calls r.stop() on every exit path

Run from the package root on the robot NUC:

    cd ~/chef_ai/Chef-Stretch-Programs/stretch_fruit_finder
    python3 bringup/test_03b_track.py --target apple

If the head tracks the WRONG direction (error grows — head turns AWAY
from the target), Ctrl+C and flip one of the signs:

    python3 bringup/test_03b_track.py --target apple --pan-sign +1
    python3 bringup/test_03b_track.py --target apple --tilt-sign +1

Once you know the right signs, put them in config.yaml `tracking:` so
they become the default.

Optional flags:
    --target LABEL     What to look for (default apple)
    --kp FLOAT         Override config tracking.kp
    --max-step FLOAT   Override config tracking.max_step_rad
    --pan-sign {-1,+1} Override config tracking.pan_sign
    --tilt-sign {-1,+1} Override config tracking.tilt_sign
    --save-every N     Save annotated PNG every Nth frame (0 disables)
    --outdir PATH      Where to save PNGs
    --config PATH      Override config.yaml path

Exit codes:
    0 sweep found target, tracking ran, exited cleanly (stopped or lost)
    1 startup / homing / detector failure
    2 sweep finished without finding the target
    3 interrupted mid-loop
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fruit_finder.camera import CameraManager  # noqa: E402
from fruit_finder.detector import Detection, FoodDetector  # noqa: E402
import _core  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=str, default="apple")
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--max-step", type=float, default=None)
    parser.add_argument("--pan-sign", type=int, default=None, choices=[-1, 1])
    parser.add_argument("--tilt-sign", type=int, default=None, choices=[-1, 1])
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=_HERE / "bringup_out")
    parser.add_argument("--config", type=Path, default=_PKG_ROOT / "config.yaml")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def annotate(
    color: np.ndarray,
    detections: list[Detection],
    real_pan: float,
    real_tilt: float,
    mode: str,
) -> np.ndarray:
    """Draw boxes + a header showing mode and head pose."""
    out = color.copy()
    h, w = out.shape[:2]

    # Crosshair at frame center.
    cv2.line(out, (w // 2 - 15, h // 2), (w // 2 + 15, h // 2), (64, 64, 64), 1)
    cv2.line(out, (w // 2, h // 2 - 15), (w // 2, h // 2 + 15), (64, 64, 64), 1)

    # Header bar.
    header = f"{mode}  pan={real_pan:+.2f}  tilt={real_tilt:+.2f}  dets={len(detections)}"
    cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(
        out, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
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
        cv2.rectangle(out, (det.x1, det.y1 - th - 6), (det.x1 + tw + 4, det.y1), box_color, -1)
        cv2.putText(
            out, label_text, (det.x1 + 2, det.y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

        if det.is_target:
            cx, cy = det.center
            cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)
            cv2.line(out, (w // 2, h // 2), (cx, cy), (0, 0, 255), 1)

    return out


def main() -> int:
    args = parse_args()
    # force=True because ultralytics.YOLO reconfigures the root logger when
    # it loads, suppressing our INFO output. force=True re-establishes it.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Build runtime params and let command-line overrides take precedence.
    params = _core.TrackerParams.from_config(config)
    if args.kp is not None:
        params.kp = args.kp
    if args.max_step is not None:
        params.max_step_rad = args.max_step
    if args.pan_sign is not None:
        params.pan_sign = float(args.pan_sign)
    if args.tilt_sign is not None:
        params.tilt_sign = float(args.tilt_sign)

    print(f"Starting CameraManager...")
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
    exit_code = 0
    stop_event = threading.Event()

    # Counter + outdir for optional snapshot saving.
    frame_n = {"count": 0}
    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    def on_pose(color, detections, real_pan, real_tilt):
        # Called from both sweep and track loops.
        mode = state.get("mode", "?")
        frame_n["count"] += 1

        target = _core.pick_target(detections, params.target_conf_min)
        if target is not None:
            tag = f"TARGET {target.label} conf={target.confidence:.2f}"
        elif detections:
            tag = f"{len(detections)} detection(s), no target match"
        else:
            tag = "no detections"

        if mode == "track":
            # Spam control: only log every 10th frame in track mode.
            if frame_n["count"] % 10 == 0:
                print(
                    f"[track #{frame_n['count']:04d}] pan={real_pan:+.2f} "
                    f"tilt={real_tilt:+.2f}  {tag}"
                )

        if args.save_every and frame_n["count"] % args.save_every == 0:
            annotated = annotate(color, detections, real_pan, real_tilt, mode)
            out_path = args.outdir / f"l3b_{stamp}_{frame_n['count']:05d}.png"
            cv2.imwrite(str(out_path), annotated)

    state = {"mode": "startup"}

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

        limits = _core.get_head_limits(robot)
        print(
            f"head limits: pan=[{limits.pan_lo:+.2f}, {limits.pan_hi:+.2f}] "
            f"tilt=[{limits.tilt_lo:+.2f}, {limits.tilt_hi:+.2f}]"
        )
        print(
            f"tracker params: kp={params.kp} max_step={params.max_step_rad} "
            f"pan_sign={int(params.pan_sign):+d} tilt_sign={int(params.tilt_sign):+d}"
        )

        # Phase 1: sweep
        state["mode"] = "sweep"
        print(f"Sweeping for target {args.target!r}...")
        result = _core.sweep_until_target(
            robot, cam, detector, config, args.target, stop_event, on_pose=on_pose
        )

        if result is None:
            if stop_event.is_set():
                print("Interrupted during sweep.")
                exit_code = 3
            else:
                print(f"Sweep finished without finding {args.target!r}.")
                exit_code = 2
            return exit_code

        target_det, acq_pan, acq_tilt = result
        print(
            f"ACQUIRED {target_det.label!r} at pan={acq_pan:+.2f} tilt={acq_tilt:+.2f} "
            f"conf={target_det.confidence:.2f}"
        )
        print("Entering track mode. Ctrl+C to stop.")

        # Phase 2: track
        state["mode"] = "track"
        reason = _core.track(
            robot, cam, detector, config, stop_event, on_pose=on_pose, params=params
        )
        print(f"Track loop exited: {reason}")

    except KeyboardInterrupt:
        print("\nCtrl+C received; shutting down.", file=sys.stderr)
        stop_event.set()
        exit_code = 3
    except Exception as exc:
        print(f"\nERROR during run: {exc}", file=sys.stderr)
        stop_event.set()
        exit_code = 1
    finally:
        if started:
            _core.center_head(robot)
            try:
                robot.stop()
            except Exception:
                pass
        cam.stop()

    if exit_code == 0:
        print("Level 3b search+track test: OK")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
