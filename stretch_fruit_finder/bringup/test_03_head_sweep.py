"""
Level 3 bringup test — head pan sweep with detection at each pose.

First bringup test that actually moves the robot. Head only: no arm, no
base, no gripper. Tilts the head down to the configured search tilt, then
pans from search_pan_min_rad to search_pan_max_rad in search_pan_step_rad
increments. At each pose we wait for the camera to stabilise, grab a
frame, run YOLO, print every detection, and save an annotated PNG.

Safety rails:
- Refuses to run if `r.is_homed()` is False. Run `stretch_robot_home.py`
  first if that happens.
- Clamps the configured sweep range to each joint's hard soft-motion
  limits before the first move.
- try/finally block centers the head and calls r.stop() even on Ctrl+C
  or exception.

Run from the package root on the robot NUC:

    cd ~/chef_ai/Chef-Stretch-Programs/stretch_fruit_finder
    python3 bringup/test_03_head_sweep.py
    python3 bringup/test_03_head_sweep.py --target apple

Optional flags:
    --target LABEL     Mark matching detections as TARGET in output.
    --warmup N         Camera warmup frames per pose (default 6).
    --pause S          Extra seconds to dwell at each pose (default
                       from config navigation.search_pause_sec).
    --outdir PATH      Directory to save annotated PNGs.
    --config PATH      Override config.yaml path.

Exit codes:
    0  sweep completed, head returned to center, robot stopped cleanly
    1  camera/detector failed to start, robot not homed, or startup failed
    2  robot interrupted mid-sweep (still attempts cleanup)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Import path shim so `fruit_finder` resolves regardless of where this
# script is invoked from.
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from fruit_finder.camera import CameraManager  # noqa: E402
from fruit_finder.detector import Detection, FoodDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--pause", type=float, default=None)
    parser.add_argument("--outdir", type=Path, default=_HERE / "bringup_out")
    parser.add_argument("--config", type=Path, default=_PKG_ROOT / "config.yaml")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def drain_frames(cam: CameraManager, n: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Grab n frames, returning the last good (color, depth) pair."""
    last_color: np.ndarray | None = None
    last_depth: np.ndarray | None = None
    for _ in range(n):
        color, depth = cam.get_frames()
        if color is not None and depth is not None:
            last_color, last_depth = color, depth
    return last_color, last_depth


def median_depth_at(depth: np.ndarray, cx: int, cy: int, half: int = 5) -> float:
    """Median depth in mm over a window around (cx, cy), zeros excluded."""
    h, w = depth.shape
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    window = depth[y0:y1, x0:x1]
    valid = window[window > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def annotate(
    color: np.ndarray,
    detections: list[Detection],
    depth_mm: np.ndarray,
    pan_rad: float,
    tilt_rad: float,
) -> np.ndarray:
    """Draw boxes and a header with the current head pose."""
    out = color.copy()

    header = f"pan={pan_rad:+.2f} rad  tilt={tilt_rad:+.2f} rad  dets={len(detections)}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
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
        cx, cy = det.center
        depth_m = median_depth_at(depth_mm, cx, cy) / 1000.0
        depth_str = f" {depth_m:.2f}m" if depth_m > 0 else ""
        label_text = f"{det.label} {det.confidence:.2f}{depth_str}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            out,
            (det.x1, det.y1 - th - 6),
            (det.x1 + tw + 4, det.y1),
            box_color,
            -1,
        )
        cv2.putText(
            out,
            label_text,
            (det.x1 + 2, det.y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def read_head_pose(robot) -> tuple[float, float]:
    """Best-effort read of the current head pan/tilt in radians."""
    try:
        pan = float(robot.head.get_joint("head_pan").status["pos"])
    except Exception:
        pan = 0.0
    try:
        tilt = float(robot.head.get_joint("head_tilt").status["pos"])
    except Exception:
        tilt = 0.0
    return pan, tilt


def joint_hard_limits(robot, joint_name: str) -> tuple[float, float] | None:
    """Return (lo, hi) hard soft-motion limits for a head joint, or None."""
    try:
        joint = robot.head.get_joint(joint_name)
        hard = joint.soft_motion_limits["hard"]
        return float(hard[0]), float(hard[1])
    except Exception:
        return None


def clamp_sweep(
    pan_min: float,
    pan_max: float,
    step: float,
    hard_limits: tuple[float, float] | None,
) -> tuple[float, float, float]:
    """Clamp the requested sweep to the joint's hard limits if available."""
    if hard_limits is not None:
        lo, hi = hard_limits
        # Keep a small safety margin off the hardware limit.
        margin = 0.05
        pan_min = max(pan_min, lo + margin)
        pan_max = min(pan_max, hi - margin)
    if pan_max < pan_min:
        # If clamping inverted the range, fall back to a safe centered band.
        pan_min, pan_max = -0.5, 0.5
    return pan_min, pan_max, step


def pan_angles(pan_min: float, pan_max: float, step: float) -> list[float]:
    """Build a list of pan angles from min to max inclusive, step > 0."""
    if step <= 0:
        return [0.0]
    n = int(round((pan_max - pan_min) / step)) + 1
    return [float(pan_min + i * step) for i in range(n) if (pan_min + i * step) <= pan_max + 1e-6]


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    nav = config["navigation"]
    pan_min_cfg = float(nav["search_pan_min_rad"])
    pan_max_cfg = float(nav["search_pan_max_rad"])
    pan_step = float(nav["search_pan_step_rad"])
    tilt_target = float(nav["search_tilt_rad"])
    dwell = float(args.pause if args.pause is not None else nav["search_pause_sec"])

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
    if args.target:
        detector.set_target(args.target)

    # Import stretch_body here, not at module top, so `--help` works without it.
    print("Connecting to stretch_body.robot.Robot()...")
    try:
        import stretch_body.robot as sb_robot
    except Exception as exc:
        print(f"ERROR: cannot import stretch_body: {exc}", file=sys.stderr)
        cam.stop()
        return 1

    robot = sb_robot.Robot()
    started = False
    interrupted = False
    exit_code = 0

    try:
        started = robot.startup()
        if not started:
            print("ERROR: robot.startup() returned False", file=sys.stderr)
            return 1

        if not robot.is_homed():
            print(
                "ERROR: robot is not homed. Run `stretch_robot_home.py` first, "
                "then re-run this test.",
                file=sys.stderr,
            )
            return 1

        # Apply hard-limit clamping to the configured sweep range.
        hard_pan = joint_hard_limits(robot, "head_pan")
        hard_tilt = joint_hard_limits(robot, "head_tilt")
        print(f"head_pan  hard limits: {hard_pan}")
        print(f"head_tilt hard limits: {hard_tilt}")

        pan_min, pan_max, _ = clamp_sweep(pan_min_cfg, pan_max_cfg, pan_step, hard_pan)
        # Clamp tilt too so a bad config can't crash the Dynamixel.
        if hard_tilt is not None:
            tilt_target = max(hard_tilt[0] + 0.05, min(hard_tilt[1] - 0.05, tilt_target))

        print(
            f"Sweep plan: pan {pan_min:+.2f} -> {pan_max:+.2f} rad "
            f"in {pan_step:+.2f} rad steps, tilt {tilt_target:+.2f} rad, "
            f"dwell {dwell:.2f} s"
        )

        angles = pan_angles(pan_min, pan_max, pan_step)
        print(f"Will visit {len(angles)} poses: {[round(a, 2) for a in angles]}")

        args.outdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")

        # Move to search tilt and center pan before the sweep.
        print(f"Tilting head to {tilt_target:+.2f} rad and centering pan...")
        robot.head.move_to("head_pan", 0.0)
        robot.head.move_to("head_tilt", tilt_target)
        robot.wait_command()
        time.sleep(0.2)  # Let the Dynamixel settle.

        # Drain a handful of frames so auto-exposure catches up.
        drain_frames(cam, args.warmup)

        sightings: dict[str, list[tuple[float, float]]] = {}  # label -> [(pan, conf), ...]

        for step_idx, pan in enumerate(angles):
            print(f"[{step_idx + 1}/{len(angles)}] pan -> {pan:+.2f} rad")
            robot.head.move_to("head_pan", pan)
            robot.wait_command()
            time.sleep(dwell)

            color, depth = drain_frames(cam, args.warmup)
            if color is None or depth is None:
                print("  WARN no frame arrived; skipping this pose")
                continue

            real_pan, real_tilt = read_head_pose(robot)
            detections = detector.detect(color)

            if not detections:
                print("  (no food/kitchen detections)")
            for det in detections:
                cx, cy = det.center
                depth_m = median_depth_at(depth, cx, cy) / 1000.0
                tags = []
                if det.is_target:
                    tags.append("TARGET")
                elif det.is_food:
                    tags.append("food")
                elif det.is_kitchen:
                    tags.append("kitchen")
                depth_str = f"{depth_m:.2f} m" if depth_m > 0 else "no depth"
                tag_str = ",".join(tags) if tags else "-"
                print(
                    f"  {det.label:<12} conf={det.confidence:.2f}  "
                    f"depth={depth_str}  [{tag_str}]"
                )
                sightings.setdefault(det.label, []).append((real_pan, det.confidence))

            annotated = annotate(color, detections, depth, real_pan, real_tilt)
            out_path = (
                args.outdir
                / f"l3_pan_{stamp}_{step_idx:02d}_{int(real_pan * 100):+04d}.png"
            )
            cv2.imwrite(str(out_path), annotated)

        # Summary: best angle per label by confidence.
        if sightings:
            print("\nBest sighting per label (by confidence):")
            for label in sorted(sightings):
                best = max(sightings[label], key=lambda pc: pc[1])
                print(f"  {label:<12} pan={best[0]:+.2f} rad  conf={best[1]:.2f}")
        else:
            print("\nNo food/kitchen items seen during the sweep.")

        print("Sweep complete. Returning head to center...")
        robot.head.move_to("head_pan", 0.0)
        robot.head.move_to("head_tilt", 0.0)
        robot.wait_command()

    except KeyboardInterrupt:
        print("\nInterrupted by user; returning head to center and shutting down.", file=sys.stderr)
        interrupted = True
        exit_code = 2
    except Exception as exc:
        print(f"\nERROR during sweep: {exc}", file=sys.stderr)
        interrupted = True
        exit_code = 2
    finally:
        if started:
            try:
                # Best-effort re-center on any exit path.
                robot.head.move_to("head_pan", 0.0)
                robot.head.move_to("head_tilt", 0.0)
                robot.wait_command()
            except Exception:
                pass
            try:
                robot.stop()
            except Exception:
                pass
        cam.stop()

    if not interrupted:
        print("Level 3 head sweep test: OK")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
