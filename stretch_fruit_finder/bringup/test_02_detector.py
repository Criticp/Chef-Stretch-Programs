"""
Level 2 bringup test — camera + YOLO food detector, still no motion.

Builds on Level 1. Opens the D435i via `fruit_finder.camera.CameraManager`,
loads the YOLO food/kitchen detector via `fruit_finder.detector.FoodDetector`,
runs inference on one stable frame, prints every detection it found (label,
confidence, bbox, center pixel, depth at that pixel), and saves an annotated
color PNG plus a colorised depth PNG.

This script does NOT touch stretch_body. It does NOT move the robot. Head
pan/tilt are not queried — we can't do that until Level 3 — so we don't try
to compute robot-frame 3D coordinates here. Just camera-frame depth at the
box center, which is enough to sanity-check the pipeline.

Run from the package root on the robot NUC:

    cd ~/chef_ai/Chef-Stretch-Programs/stretch_fruit_finder
    python3 bringup/test_02_detector.py

Optional flags:
    --frames N         Warmup frames before inference (default 15).
    --target LABEL     Mark any detection whose label contains LABEL as
                       is_target=True (e.g. --target apple). Optional.
    --outdir PATH      Where to save PNGs (default bringup/bringup_out).
    --config PATH      Override config.yaml location.

Exit codes:
    0  detector loaded, at least one inference ran cleanly
    1  camera failed to start, or no frame ever arrived, or detector load failed

Notes:
- First run downloads yolov8n.pt (~6 MB) and exports to OpenVINO. That takes
  ~30-60 s and requires internet. Both artifacts land in the current working
  directory, which is why you must run from stretch_fruit_finder/.
- Zero detections is not a failure of this test. It just means nothing YOLO
  recognises as food/kitchen was in front of the camera. Point the head at
  an apple or a mug and try again.
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

# Make `fruit_finder` importable regardless of where the script is invoked from.
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from fruit_finder.camera import CameraManager  # noqa: E402
from fruit_finder.detector import Detection, FoodDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=15)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--outdir", type=Path, default=_HERE / "bringup_out")
    parser.add_argument("--config", type=Path, default=_PKG_ROOT / "config.yaml")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def grab_stable_frame(
    cam: CameraManager, warmup_frames: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Drain `warmup_frames` frames to let auto-exposure settle, then grab one."""
    last_color: np.ndarray | None = None
    last_depth: np.ndarray | None = None
    for i in range(warmup_frames):
        color, depth = cam.get_frames()
        if color is not None and depth is not None:
            last_color, last_depth = color, depth
    return last_color, last_depth


def median_depth_at(depth: np.ndarray, cx: int, cy: int, half: int = 5) -> float:
    """Median depth in mm in a (2*half+1)^2 window around (cx, cy), excluding zeros."""
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


def describe_detection(det: Detection, depth_mm: np.ndarray) -> str:
    cx, cy = det.center
    depth_m = median_depth_at(depth_mm, cx, cy) / 1000.0

    tags = []
    if det.is_food:
        tags.append("food")
    if det.is_kitchen:
        tags.append("kitchen")
    if det.is_target:
        tags.append("TARGET")
    tag_str = ",".join(tags) if tags else "-"

    depth_str = f"{depth_m:.2f} m" if depth_m > 0 else "no depth"
    return (
        f"  {det.label:<12} conf={det.confidence:.2f}  "
        f"bbox=({det.x1},{det.y1})-({det.x2},{det.y2})  "
        f"center=({cx},{cy})  depth={depth_str}  [{tag_str}]"
    )


def annotate(
    color: np.ndarray, detections: list[Detection], depth_mm: np.ndarray
) -> np.ndarray:
    """Draw boxes, labels, and depth-at-center on a copy of the color frame."""
    out = color.copy()
    for det in detections:
        # Target gets red, food gets green, kitchen gets blue, other gets grey.
        if det.is_target:
            color_bgr = (0, 0, 255)
        elif det.is_food:
            color_bgr = (0, 255, 0)
        elif det.is_kitchen:
            color_bgr = (255, 128, 0)
        else:
            color_bgr = (160, 160, 160)

        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color_bgr, 2)

        cx, cy = det.center
        depth_m = median_depth_at(depth_mm, cx, cy) / 1000.0
        depth_str = f" {depth_m:.2f}m" if depth_m > 0 else ""
        label_text = f"{det.label} {det.confidence:.2f}{depth_str}"

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            out,
            (det.x1, det.y1 - th - 6),
            (det.x1 + tw + 4, det.y1),
            color_bgr,
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


def colorise_depth(depth_mm: np.ndarray) -> np.ndarray:
    depth_m = depth_mm.astype(np.float32) / 1000.0
    clamped = np.clip(depth_m, 0.0, 4.0)
    if clamped.max() > 0:
        norm = (clamped / clamped.max() * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(depth_mm, dtype=np.uint8)
    viz = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    viz[depth_mm == 0] = 0
    return viz


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    print("Starting CameraManager...")
    cam = CameraManager(config)
    try:
        cam.start()
    except Exception as exc:
        print(f"ERROR: camera failed to start: {exc}", file=sys.stderr)
        return 1

    try:
        print("Loading FoodDetector (first run downloads + exports YOLO, ~30-60s)...")
        detector = FoodDetector(config)
        try:
            detector.load()
        except Exception as exc:
            print(f"ERROR: detector load failed: {exc}", file=sys.stderr)
            print(
                "Hint: first run needs internet to download yolov8n.pt "
                "from Ultralytics. Run `ping ultralytics.com` to check.",
                file=sys.stderr,
            )
            return 1

        if args.target:
            detector.set_target(args.target)

        print(f"Warming up camera for {args.frames} frames...")
        color, depth = grab_stable_frame(cam, args.frames)
        if color is None or depth is None:
            print("ERROR: no camera frame arrived.", file=sys.stderr)
            return 1

        print(f"Frame: {color.shape} {color.dtype} / depth {depth.shape} {depth.dtype}")

        print("Running YOLO inference...")
        t0 = time.time()
        detections = detector.detect(color)
        dt = time.time() - t0
        print(f"Inference time: {dt * 1000:.1f} ms")
        print(f"Detections: {len(detections)}")
        if args.target:
            print(f"Target set to: {args.target!r}")

        for det in detections:
            print(describe_detection(det, depth))

        if not detections:
            print(
                "  (nothing matched the food+kitchen class filter — this is "
                "normal if the camera isn't pointed at an apple, mug, bottle, etc.)"
            )

        args.outdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")

        annotated = annotate(color, detections, depth)
        ann_path = args.outdir / f"detect_color_{stamp}.png"
        cv2.imwrite(str(ann_path), annotated)
        print(f"Saved {ann_path}")

        depth_path = args.outdir / f"detect_depth_{stamp}.png"
        cv2.imwrite(str(depth_path), colorise_depth(depth))
        print(f"Saved {depth_path}")

        print("Level 2 detector test: OK")
        return 0
    finally:
        cam.stop()


if __name__ == "__main__":
    sys.exit(main())
