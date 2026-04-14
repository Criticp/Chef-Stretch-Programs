"""
Level 1 bringup test — camera only, no AI, no motion.

Opens the Stretch D435i head camera via `fruit_finder.camera.CameraManager`,
grabs a handful of aligned color+depth frames, prints their shape / dtype /
depth range, and saves the latest pair to disk so you can eyeball them.

This script does NOT touch stretch_body. It does NOT move the robot. It only
verifies that the RealSense pipeline opens, frames arrive, and the camera
wrapper in fruit_finder/camera.py behaves.

Run from the repo root on the robot NUC:

    cd ~/chef_ai/Chef-Stretch-Programs/stretch_fruit_finder
    python3 bringup/test_01_camera.py

Optional flags:
    --frames N       How many frames to grab before saving (default 10).
                     RealSense needs a few frames to stabilise auto-exposure.
    --outdir PATH    Where to save the PNGs (default ./bringup_out).
    --config PATH    Override config.yaml location.

Exit codes:
    0  success (frames captured and saved)
    1  camera failed to start, or no frame ever arrived
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

# Make `fruit_finder` importable whether this script is run as
# `python3 bringup/test_01_camera.py` from the package root or with an
# absolute path from somewhere else.
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from fruit_finder.camera import CameraManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of frames to grab before saving (default: 10).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=_HERE / "bringup_out",
        help="Directory to save PNGs into (default: ./bringup/bringup_out).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PKG_ROOT / "config.yaml",
        help="Path to config.yaml.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def describe_frame(color: np.ndarray, depth: np.ndarray, depth_scale: float) -> None:
    """Print shape, dtype, and depth range for one frame pair."""
    print(f"  color: shape={color.shape}, dtype={color.dtype}")
    print(f"  depth: shape={depth.shape}, dtype={depth.dtype}")

    valid = depth[depth > 0]
    if valid.size == 0:
        print("  depth: all zeros (camera may need to warm up or face a surface)")
        return

    # depth is uint16 in millimetres by default; depth_scale converts to metres.
    depth_m = valid.astype(np.float32) * depth_scale
    print(
        f"  depth valid pixels: {valid.size}/{depth.size} "
        f"({100.0 * valid.size / depth.size:.1f}%)"
    )
    print(
        f"  depth range: min={depth_m.min():.3f} m  "
        f"median={np.median(depth_m):.3f} m  max={depth_m.max():.3f} m"
    )


def save_outputs(
    color: np.ndarray,
    depth: np.ndarray,
    outdir: Path,
) -> None:
    """Save the color frame and a colorised depth visualization."""
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    color_path = outdir / f"color_{stamp}.png"
    cv2.imwrite(str(color_path), color)
    print(f"  saved {color_path}")

    # Colorise depth for a sanity-check visualization: clamp to a sensible
    # indoor range, normalise, apply a colormap, zero where no depth.
    depth_m = depth.astype(np.float32) / 1000.0  # uint16 mm -> float m
    depth_clamped = np.clip(depth_m, 0.0, 4.0)
    if depth_clamped.max() > 0:
        norm = (depth_clamped / depth_clamped.max() * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(depth, dtype=np.uint8)
    colorised = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    # Mask out pixels with no depth so they show black, not the low end of the map.
    colorised[depth == 0] = 0

    depth_path = outdir / f"depth_viz_{stamp}.png"
    cv2.imwrite(str(depth_path), colorised)
    print(f"  saved {depth_path}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    depth_scale = float(config["camera"]["depth_scale"])

    print("Starting CameraManager...")
    cam = CameraManager(config)
    try:
        cam.start()
    except Exception as exc:
        print(f"ERROR: camera failed to start: {exc}", file=sys.stderr)
        print(
            "Hint: is another process holding the RealSense? "
            "Run `rs-enumerate-devices` to confirm the device is visible.",
            file=sys.stderr,
        )
        return 1

    last_color: np.ndarray | None = None
    last_depth: np.ndarray | None = None
    got_frames = 0

    try:
        print(f"Grabbing {args.frames} frames (first few may be empty, that's normal)...")
        for i in range(args.frames):
            color, depth = cam.get_frames()
            if color is None or depth is None:
                print(f"  frame {i}: no data yet")
                continue
            got_frames += 1
            last_color, last_depth = color, depth
            if i == 0 or i == args.frames - 1:
                print(f"  frame {i}:")
                describe_frame(color, depth, depth_scale)
    finally:
        cam.stop()

    if last_color is None or last_depth is None:
        print(
            "ERROR: camera started but no frames arrived. "
            "Check the head camera USB connection.",
            file=sys.stderr,
        )
        return 1

    print(f"Captured {got_frames}/{args.frames} frames total.")
    print(f"Saving last good frame pair to {args.outdir}...")
    save_outputs(last_color, last_depth, args.outdir)
    print("Level 1 camera test: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
