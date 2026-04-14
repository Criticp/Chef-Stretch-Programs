"""
Camera module — pyrealsense2 wrapper for the D435i head camera.

Handles:
- Frame capture (color + aligned depth)
- Pixel-to-3D deprojection
- Camera-frame to robot-base-frame transform using head pan/tilt
- Optional 90-degree image rotation (if camera is mounted rotated)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Thread-safe container for the latest camera frames."""
    color: Optional[np.ndarray] = None       # BGR uint8 (H, W, 3)
    depth: Optional[np.ndarray] = None       # uint16 millimeters (H, W)
    timestamp: float = 0.0
    frame_id: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, color: np.ndarray, depth: np.ndarray) -> None:
        with self.lock:
            self.color = color
            self.depth = depth
            self.timestamp = time.time()
            self.frame_id += 1

    def get(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """Return (color, depth, frame_id) — copies to avoid races."""
        with self.lock:
            if self.color is None:
                return None, None, self.frame_id
            return self.color.copy(), self.depth.copy(), self.frame_id


class CameraManager:
    """Manages the Intel RealSense D435i pipeline and 3D projection."""

    def __init__(self, config: dict):
        cam_cfg = config["camera"]
        self._width = cam_cfg["width"]
        self._height = cam_cfg["height"]
        self._fps = cam_cfg["fps"]
        self._depth_scale = cam_cfg["depth_scale"]
        self._min_depth = cam_cfg["min_depth_m"]
        self._max_depth = cam_cfg["max_depth_m"]
        self._rotated = cam_cfg.get("rotated_90", False)
        self._head_height = config["robot"]["head_camera_height_m"]

        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._intrinsics: Optional[rs.intrinsics] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the RealSense pipeline and extract camera intrinsics."""
        self._pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(
            rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps
        )
        rs_config.enable_stream(
            rs.stream.depth, self._width, self._height, rs.format.z16, self._fps
        )

        profile = self._pipeline.start(rs_config)
        self._align = rs.align(rs.stream.color)

        # Extract intrinsics from the color stream.
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._intrinsics = color_profile.get_intrinsics()

        # If the camera is rotated 90 degrees, swap width/height in intrinsics.
        if self._rotated:
            intr = self._intrinsics
            # After rotation: new_width = old_height, new_height = old_width.
            # Principal point and focal length swap accordingly.
            intr.width, intr.height = intr.height, intr.width
            intr.fx, intr.fy = intr.fy, intr.fx
            intr.ppx, intr.ppy = intr.ppy, intr.ppx

        self._running = True
        logger.info(
            "Camera started: %dx%d @%dfps (rotated_90=%s)",
            self._width, self._height, self._fps, self._rotated,
        )

    def stop(self) -> None:
        """Stop the RealSense pipeline."""
        self._running = False
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        logger.info("Camera stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def intrinsics(self) -> Optional[rs.intrinsics]:
        return self._intrinsics

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Wait for the next aligned color + depth frame pair.

        Returns:
            (color_bgr, depth_mm) or (None, None) if no frame available.
            color_bgr: uint8 (H, W, 3) BGR image.
            depth_mm:  uint16 (H, W) depth in millimeters.
        """
        if not self._running or self._pipeline is None:
            return None, None

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            logger.warning("Camera frame timeout")
            return None, None

        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color = np.asanyarray(color_frame.get_data())   # BGR uint8
        depth = np.asanyarray(depth_frame.get_data())    # uint16 mm

        # Handle 90-degree rotation if the camera is mounted sideways.
        if self._rotated:
            color = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)
            depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)

        return color, depth

    # ------------------------------------------------------------------
    # Depth-to-3D projection
    # ------------------------------------------------------------------

    def pixel_to_3d_camera(
        self, px: int, py: int, depth_mm: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Deproject a pixel to a 3D point in the camera optical frame.

        Samples a region around (px, py) and takes the median depth to
        reject outliers (zero-depth holes, noise).

        Returns:
            np.ndarray [x_right, y_down, z_forward] in meters, or None.
        """
        if self._intrinsics is None:
            return None

        h, w = depth_mm.shape

        # Sample a 20x20 region centered on the pixel.
        half = 10
        y0 = max(0, py - half)
        y1 = min(h, py + half)
        x0 = max(0, px - half)
        x1 = min(w, px + half)
        region = depth_mm[y0:y1, x0:x1].astype(np.float32) * self._depth_scale

        # Filter out zeros and out-of-range values.
        valid = region[(region > self._min_depth) & (region < self._max_depth)]
        if len(valid) == 0:
            return None

        depth_m = float(np.median(valid))
        point = rs.rs2_deproject_pixel_to_point(self._intrinsics, [px, py], depth_m)
        return np.array(point, dtype=np.float64)

    def camera_to_robot_frame(
        self,
        point_cam: np.ndarray,
        head_pan: float,
        head_tilt: float,
    ) -> np.ndarray:
        """
        Transform a 3D point from camera optical frame to robot base frame.

        Camera optical frame (OpenCV): x=right, y=down, z=forward
        Robot base frame:              x=forward, y=left, z=up

        Steps:
            1. Fixed rotation: camera optical -> head link
            2. Apply head tilt (rotation around Y in head frame)
            3. Apply head pan  (rotation around Z in head frame)
            4. Add head height offset
        """
        # Step 1: camera optical frame -> head link frame.
        # x_head =  z_cam  (forward)
        # y_head = -x_cam  (left)
        # z_head = -y_cam  (up)
        r_cam_to_head = np.array([
            [0.0,  0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        point_head = r_cam_to_head @ point_cam

        # Step 2: apply head tilt (rotation around Y axis).
        ct = np.cos(head_tilt)
        st = np.sin(head_tilt)
        r_tilt = np.array([
            [ct,  0.0, st],
            [0.0, 1.0, 0.0],
            [-st, 0.0, ct],
        ])

        # Step 3: apply head pan (rotation around Z axis).
        cp = np.cos(head_pan)
        sp = np.sin(head_pan)
        r_pan = np.array([
            [cp, -sp, 0.0],
            [sp,  cp, 0.0],
            [0.0, 0.0, 1.0],
        ])

        point_base = r_pan @ (r_tilt @ point_head)

        # Step 4: add head height offset.
        point_base[2] += self._head_height

        return point_base

    def pixel_to_robot_frame(
        self,
        px: int,
        py: int,
        depth_mm: np.ndarray,
        head_pan: float,
        head_tilt: float,
    ) -> Optional[np.ndarray]:
        """
        Full pipeline: pixel (px, py) + depth -> robot base frame [x, y, z].

        Returns None if depth is invalid at that pixel.
        """
        point_cam = self.pixel_to_3d_camera(px, py, depth_mm)
        if point_cam is None:
            return None
        return self.camera_to_robot_frame(point_cam, head_pan, head_tilt)


class CameraThread(threading.Thread):
    """
    Background thread that continuously captures frames from the camera
    and optionally runs YOLO detection on every Nth frame.
    """

    def __init__(
        self,
        camera: CameraManager,
        frame_data: FrameData,
        detector=None,
        detection_result=None,
        head_state_fn=None,
        infer_every: int = 3,
    ):
        super().__init__(daemon=True, name="CameraThread")
        self._camera = camera
        self._frame_data = frame_data
        self._detector = detector
        self._detection_result = detection_result
        self._head_state_fn = head_state_fn  # callable returning (pan, tilt)
        self._infer_every = max(1, infer_every)
        self._stop_event = threading.Event()
        self._frame_count = 0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info("CameraThread started")
        while not self._stop_event.is_set():
            color, depth = self._camera.get_frames()
            if color is None:
                time.sleep(0.01)
                continue

            self._frame_data.update(color, depth)
            self._frame_count += 1

            # Run detection every N frames.
            if (
                self._detector is not None
                and self._detection_result is not None
                and self._frame_count % self._infer_every == 0
            ):
                detections = self._detector.detect(color)

                # Compute 3D positions for each detection.
                head_pan, head_tilt = 0.0, 0.0
                if self._head_state_fn is not None:
                    try:
                        head_pan, head_tilt = self._head_state_fn()
                    except Exception:
                        pass

                for det in detections:
                    cx = (det.x1 + det.x2) // 2
                    cy = (det.y1 + det.y2) // 2
                    pos = self._camera.pixel_to_robot_frame(
                        cx, cy, depth, head_pan, head_tilt
                    )
                    det.position_3d = pos

                self._detection_result.update(detections)

        logger.info("CameraThread stopped")
