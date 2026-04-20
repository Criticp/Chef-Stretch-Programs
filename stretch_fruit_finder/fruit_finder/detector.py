"""
Detector module — YOLOv8 food and kitchen object detection.

Handles:
- Model loading (PyTorch .pt or OpenVINO exported model)
- One-time OpenVINO export for faster CPU inference
- Filtering detections to food and kitchen COCO classes
- Target matching (which food the user is looking for)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional
import threading

import numpy as np

logger = logging.getLogger(__name__)

# COCO class names for the food and kitchen IDs we care about.
COCO_FOOD_NAMES = {
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
}

COCO_KITCHEN_NAMES = {
    39: "bottle",
    41: "cup",
    43: "knife",
    44: "spoon",
    45: "bowl",
    60: "dining table",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
}


@dataclass
class Detection:
    """A single detected object."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    class_id: int
    is_food: bool = False
    is_kitchen: bool = False
    is_target: bool = False
    position_3d: Optional[np.ndarray] = None   # Set by CameraThread after detection

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class DetectionResult:
    """Thread-safe container for the latest detection results."""
    detections: List[Detection] = field(default_factory=list)
    frame_id: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, detections: List[Detection]) -> None:
        with self.lock:
            self.detections = detections
            self.frame_id += 1

    def get(self) -> List[Detection]:
        """Return a copy of the current detections."""
        with self.lock:
            return list(self.detections)

    def get_targets(self) -> List[Detection]:
        """Return only detections that match the current target."""
        with self.lock:
            return [d for d in self.detections if d.is_target]

    def get_best_target(self) -> Optional[Detection]:
        """Return the highest-confidence target detection."""
        targets = self.get_targets()
        if not targets:
            return None
        return max(targets, key=lambda d: d.confidence)


class FoodDetector:
    """YOLOv8-based food and kitchen object detector."""

    def __init__(self, config: dict):
        det_cfg = config["detection"]
        self._model_path = det_cfg["model"]
        self._openvino_path = det_cfg.get("openvino_model", "yolov8n_openvino_model")
        self._use_openvino = det_cfg.get("use_openvino", True)
        self._confidence = det_cfg["confidence"]
        self._food_classes = set(det_cfg["food_classes"])
        self._kitchen_classes = set(det_cfg["kitchen_classes"])
        self._all_classes = list(self._food_classes | self._kitchen_classes)

        self._target_label: Optional[str] = None
        self._model = None

    def load(self) -> None:
        """
        Load the YOLO model.

        If use_openvino is true, checks for an existing OpenVINO export.
        If not found, loads the .pt model, exports to OpenVINO, then reloads.
        """
        from ultralytics import YOLO

        if self._use_openvino and os.path.isdir(self._openvino_path):
            logger.info("Loading OpenVINO model from %s", self._openvino_path)
            self._model = YOLO(self._openvino_path, task="detect")
        elif self._use_openvino:
            logger.info(
                "OpenVINO model not found at %s — exporting from %s (one-time)",
                self._openvino_path,
                self._model_path,
            )
            pt_model = YOLO(self._model_path)
            pt_model.export(format="openvino")
            # The export creates a directory like yolov8n_openvino_model/
            if os.path.isdir(self._openvino_path):
                logger.info("Export complete — loading OpenVINO model")
                self._model = YOLO(self._openvino_path, task="detect")
            else:
                logger.warning(
                    "OpenVINO export directory not found at expected path %s — "
                    "falling back to PyTorch model",
                    self._openvino_path,
                )
                self._model = pt_model
        else:
            logger.info("Loading PyTorch model from %s", self._model_path)
            self._model = YOLO(self._model_path)

        logger.info("YOLO model loaded successfully")

    def set_target(self, label) -> None:
        """
        Set the food item name to search for.

        Pass a specific label (e.g. 'apple', 'banana') to target only that
        label. Pass None, an empty string, or 'any' / '(any)' / 'any food'
        to treat EVERY food-class detection as a target (used by the
        sweep's "any food" mode, which narrows to a specific label after
        the first acquisition).
        """
        if label is None:
            self._target_label = None
            logger.info("Detection target cleared")
            return
        normalised = str(label).strip().lower()
        if normalised in ("", "any", "any food", "(any)", "(any food)", "*"):
            # Empty-string sentinel means "any food-class detection counts".
            self._target_label = ""
            logger.info("Detection target: (any food)")
        else:
            self._target_label = normalised
            logger.info("Detection target: %r", self._target_label)

    @property
    def target_label(self) -> Optional[str]:
        return self._target_label

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run YOLO inference on a BGR frame.

        Returns a list of Detection objects filtered to food and kitchen classes.
        Each detection is tagged with is_food, is_kitchen, and is_target flags.
        """
        if self._model is None:
            return []

        results = self._model.predict(
            source=frame_bgr,
            conf=self._confidence,
            classes=self._all_classes,
            verbose=False,
        )

        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label = self._model.names[class_id].lower()

            is_food = class_id in self._food_classes
            is_kitchen = class_id in self._kitchen_classes
            if is_food and self._target_label is not None:
                if self._target_label == "":
                    # "any food" mode — every food detection is a candidate
                    is_target = True
                else:
                    is_target = self._target_label in label
            else:
                is_target = False

            det = Detection(
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                label=label,
                confidence=conf,
                class_id=class_id,
                is_food=is_food,
                is_kitchen=is_kitchen,
                is_target=is_target,
            )
            detections.append(det)

        return detections
