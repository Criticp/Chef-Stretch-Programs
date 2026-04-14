"""
Arm controller module — stateless arm positioning math.

Computes target joint positions for the lift, arm extension, and wrist
to position the gripper above a detected object. Returns MotorCommand
lists — does NOT call stretch_body directly.
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MotorCommand:
    """
    A single motor command to be executed on the main thread.

    joint: one of "base_translate", "base_rotate", "lift", "arm",
           "head_pan", "head_tilt", "wrist_yaw", "wrist_pitch",
           "wrist_roll", "gripper"
    value: target position (absolute) or displacement (relative)
    is_relative: True for base_translate/base_rotate, False for absolute targets
    """
    joint: str
    value: float
    is_relative: bool = False


class ArmController:
    """Computes arm/lift/wrist commands from 3D target positions."""

    def __init__(self, config: dict):
        arm_cfg = config["arm"]
        self._height_above = arm_cfg["height_above_object_m"]
        self._max_extension = arm_cfg["max_extension_m"]
        self._min_lift = arm_cfg["min_lift_m"]
        self._max_lift = arm_cfg["max_lift_m"]
        self._stow_lift = arm_cfg["stow_lift_m"]
        self._stow_arm = arm_cfg["stow_arm_m"]
        self._stow_wrist_yaw = arm_cfg.get("stow_wrist_yaw", 0.0)
        self._stow_wrist_pitch = arm_cfg.get("stow_wrist_pitch", -0.5)
        self._stow_wrist_roll = arm_cfg.get("stow_wrist_roll", 0.0)
        self._search_lift = arm_cfg.get("search_lift_m", 0.8)
        self._reach_wrist_pitch = arm_cfg.get("reach_wrist_pitch", -1.57)

    def position_above(self, target_3d: np.ndarray) -> List[MotorCommand]:
        """
        Compute commands to position the gripper above a detected object.

        target_3d: [x_forward, y_left, z_height] in robot base frame (meters).

        The lift moves to target_z + height_above.
        The arm extends to target_x (clamped to max_extension).
        The wrist pitches down to point at the object.

        NOTE: This assumes the robot is already facing the object (y ~= 0).
        The approach phase should have aligned the robot to face the target.
        """
        x_forward = float(target_3d[0])
        z_height = float(target_3d[2])

        # Compute lift height: object height + offset above.
        lift_target = z_height + self._height_above
        lift_target = np.clip(lift_target, self._min_lift, self._max_lift)

        # Compute arm extension: how far forward the object is.
        # The arm extends from the base, so extension = x_forward minus the
        # base-to-shoulder offset (~0.05m, absorbed into the target).
        arm_target = max(0.0, x_forward - 0.05)
        arm_target = min(arm_target, self._max_extension)

        commands = [
            MotorCommand("lift", lift_target),
            MotorCommand("arm", arm_target),
            MotorCommand("wrist_yaw", 0.0),
            MotorCommand("wrist_pitch", self._reach_wrist_pitch),
            MotorCommand("wrist_roll", 0.0),
            MotorCommand("gripper", 50),  # Open gripper
        ]

        logger.info(
            "Arm position_above: lift=%.2f arm=%.2f (target_3d=[%.2f, %.2f, %.2f])",
            lift_target,
            arm_target,
            target_3d[0],
            target_3d[1],
            target_3d[2],
        )
        return commands

    def stow(self) -> List[MotorCommand]:
        """Return commands to stow the arm in a safe, compact position."""
        commands = [
            MotorCommand("arm", self._stow_arm),
            MotorCommand("lift", self._stow_lift),
            MotorCommand("wrist_yaw", self._stow_wrist_yaw),
            MotorCommand("wrist_pitch", self._stow_wrist_pitch),
            MotorCommand("wrist_roll", self._stow_wrist_roll),
            MotorCommand("gripper", 50),  # Open
        ]
        logger.info("Arm stow commanded")
        return commands

    def search_pose(self) -> List[MotorCommand]:
        """
        Return commands to put the arm in a safe search posture.
        Arm retracted, lift at a comfortable height.
        """
        commands = [
            MotorCommand("arm", 0.0),
            MotorCommand("lift", self._search_lift),
            MotorCommand("wrist_yaw", 0.0),
            MotorCommand("wrist_pitch", self._stow_wrist_pitch),
            MotorCommand("wrist_roll", 0.0),
        ]
        logger.info("Arm search_pose commanded")
        return commands
