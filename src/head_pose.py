"""
Head Pose Estimation Module
============================
Estimates the driver's head orientation (yaw, pitch, roll) using
facial landmarks and a Perspective-n-Point (PnP) solver.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class HeadPose:
    """Represents head orientation angles in degrees."""

    yaw: float  # Left/right rotation
    pitch: float  # Up/down rotation
    roll: float  # Tilt rotation
    is_looking_forward: bool  # Within acceptable thresholds

    @property
    def is_distracted(self) -> bool:
        """Check if head pose indicates distraction."""
        return abs(self.yaw) > 30 or abs(self.pitch) > 25

    @property
    def distraction_level(self) -> float:
        """Compute a normalized distraction level from 0 to 1."""
        yaw_norm = min(abs(self.yaw) / 90.0, 1.0)
        pitch_norm = min(abs(self.pitch) / 90.0, 1.0)
        return max(yaw_norm, pitch_norm)


# 3D model points for a generic face (nose tip, chin, left/right eye corners,
# left/right mouth corners) in a canonical coordinate system.
MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)


class HeadPoseEstimator:
    """
    Estimates head pose using 2D-3D point correspondences
    and OpenCV's solvePnP.
    """

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        yaw_threshold: float = 20.0,
        pitch_threshold: float = 15.0,
    ):
        """
        Initialize head pose estimator.

        Args:
            frame_width: Width of the input frame.
            frame_height: Height of the input frame.
            yaw_threshold: Maximum acceptable yaw angle (degrees).
            pitch_threshold: Maximum acceptable pitch angle (degrees).
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

        # Camera matrix approximation
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        self.camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    def estimate(
        self,
        landmarks_2d: np.ndarray,
    ) -> Optional[HeadPose]:
        """
        Estimate head pose from 2D facial landmarks.

        Args:
            landmarks_2d: Array of 6 (x, y) points corresponding to
                         nose tip, chin, left eye, right eye,
                         left mouth, right mouth.

        Returns:
            HeadPose object or None if estimation fails.
        """
        if landmarks_2d.shape != (6, 2):
            return None

        image_points = landmarks_2d.astype(np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix, then extract Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles = self._rotation_matrix_to_euler(rotation_matrix)

        yaw, pitch, roll = angles
        is_forward = abs(yaw) <= self.yaw_threshold and abs(pitch) <= self.pitch_threshold

        return HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            is_looking_forward=is_forward,
        )

    @staticmethod
    def _rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert a 3x3 rotation matrix to Euler angles (yaw, pitch, roll).

        Args:
            rotation_matrix: 3x3 rotation matrix.

        Returns:
            Tuple of (yaw, pitch, roll) in degrees.
        """
        sy = np.sqrt(
            rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2
        )
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0

        return (
            np.degrees(z),   # yaw
            np.degrees(x),   # pitch
            np.degrees(y),   # roll
        )

    def draw_pose_axes(
        self,
        frame: np.ndarray,
        landmarks_2d: np.ndarray,
        axis_length: float = 100.0,
    ) -> np.ndarray:
        """
        Draw 3D axes on the frame showing head orientation.

        Args:
            frame: Input BGR frame.
            landmarks_2d: 6 facial landmark points.
            axis_length: Length of drawn axes.

        Returns:
            Frame with axes drawn.
        """
        image_points = landmarks_2d.astype(np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return frame

        # Project 3D axis points
        axis_points = np.float64(
            [
                [axis_length, 0, 0],   # X axis (red)
                [0, axis_length, 0],    # Y axis (green)
                [0, 0, axis_length],    # Z axis (blue)
            ]
        )

        projected, _ = cv2.projectPoints(
            axis_points,
            rotation_vector,
            translation_vector,
            self.camera_matrix,
            self.dist_coeffs,
        )

        nose_point = tuple(image_points[0].astype(int))
        result = frame.copy()

        cv2.line(result, nose_point, tuple(projected[0].ravel().astype(int)), (0, 0, 255), 2)
        cv2.line(result, nose_point, tuple(projected[1].ravel().astype(int)), (0, 255, 0), 2)
        cv2.line(result, nose_point, tuple(projected[2].ravel().astype(int)), (255, 0, 0), 2)

        return result


def estimate_head_pose_simple(
    face_bbox: Tuple[int, int, int, int],
    left_eye: Optional[Tuple[int, int]],
    right_eye: Optional[Tuple[int, int]],
) -> HeadPose:
    """
    Simple head pose estimation based on face geometry.

    Uses the relative position of eyes within the face bounding box
    to approximate head orientation. Less accurate than PnP-based
    estimation but works without full facial landmarks.

    Args:
        face_bbox: Face bounding box (x, y, w, h).
        left_eye: Left eye center (x, y) or None.
        right_eye: Right eye center (x, y) or None.

    Returns:
        Estimated HeadPose.
    """
    x, y, w, h = face_bbox
    face_cx = x + w / 2
    face_cy = y + h / 2

    yaw = 0.0
    pitch = 0.0
    roll = 0.0

    if left_eye is not None and right_eye is not None:
        eye_cx = (left_eye[0] + right_eye[0]) / 2
        eye_cy = (left_eye[1] + right_eye[1]) / 2

        # Yaw estimation: horizontal offset of eyes from face center
        horizontal_offset = (eye_cx - face_cx) / (w / 2)
        yaw = horizontal_offset * 45  # Scale to approximate degrees

        # Pitch estimation: vertical offset
        expected_eye_y = face_cy - h * 0.15
        vertical_offset = (eye_cy - expected_eye_y) / (h / 2)
        pitch = vertical_offset * 30

        # Roll estimation: angle between eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        if dx != 0:
            roll = np.degrees(np.arctan2(dy, dx))

    is_forward = abs(yaw) <= 20 and abs(pitch) <= 15

    return HeadPose(yaw=yaw, pitch=pitch, roll=roll, is_looking_forward=is_forward)
