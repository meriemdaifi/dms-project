"""
Video Acquisition Module
========================
Handles video capture from camera or file sources.
Provides frame-by-frame access with configurable resolution and FPS.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator


class VideoAcquisition:
    """Manages video input from camera or file for the DMS pipeline."""

    def __init__(
        self,
        source: int | str = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
    ):
        """
        Initialize video acquisition.

        Args:
            source: Camera index (int) or video file path (str).
            resolution: Desired frame resolution (width, height).
            fps: Desired frames per second.
        """
        self.source = source
        self.resolution = resolution
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open the video source. Returns True if successful."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame. Returns None if no frame available."""
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames from the video source."""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def release(self) -> None:
        """Release the video capture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def is_opened(self) -> bool:
        """Check if the video source is currently open."""
        return self.cap is not None and self.cap.isOpened()

    def get_properties(self) -> dict:
        """Return current video properties."""
        if self.cap is None:
            return {}
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


def generate_synthetic_frame(
    width: int = 640,
    height: int = 480,
    face_present: bool = True,
    eyes_open: bool = True,
    head_angle: float = 0.0,
) -> np.ndarray:
    """
    Generate a synthetic frame for testing purposes.

    This creates a simple simulated frame with an oval face,
    eyes (open or closed), and a rotated head pose indicator.

    Args:
        width: Frame width.
        height: Frame height.
        face_present: Whether to draw a face.
        eyes_open: Whether eyes are open.
        head_angle: Head rotation angle in degrees.

    Returns:
        Synthetic BGR frame as numpy array.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (180, 180, 180)  # Light gray background

    if face_present:
        cx, cy = width // 2, height // 2
        # Draw face oval
        cv2.ellipse(frame, (cx, cy), (80, 110), head_angle, 0, 360, (200, 180, 160), -1)
        cv2.ellipse(frame, (cx, cy), (80, 110), head_angle, 0, 360, (100, 80, 60), 2)

        # Compute eye positions with head angle offset
        angle_rad = np.radians(head_angle)
        left_eye_x = int(cx - 30 * np.cos(angle_rad))
        left_eye_y = int(cy - 20 + 30 * np.sin(angle_rad))
        right_eye_x = int(cx + 30 * np.cos(angle_rad))
        right_eye_y = int(cy - 20 - 30 * np.sin(angle_rad))

        if eyes_open:
            # Open eyes - circles
            cv2.circle(frame, (left_eye_x, left_eye_y), 12, (255, 255, 255), -1)
            cv2.circle(frame, (right_eye_x, right_eye_y), 12, (255, 255, 255), -1)
            cv2.circle(frame, (left_eye_x, left_eye_y), 6, (80, 50, 30), -1)
            cv2.circle(frame, (right_eye_x, right_eye_y), 6, (80, 50, 30), -1)
        else:
            # Closed eyes - horizontal lines
            cv2.line(
                frame,
                (left_eye_x - 10, left_eye_y),
                (left_eye_x + 10, left_eye_y),
                (80, 50, 30),
                2,
            )
            cv2.line(
                frame,
                (right_eye_x - 10, right_eye_y),
                (right_eye_x + 10, right_eye_y),
                (80, 50, 30),
                2,
            )

        # Mouth
        mouth_y = cy + 40
        cv2.ellipse(frame, (cx, mouth_y), (25, 10), 0, 0, 180, (100, 60, 60), 2)

    return frame
