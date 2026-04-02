"""
Face and Eye Detection Module (CNN-based)
==========================================
Implements a CNN model for face detection and eye state classification.
Provides both a lightweight CNN architecture and integration with
OpenCV's DNN module for robust detection.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FaceDetection:
    """Represents a detected face with bounding box and landmarks."""

    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    nose: Optional[Tuple[int, int]] = None
    mouth_left: Optional[Tuple[int, int]] = None
    mouth_right: Optional[Tuple[int, int]] = None


@dataclass
class EyeState:
    """Represents the state of an eye."""

    is_open: bool
    openness_ratio: float  # 0.0 (closed) to 1.0 (fully open)
    confidence: float


class EyeStateCNN(nn.Module):
    """
    Convolutional Neural Network for eye state classification.

    Architecture:
        - 3 convolutional blocks with batch normalization and max pooling
        - 2 fully connected layers with dropout
        - Binary output: open (1) vs closed (0)

    Input: 24x24 grayscale eye region images.
    """

    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers (24x24 -> 12x12 -> 6x6 -> 3x3 after 3 pools)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 2)  # 2 classes: open, closed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class FaceDetectionCNN(nn.Module):
    """
    CNN for face detection confidence scoring.

    Takes a 64x64 grayscale image patch and outputs a confidence
    score indicating whether a face is present.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FaceDetector:
    """
    Face detector using OpenCV's Haar cascades and CNN refinement.

    Combines traditional cascade classifiers for initial detection
    with CNN-based confidence scoring for robust face detection.
    """

    def __init__(self, use_cnn: bool = True):
        """
        Initialize the face detector.

        Args:
            use_cnn: Whether to use CNN for confidence refinement.
        """
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.use_cnn = use_cnn

        if use_cnn:
            self.face_cnn = FaceDetectionCNN()
            self.face_cnn.eval()

    def detect_faces(
        self,
        frame: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (60, 60),
    ) -> List[FaceDetection]:
        """
        Detect faces in a frame.

        Args:
            frame: BGR input image.
            scale_factor: Cascade scale factor.
            min_neighbors: Minimum neighbors for cascade detection.
            min_size: Minimum face size.

        Returns:
            List of FaceDetection objects.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces_rect = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )

        detections = []
        for x, y, w, h in faces_rect:
            confidence = 1.0
            if self.use_cnn:
                confidence = self._cnn_confidence(gray, x, y, w, h)

            # Detect eyes within face region
            face_roi = gray[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(
                face_roi, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
            )

            left_eye = None
            right_eye = None
            if len(eyes) >= 2:
                # Sort by x coordinate to identify left/right
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                ex1, ey1, ew1, eh1 = eyes_sorted[0]
                ex2, ey2, ew2, eh2 = eyes_sorted[1]
                left_eye = (x + ex1 + ew1 // 2, y + ey1 + eh1 // 2)
                right_eye = (x + ex2 + ew2 // 2, y + ey2 + eh2 // 2)

            detection = FaceDetection(
                bbox=(x, y, w, h),
                confidence=confidence,
                left_eye=left_eye,
                right_eye=right_eye,
            )
            detections.append(detection)

        return detections

    def _cnn_confidence(
        self, gray: np.ndarray, x: int, y: int, w: int, h: int
    ) -> float:
        """Compute CNN-based face confidence for a detected region."""
        face_patch = gray[y : y + h, x : x + w]
        face_resized = cv2.resize(face_patch, (64, 64))
        tensor = torch.FloatTensor(face_resized).unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            output = self.face_cnn(tensor)
            prob = F.softmax(output, dim=1)
            return prob[0, 1].item()


class EyeStateClassifier:
    """
    Classifies whether eyes are open or closed using the EyeStateCNN.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the eye state classifier.

        Args:
            model_path: Path to a pre-trained model weights file.
        """
        self.model = EyeStateCNN()
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        self.eye_size = (24, 24)

    def classify(self, eye_region: np.ndarray) -> EyeState:
        """
        Classify the state of an eye region.

        Args:
            eye_region: Grayscale image of an eye region.

        Returns:
            EyeState with classification result.
        """
        if len(eye_region.shape) == 3:
            eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

        eye_resized = cv2.resize(eye_region, self.eye_size)
        tensor = torch.FloatTensor(eye_resized).unsqueeze(0).unsqueeze(0) / 255.0

        with torch.no_grad():
            output = self.model(tensor)
            prob = F.softmax(output, dim=1)
            is_open = prob[0, 1].item() > 0.5

        return EyeState(
            is_open=is_open,
            openness_ratio=prob[0, 1].item(),
            confidence=max(prob[0, 0].item(), prob[0, 1].item()),
        )

    def extract_eye_region(
        self,
        frame: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        eye_center: Tuple[int, int],
        eye_radius: int = 20,
    ) -> np.ndarray:
        """
        Extract an eye region from a frame.

        Args:
            frame: Input image.
            face_bbox: Face bounding box (x, y, w, h).
            eye_center: Center point of the eye.
            eye_radius: Radius around the eye center to extract.

        Returns:
            Cropped eye region image.
        """
        ex = max(0, eye_center[0] - eye_radius)
        ey = max(0, eye_center[1] - eye_radius)
        ew = min(frame.shape[1], eye_center[0] + eye_radius)
        eh = min(frame.shape[0], eye_center[1] + eye_radius)
        return frame[ey:eh, ex:ew]


def compute_eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) from eye landmarks.

    The EAR is a scalar value that indicates how open an eye is.
    When the eye is open, EAR is relatively large. When closed,
    it approaches zero.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        eye_landmarks: 6 (x, y) landmark points around the eye.

    Returns:
        Eye aspect ratio value.
    """
    # Vertical distances
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Horizontal distance
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return ear
