"""
Behavioral Analysis Module
===========================
Analyzes the driver's behavioral state based on detection outputs.
Computes fatigue indicators (PERCLOS, blink rate) and distraction metrics.
"""

import time
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class DriverState(Enum):
    """Enumeration of possible driver states."""

    ATTENTIVE = "attentive"
    FATIGUED = "fatigued"
    DISTRACTED = "distracted"
    DROWSY = "drowsy"  # Severe fatigue
    UNKNOWN = "unknown"


@dataclass
class BehavioralMetrics:
    """Container for behavioral analysis metrics."""

    # Eye-based metrics
    perclos: float = 0.0  # Percentage of eye closure (0-100)
    blink_rate: float = 0.0  # Blinks per minute
    avg_blink_duration: float = 0.0  # Average blink duration in seconds
    current_eye_closure_duration: float = 0.0  # Current closure duration

    # Head pose metrics
    avg_yaw: float = 0.0
    avg_pitch: float = 0.0
    head_movement_frequency: float = 0.0  # Movements per minute
    time_looking_away: float = 0.0  # Percentage of time not looking forward

    # Combined state
    driver_state: DriverState = DriverState.UNKNOWN
    fatigue_score: float = 0.0  # 0.0 (alert) to 1.0 (very fatigued)
    distraction_score: float = 0.0  # 0.0 (attentive) to 1.0 (very distracted)
    confidence: float = 0.0


@dataclass
class BlinkEvent:
    """Records a single blink event."""

    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class BehavioralAnalyzer:
    """
    Analyzes driver behavior over time using rolling windows.

    Tracks eye closure patterns, blink events, and head pose
    to compute fatigue and distraction scores.
    """

    # Thresholds for driver state classification
    PERCLOS_FATIGUE_THRESHOLD = 40.0  # % eye closure indicating fatigue
    PERCLOS_DROWSY_THRESHOLD = 70.0  # % eye closure indicating drowsiness
    BLINK_RATE_LOW = 10  # Blinks/min below this suggests fatigue
    BLINK_RATE_HIGH = 30  # Blinks/min above this suggests stress/fatigue
    LONG_BLINK_THRESHOLD = 0.4  # Seconds; blinks longer than this are fatigue indicators
    DISTRACTION_YAW_THRESHOLD = 25.0  # Degrees
    DISTRACTION_TIME_THRESHOLD = 30.0  # % time looking away

    def __init__(self, window_size: float = 60.0, sample_rate: float = 30.0):
        """
        Initialize the behavioral analyzer.

        Args:
            window_size: Analysis window in seconds.
            sample_rate: Expected samples per second (FPS).
        """
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.max_samples = int(window_size * sample_rate)

        # Rolling buffers
        self._eye_open_history: deque = deque(maxlen=self.max_samples)
        self._timestamps: deque = deque(maxlen=self.max_samples)
        self._yaw_history: deque = deque(maxlen=self.max_samples)
        self._pitch_history: deque = deque(maxlen=self.max_samples)
        self._looking_forward_history: deque = deque(maxlen=self.max_samples)

        # Blink tracking
        self._blink_events: List[BlinkEvent] = []
        self._eye_was_open: bool = True
        self._eye_closed_start: Optional[float] = None

    def update(
        self,
        eyes_open: bool,
        yaw: float = 0.0,
        pitch: float = 0.0,
        looking_forward: bool = True,
        timestamp: Optional[float] = None,
    ) -> BehavioralMetrics:
        """
        Update the analyzer with a new frame's data.

        Args:
            eyes_open: Whether the driver's eyes are open.
            yaw: Head yaw angle in degrees.
            pitch: Head pitch angle in degrees.
            looking_forward: Whether driver is looking forward.
            timestamp: Frame timestamp (defaults to current time).

        Returns:
            Updated BehavioralMetrics.
        """
        if timestamp is None:
            timestamp = time.time()

        # Record samples
        self._eye_open_history.append(eyes_open)
        self._timestamps.append(timestamp)
        self._yaw_history.append(yaw)
        self._pitch_history.append(pitch)
        self._looking_forward_history.append(looking_forward)

        # Track blinks
        self._track_blinks(eyes_open, timestamp)

        # Clean old blink events
        cutoff = timestamp - self.window_size
        self._blink_events = [b for b in self._blink_events if b.end_time > cutoff]

        # Compute metrics
        return self._compute_metrics(timestamp)

    def _track_blinks(self, eyes_open: bool, timestamp: float) -> None:
        """Track blink events based on eye state transitions."""
        if self._eye_was_open and not eyes_open:
            # Eye just closed - start of potential blink
            self._eye_closed_start = timestamp
        elif not self._eye_was_open and eyes_open and self._eye_closed_start is not None:
            # Eye just opened - end of blink
            blink = BlinkEvent(
                start_time=self._eye_closed_start,
                end_time=timestamp,
            )
            self._blink_events.append(blink)
            self._eye_closed_start = None

        self._eye_was_open = eyes_open

    def _compute_metrics(self, current_time: float) -> BehavioralMetrics:
        """Compute all behavioral metrics from the rolling buffers."""
        if len(self._eye_open_history) == 0:
            return BehavioralMetrics()

        # PERCLOS: percentage of time eyes are closed
        closed_count = sum(1 for e in self._eye_open_history if not e)
        total_count = len(self._eye_open_history)
        perclos = (closed_count / total_count) * 100.0

        # Blink rate (blinks per minute)
        window_duration = self._get_window_duration()
        if window_duration > 0:
            blink_rate = (len(self._blink_events) / window_duration) * 60.0
        else:
            blink_rate = 0.0

        # Average blink duration
        if self._blink_events:
            avg_blink_duration = np.mean([b.duration for b in self._blink_events])
        else:
            avg_blink_duration = 0.0

        # Current eye closure duration
        current_closure = 0.0
        if self._eye_closed_start is not None:
            current_closure = current_time - self._eye_closed_start

        # Head pose metrics
        avg_yaw = np.mean(list(self._yaw_history)) if self._yaw_history else 0.0
        avg_pitch = np.mean(list(self._pitch_history)) if self._pitch_history else 0.0

        # Time looking away
        if self._looking_forward_history:
            away_count = sum(1 for f in self._looking_forward_history if not f)
            time_looking_away = (away_count / len(self._looking_forward_history)) * 100.0
        else:
            time_looking_away = 0.0

        # Compute fatigue score
        fatigue_score = self._compute_fatigue_score(
            perclos, blink_rate, avg_blink_duration, current_closure
        )

        # Compute distraction score
        distraction_score = self._compute_distraction_score(
            avg_yaw, avg_pitch, time_looking_away
        )

        # Determine driver state
        driver_state = self._classify_state(fatigue_score, distraction_score)

        # Confidence based on sample count
        confidence = min(total_count / (self.sample_rate * 5), 1.0)  # Full confidence after 5s

        return BehavioralMetrics(
            perclos=perclos,
            blink_rate=blink_rate,
            avg_blink_duration=avg_blink_duration,
            current_eye_closure_duration=current_closure,
            avg_yaw=avg_yaw,
            avg_pitch=avg_pitch,
            time_looking_away=time_looking_away,
            driver_state=driver_state,
            fatigue_score=fatigue_score,
            distraction_score=distraction_score,
            confidence=confidence,
        )

    def _get_window_duration(self) -> float:
        """Get the time span of the current analysis window."""
        if len(self._timestamps) < 2:
            return 0.0
        return self._timestamps[-1] - self._timestamps[0]

    def _compute_fatigue_score(
        self,
        perclos: float,
        blink_rate: float,
        avg_blink_duration: float,
        current_closure: float,
    ) -> float:
        """
        Compute a normalized fatigue score from 0 to 1.

        Uses a weighted combination of PERCLOS, blink rate anomaly,
        blink duration, and current eye closure.
        """
        # PERCLOS component (0-1)
        perclos_component = min(perclos / 100.0, 1.0)

        # Blink rate anomaly (low or high blink rate indicates fatigue)
        if blink_rate < self.BLINK_RATE_LOW:
            blink_rate_component = 1.0 - (blink_rate / self.BLINK_RATE_LOW)
        elif blink_rate > self.BLINK_RATE_HIGH:
            blink_rate_component = min(
                (blink_rate - self.BLINK_RATE_HIGH) / self.BLINK_RATE_HIGH, 1.0
            )
        else:
            blink_rate_component = 0.0

        # Long blink component
        long_blink_component = min(avg_blink_duration / 1.0, 1.0)

        # Current closure component (prolonged closure)
        closure_component = min(current_closure / 3.0, 1.0)

        # Weighted combination
        fatigue_score = (
            0.4 * perclos_component
            + 0.2 * blink_rate_component
            + 0.2 * long_blink_component
            + 0.2 * closure_component
        )

        return min(max(fatigue_score, 0.0), 1.0)

    def _compute_distraction_score(
        self,
        avg_yaw: float,
        avg_pitch: float,
        time_looking_away: float,
    ) -> float:
        """
        Compute a normalized distraction score from 0 to 1.
        """
        # Yaw component
        yaw_component = min(abs(avg_yaw) / 90.0, 1.0)

        # Pitch component
        pitch_component = min(abs(avg_pitch) / 90.0, 1.0)

        # Time away component
        away_component = time_looking_away / 100.0

        # Weighted combination
        distraction_score = (
            0.4 * yaw_component + 0.2 * pitch_component + 0.4 * away_component
        )

        return min(max(distraction_score, 0.0), 1.0)

    def _classify_state(
        self, fatigue_score: float, distraction_score: float
    ) -> DriverState:
        """Classify the overall driver state."""
        if fatigue_score > 0.7:
            return DriverState.DROWSY
        elif fatigue_score > 0.4:
            return DriverState.FATIGUED
        elif distraction_score > 0.4:
            return DriverState.DISTRACTED
        else:
            return DriverState.ATTENTIVE

    def reset(self) -> None:
        """Reset all internal buffers and state."""
        self._eye_open_history.clear()
        self._timestamps.clear()
        self._yaw_history.clear()
        self._pitch_history.clear()
        self._looking_forward_history.clear()
        self._blink_events.clear()
        self._eye_was_open = True
        self._eye_closed_start = None
