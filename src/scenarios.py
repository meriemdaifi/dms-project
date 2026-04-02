"""
Test Scenarios for DMS Validation
===================================
Defines simulation scenarios for validating the DMS system:
- Attentive driver
- Fatigued driver
- Distracted driver
"""

import numpy as np
from typing import List


def generate_attentive_scenario(
    duration_seconds: float = 120.0, fps: float = 30.0
) -> List[dict]:
    """
    Generate an attentive driver scenario.

    The driver maintains focus with:
    - Eyes consistently open (occasional normal blinks)
    - Head facing forward with minor natural movements
    - Normal blink pattern (~15-20 blinks/min)

    Args:
        duration_seconds: Scenario duration.
        fps: Frames per second.

    Returns:
        List of frame data dicts.
    """
    n_frames = int(duration_seconds * fps)
    data = []

    blink_interval = fps * 3  # Blink every ~3 seconds (20 blinks/min)
    blink_duration = int(fps * 0.15)  # 150ms blink

    for i in range(n_frames):
        # Normal blink pattern
        cycle_pos = i % int(blink_interval)
        eyes_open = cycle_pos >= blink_duration

        # Small natural head movements
        t = i / fps
        yaw = 3.0 * np.sin(0.1 * t) + np.random.normal(0, 1.5)
        pitch = 2.0 * np.sin(0.15 * t) + np.random.normal(0, 1.0)

        data.append(
            {
                "eyes_open": eyes_open,
                "yaw": np.clip(yaw, -15, 15),
                "pitch": np.clip(pitch, -10, 10),
                "looking_forward": True,
            }
        )

    return data


def generate_fatigued_scenario(
    duration_seconds: float = 120.0, fps: float = 30.0
) -> List[dict]:
    """
    Generate a fatigued driver scenario.

    The driver shows progressive fatigue:
    - Phase 1 (0-30s): Normal, slightly tired
    - Phase 2 (30-60s): Increasing blink duration, occasional microsleeps
    - Phase 3 (60-90s): Frequent long eye closures
    - Phase 4 (90-120s): Extended eye closures, near-drowsy state

    Args:
        duration_seconds: Scenario duration.
        fps: Frames per second.

    Returns:
        List of frame data dicts.
    """
    n_frames = int(duration_seconds * fps)
    data = []

    for i in range(n_frames):
        t = i / fps
        progress = t / duration_seconds  # 0 to 1

        # Progressive fatigue: eye closure increases over time
        if progress < 0.25:
            # Phase 1: Normal with slightly longer blinks
            blink_interval = fps * 3
            blink_duration = int(fps * 0.2)
            cycle_pos = i % int(blink_interval)
            eyes_open = cycle_pos >= blink_duration
        elif progress < 0.5:
            # Phase 2: Longer blinks, occasional microsleeps
            blink_interval = fps * 2.5
            blink_duration = int(fps * 0.4)
            cycle_pos = i % int(blink_interval)
            eyes_open = cycle_pos >= blink_duration
            # Add microsleeps (1-2s closures)
            if (i % int(fps * 15)) < int(fps * 1.5):
                eyes_open = False
        elif progress < 0.75:
            # Phase 3: Frequent long closures
            blink_interval = fps * 2
            blink_duration = int(fps * 0.6)
            cycle_pos = i % int(blink_interval)
            eyes_open = cycle_pos >= blink_duration
            # More frequent microsleeps
            if (i % int(fps * 10)) < int(fps * 2):
                eyes_open = False
        else:
            # Phase 4: Extended closures
            blink_interval = fps * 1.5
            blink_duration = int(fps * 0.8)
            cycle_pos = i % int(blink_interval)
            eyes_open = cycle_pos >= blink_duration
            # Extended microsleeps
            if (i % int(fps * 8)) < int(fps * 3):
                eyes_open = False

        # Head tends to droop with fatigue
        base_pitch = -5.0 * progress  # Head tilts down
        yaw = 2.0 * np.sin(0.05 * t) + np.random.normal(0, 2.0 + 3.0 * progress)
        pitch = base_pitch + np.random.normal(0, 2.0 + 2.0 * progress)

        data.append(
            {
                "eyes_open": eyes_open,
                "yaw": np.clip(yaw, -20, 20),
                "pitch": np.clip(pitch, -20, 10),
                "looking_forward": abs(yaw) < 20 and abs(pitch) < 15,
            }
        )

    return data


def generate_distracted_scenario(
    duration_seconds: float = 120.0, fps: float = 30.0
) -> List[dict]:
    """
    Generate a distracted driver scenario.

    The driver frequently looks away from the road:
    - Phase 1 (0-30s): Normal driving
    - Phase 2 (30-60s): Occasional glances to side (phone check)
    - Phase 3 (60-90s): Frequent and prolonged looking away
    - Phase 4 (90-120s): Sustained distraction (texting/looking down)

    Args:
        duration_seconds: Scenario duration.
        fps: Frames per second.

    Returns:
        List of frame data dicts.
    """
    n_frames = int(duration_seconds * fps)
    data = []

    for i in range(n_frames):
        t = i / fps
        progress = t / duration_seconds

        # Eyes generally open (driver is awake but not paying attention)
        blink_interval = fps * 4
        blink_duration = int(fps * 0.15)
        cycle_pos = i % int(blink_interval)
        eyes_open = cycle_pos >= blink_duration

        if progress < 0.25:
            # Phase 1: Normal driving
            yaw = 3.0 * np.sin(0.1 * t) + np.random.normal(0, 2.0)
            pitch = 2.0 * np.sin(0.15 * t) + np.random.normal(0, 1.5)
        elif progress < 0.5:
            # Phase 2: Occasional glances to the side
            if (i % int(fps * 10)) < int(fps * 3):
                yaw = 40.0 + np.random.normal(0, 5.0)  # Looking right
                pitch = -10.0 + np.random.normal(0, 3.0)  # Looking down slightly
            else:
                yaw = np.random.normal(0, 3.0)
                pitch = np.random.normal(0, 2.0)
        elif progress < 0.75:
            # Phase 3: Frequent looking away
            if (i % int(fps * 7)) < int(fps * 4):
                side = 1 if (i // int(fps * 7)) % 2 == 0 else -1
                yaw = side * 50.0 + np.random.normal(0, 5.0)
                pitch = -15.0 + np.random.normal(0, 3.0)
            else:
                yaw = np.random.normal(0, 5.0)
                pitch = np.random.normal(0, 3.0)
        else:
            # Phase 4: Sustained distraction (looking down at phone)
            yaw = 30.0 * np.sin(0.3 * t) + np.random.normal(0, 8.0)
            pitch = -25.0 + np.random.normal(0, 5.0)

        looking_forward = abs(yaw) < 20 and abs(pitch) < 15

        data.append(
            {
                "eyes_open": eyes_open,
                "yaw": float(np.clip(yaw, -90, 90)),
                "pitch": float(np.clip(pitch, -45, 45)),
                "looking_forward": looking_forward,
            }
        )

    return data


def generate_mixed_scenario(
    duration_seconds: float = 180.0, fps: float = 30.0
) -> List[dict]:
    """
    Generate a mixed scenario with transitions between states.

    - 0-60s: Attentive driving
    - 60-120s: Gradual fatigue onset
    - 120-150s: Distraction (phone usage)
    - 150-180s: Return to attention

    Args:
        duration_seconds: Scenario duration.
        fps: Frames per second.

    Returns:
        List of frame data dicts.
    """
    n_frames = int(duration_seconds * fps)
    data = []

    for i in range(n_frames):
        t = i / fps

        if t < 60:
            # Attentive
            blink_interval = fps * 3
            blink_duration = int(fps * 0.15)
            cycle_pos = i % int(blink_interval)
            eyes_open = cycle_pos >= blink_duration
            yaw = np.random.normal(0, 2.0)
            pitch = np.random.normal(0, 1.5)
        elif t < 120:
            # Progressive fatigue
            fatigue_progress = (t - 60) / 60
            blink_interval = fps * (3 - 1.5 * fatigue_progress)
            blink_duration = int(fps * (0.15 + 0.5 * fatigue_progress))
            cycle_pos = i % max(1, int(blink_interval))
            eyes_open = cycle_pos >= blink_duration
            if fatigue_progress > 0.5 and (i % int(fps * 10)) < int(fps * 2):
                eyes_open = False
            yaw = np.random.normal(0, 3.0 + 4.0 * fatigue_progress)
            pitch = -5.0 * fatigue_progress + np.random.normal(0, 2.0)
        elif t < 150:
            # Distraction
            eyes_open = True
            if (i % int(fps * 8)) < int(fps * 4):
                yaw = 45.0 + np.random.normal(0, 5.0)
                pitch = -20.0 + np.random.normal(0, 3.0)
            else:
                yaw = np.random.normal(0, 3.0)
                pitch = np.random.normal(0, 2.0)
        else:
            # Return to attention
            recovery = (t - 150) / 30
            blink_interval = fps * 3
            blink_duration = int(fps * 0.15)
            cycle_pos = i % int(blink_interval)
            eyes_open = cycle_pos >= blink_duration
            yaw = np.random.normal(0, 2.0 + 3.0 * (1 - recovery))
            pitch = np.random.normal(0, 1.5 + 2.0 * (1 - recovery))

        looking_forward = abs(yaw) < 20 and abs(pitch) < 15

        data.append(
            {
                "eyes_open": bool(eyes_open),
                "yaw": float(np.clip(yaw, -90, 90)),
                "pitch": float(np.clip(pitch, -45, 45)),
                "looking_forward": looking_forward,
            }
        )

    return data
