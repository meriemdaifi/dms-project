"""
Unit Tests for the DMS System
==============================
Tests for behavioral analysis, ECU decision logic, scenarios,
and visualization modules.
"""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile

from src.behavioral_analysis import BehavioralAnalyzer, DriverState, BehavioralMetrics
from src.ecu_decision import (
    ECUDecisionEngine,
    ECUSimulator,
    AlertLevel,
    AlertType,
)
from src.scenarios import (
    generate_attentive_scenario,
    generate_fatigued_scenario,
    generate_distracted_scenario,
    generate_mixed_scenario,
)
from src.head_pose import HeadPose, HeadPoseEstimator, estimate_head_pose_simple
from src.visualization import create_scenario_dataframe, generate_summary_table


class TestBehavioralAnalyzer(unittest.TestCase):
    """Tests for the BehavioralAnalyzer class."""

    def setUp(self):
        self.analyzer = BehavioralAnalyzer(window_size=10.0, sample_rate=10.0)

    def test_initial_state(self):
        """Test initial metrics are zero/unknown."""
        metrics = self.analyzer.update(eyes_open=True, timestamp=0.0)
        self.assertEqual(metrics.perclos, 0.0)
        self.assertLess(metrics.fatigue_score, 0.5)

    def test_eyes_open_low_perclos(self):
        """Test that consistently open eyes result in low PERCLOS."""
        for i in range(100):
            metrics = self.analyzer.update(eyes_open=True, timestamp=i * 0.1)
        self.assertLess(metrics.perclos, 5.0)

    def test_eyes_closed_high_perclos(self):
        """Test that consistently closed eyes result in high PERCLOS."""
        for i in range(100):
            metrics = self.analyzer.update(eyes_open=False, timestamp=i * 0.1)
        self.assertGreater(metrics.perclos, 90.0)

    def test_fatigue_detection(self):
        """Test fatigue is detected with prolonged eye closure."""
        # First some open, then lots of closed
        for i in range(50):
            self.analyzer.update(eyes_open=True, timestamp=i * 0.1)
        for i in range(50, 200):
            metrics = self.analyzer.update(eyes_open=False, timestamp=i * 0.1)
        self.assertGreater(metrics.fatigue_score, 0.3)

    def test_blink_tracking(self):
        """Test that blinks are correctly tracked."""
        # Simulate 5 blinks (open->closed->open)
        t = 0.0
        for _ in range(5):
            for _ in range(10):  # Open for 1s
                self.analyzer.update(eyes_open=True, timestamp=t)
                t += 0.1
            for _ in range(2):  # Closed for 0.2s (blink)
                self.analyzer.update(eyes_open=False, timestamp=t)
                t += 0.1
        # Final open to register last blink end
        metrics = self.analyzer.update(eyes_open=True, timestamp=t)
        self.assertGreater(metrics.blink_rate, 0)

    def test_distraction_from_head_pose(self):
        """Test distraction detection from head pose."""
        for i in range(200):
            metrics = self.analyzer.update(
                eyes_open=True,
                yaw=45.0,
                pitch=0.0,
                looking_forward=False,
                timestamp=i * 0.1,
            )
        self.assertGreater(metrics.distraction_score, 0.3)

    def test_reset(self):
        """Test that reset clears all state."""
        for i in range(50):
            self.analyzer.update(eyes_open=False, timestamp=i * 0.1)
        self.analyzer.reset()
        metrics = self.analyzer.update(eyes_open=True, timestamp=100.0)
        self.assertEqual(metrics.perclos, 0.0)

    def test_driver_state_classification(self):
        """Test driver state is correctly classified."""
        # Attentive scenario
        analyzer = BehavioralAnalyzer(window_size=5.0, sample_rate=10.0)
        for i in range(50):
            metrics = analyzer.update(
                eyes_open=True, yaw=0.0, pitch=0.0,
                looking_forward=True, timestamp=i * 0.1
            )
        self.assertEqual(metrics.driver_state, DriverState.ATTENTIVE)


class TestECUDecisionEngine(unittest.TestCase):
    """Tests for the ECU decision engine."""

    def setUp(self):
        self.ecu = ECUDecisionEngine()

    def test_normal_operation(self):
        """Test ECU produces no alerts for normal metrics."""
        metrics = BehavioralMetrics(
            fatigue_score=0.1,
            distraction_score=0.1,
            driver_state=DriverState.ATTENTIVE,
        )
        state = self.ecu.process(metrics, timestamp=1.0)
        self.assertEqual(state.current_alert_level, AlertLevel.NONE)
        self.assertEqual(len(state.active_alerts), 0)

    def test_fatigue_warning(self):
        """Test ECU generates fatigue warning."""
        metrics = BehavioralMetrics(
            fatigue_score=0.45,
            distraction_score=0.1,
            driver_state=DriverState.FATIGUED,
        )
        state = self.ecu.process(metrics, timestamp=1.0)
        self.assertEqual(state.current_alert_level, AlertLevel.WARNING)
        self.assertTrue(len(state.active_alerts) > 0)
        self.assertEqual(
            state.active_alerts[-1].alert_type, AlertType.FATIGUE_WARNING
        )

    def test_drowsiness_emergency(self):
        """Test ECU generates drowsiness emergency."""
        metrics = BehavioralMetrics(
            fatigue_score=0.8,
            distraction_score=0.1,
            driver_state=DriverState.DROWSY,
        )
        state = self.ecu.process(metrics, timestamp=1.0)
        self.assertEqual(state.current_alert_level, AlertLevel.EMERGENCY)

    def test_distraction_warning(self):
        """Test ECU generates distraction warning."""
        metrics = BehavioralMetrics(
            fatigue_score=0.1,
            distraction_score=0.45,
            driver_state=DriverState.DISTRACTED,
        )
        state = self.ecu.process(metrics, timestamp=1.0)
        self.assertEqual(state.current_alert_level, AlertLevel.WARNING)

    def test_alert_cooldown(self):
        """Test alert cooldown prevents rapid re-alerting."""
        metrics = BehavioralMetrics(
            fatigue_score=0.5,
            distraction_score=0.1,
            driver_state=DriverState.FATIGUED,
        )
        self.ecu.process(metrics, timestamp=1.0)
        initial_count = self.ecu.state.total_alerts_generated

        # Process again within cooldown period
        self.ecu.process(metrics, timestamp=2.0)
        self.assertEqual(self.ecu.state.total_alerts_generated, initial_count)

        # Process after cooldown
        self.ecu.process(metrics, timestamp=7.0)
        self.assertGreater(self.ecu.state.total_alerts_generated, initial_count)

    def test_alert_callback(self):
        """Test alert callbacks are invoked."""
        alerts_received = []
        self.ecu.register_alert_callback(lambda a: alerts_received.append(a))

        metrics = BehavioralMetrics(
            fatigue_score=0.5,
            driver_state=DriverState.FATIGUED,
        )
        self.ecu.process(metrics, timestamp=1.0)
        self.assertEqual(len(alerts_received), 1)

    def test_acknowledge_alerts(self):
        """Test alert acknowledgement."""
        metrics = BehavioralMetrics(
            fatigue_score=0.5,
            driver_state=DriverState.FATIGUED,
        )
        self.ecu.process(metrics, timestamp=1.0)
        count = self.ecu.acknowledge_alerts()
        self.assertGreater(count, 0)

    def test_eyes_closed_long(self):
        """Test alert for prolonged eye closure."""
        metrics = BehavioralMetrics(
            fatigue_score=0.3,
            current_eye_closure_duration=3.0,
            driver_state=DriverState.FATIGUED,
        )
        state = self.ecu.process(metrics, timestamp=1.0)
        self.assertEqual(state.current_alert_level, AlertLevel.CRITICAL)

    def test_reset(self):
        """Test ECU reset."""
        metrics = BehavioralMetrics(fatigue_score=0.8, driver_state=DriverState.DROWSY)
        self.ecu.process(metrics, timestamp=1.0)
        self.ecu.reset()
        self.assertEqual(self.ecu.state.total_alerts_generated, 0)
        self.assertEqual(len(self.ecu.state.active_alerts), 0)


class TestECUSimulator(unittest.TestCase):
    """Tests for the ECU simulator."""

    def test_run_scenario(self):
        """Test running a basic scenario."""
        sim = ECUSimulator()
        data = [{"eyes_open": True, "yaw": 0, "pitch": 0, "looking_forward": True}] * 100
        results = sim.run_scenario(data, time_step=0.033)
        self.assertEqual(len(results), 100)
        self.assertIn("fatigue_score", results[0])
        self.assertIn("driver_state", results[0])

    def test_attentive_scenario_results(self):
        """Test attentive scenario produces expected results."""
        sim = ECUSimulator()
        data = generate_attentive_scenario(duration_seconds=10.0, fps=10.0)
        results = sim.run_scenario(data, time_step=0.1)
        df = pd.DataFrame(results)
        self.assertLess(df["fatigue_score"].mean(), 0.4)

    def test_fatigued_scenario_results(self):
        """Test fatigued scenario produces elevated fatigue scores."""
        sim = ECUSimulator()
        data = generate_fatigued_scenario(duration_seconds=30.0, fps=10.0)
        results = sim.run_scenario(data, time_step=0.1)
        df = pd.DataFrame(results)
        self.assertGreater(df["fatigue_score"].max(), 0.2)


class TestScenarios(unittest.TestCase):
    """Tests for scenario generation."""

    def test_attentive_scenario_length(self):
        data = generate_attentive_scenario(duration_seconds=10.0, fps=10.0)
        self.assertEqual(len(data), 100)

    def test_fatigued_scenario_length(self):
        data = generate_fatigued_scenario(duration_seconds=10.0, fps=10.0)
        self.assertEqual(len(data), 100)

    def test_distracted_scenario_length(self):
        data = generate_distracted_scenario(duration_seconds=10.0, fps=10.0)
        self.assertEqual(len(data), 100)

    def test_mixed_scenario_length(self):
        data = generate_mixed_scenario(duration_seconds=10.0, fps=10.0)
        self.assertEqual(len(data), 100)

    def test_scenario_data_keys(self):
        """Test that all scenarios produce correct data keys."""
        for gen in [
            generate_attentive_scenario,
            generate_fatigued_scenario,
            generate_distracted_scenario,
            generate_mixed_scenario,
        ]:
            data = gen(duration_seconds=1.0, fps=10.0)
            self.assertIn("eyes_open", data[0])
            self.assertIn("yaw", data[0])
            self.assertIn("pitch", data[0])
            self.assertIn("looking_forward", data[0])

    def test_attentive_mostly_eyes_open(self):
        """Attentive scenario should have eyes open most of the time."""
        data = generate_attentive_scenario(duration_seconds=30.0, fps=10.0)
        open_ratio = sum(1 for d in data if d["eyes_open"]) / len(data)
        self.assertGreater(open_ratio, 0.8)

    def test_distracted_high_yaw(self):
        """Distracted scenario should show high yaw values in later phases."""
        data = generate_distracted_scenario(duration_seconds=60.0, fps=10.0)
        late_data = data[len(data) // 2 :]
        max_yaw = max(abs(d["yaw"]) for d in late_data)
        self.assertGreater(max_yaw, 20)


class TestHeadPose(unittest.TestCase):
    """Tests for head pose estimation."""

    def test_head_pose_dataclass(self):
        pose = HeadPose(yaw=0, pitch=0, roll=0, is_looking_forward=True)
        self.assertFalse(pose.is_distracted)
        self.assertAlmostEqual(pose.distraction_level, 0.0)

    def test_head_pose_distracted(self):
        pose = HeadPose(yaw=45, pitch=0, roll=0, is_looking_forward=False)
        self.assertTrue(pose.is_distracted)
        self.assertGreater(pose.distraction_level, 0.3)

    def test_simple_estimation_forward(self):
        pose = estimate_head_pose_simple(
            face_bbox=(200, 150, 200, 250),
            left_eye=(270, 220),
            right_eye=(330, 220),
        )
        self.assertLess(abs(pose.yaw), 30)

    def test_simple_estimation_no_eyes(self):
        pose = estimate_head_pose_simple(
            face_bbox=(200, 150, 200, 250),
            left_eye=None,
            right_eye=None,
        )
        self.assertEqual(pose.yaw, 0.0)
        self.assertEqual(pose.pitch, 0.0)

    def test_estimator_invalid_landmarks(self):
        estimator = HeadPoseEstimator()
        result = estimator.estimate(np.zeros((3, 2)))
        self.assertIsNone(result)


class TestVisualization(unittest.TestCase):
    """Tests for visualization functions."""

    def test_create_dataframe(self):
        results = [
            {"time": 0, "fatigue_score": 0.1, "driver_state": "attentive"},
            {"time": 1, "fatigue_score": 0.2, "driver_state": "attentive"},
        ]
        df = create_scenario_dataframe(results)
        self.assertEqual(len(df), 2)
        self.assertIn("fatigue_score", df.columns)

    def test_summary_table(self):
        sim = ECUSimulator()
        data = generate_attentive_scenario(duration_seconds=5.0, fps=10.0)
        results = sim.run_scenario(data, time_step=0.1)
        df = create_scenario_dataframe(results)
        summary = generate_summary_table(df)
        self.assertIn("Metric", summary.columns)
        self.assertIn("Value", summary.columns)
        self.assertGreater(len(summary), 5)


if __name__ == "__main__":
    unittest.main()
