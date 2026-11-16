"""Unit tests for MultiSensorFusionTracker, ActuatorDynamicsFilter, SOTIF faults, metrics & CARLA launcher."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.engine import DrivingRuntimeEngine
from src.sim.actuator_and_faults import (
    VALID_FAULT_MODES,
    ActuatorDynamicsFilter,
    SOTIFSensorFaultInjector,
)
from src.sim.carla_launcher import CarlaServerManager, find_carla_executable
from src.sim.metrics import EpisodeSafetyComfortTracker, aggregate_leaderboard
from src.sim.sensor_fusion import MultiSensorFusionTracker, compute_camera_health


class TestSimFusionAndMetrics(unittest.TestCase):
    """Verify multi-sensor fusion, actuator/SOTIF modeling, safety/comfort KPIs, and CARLA discovery."""

    def test_multi_sensor_fusion_cut_in_and_blindspot(self) -> None:
        tracker = MultiSensorFusionTracker()
        bright_img = np.full((196, 392, 3), 120, dtype=np.uint8)
        bright_img[:60, :] = 180

        aux_vision = {
            "pred_lead_distance_m": 58.0,  # Monocular head hasn't seen cut-in yet
            "pred_ped_distance_m": 100.0,
            "pred_range_rate_mps": 0.0,
            "pred_lane_offset_m": 0.02,
            "pred_traffic_light": "green",
            "pred_hazard_source": "none",
            "pred_blindspot_left": 0.0,
            "pred_blindspot_right": 0.0,
        }
        radar_targets = [
            # Aggressive cut-in vehicle from right lane cutting left toward ego path
            {
                "x_m": 16.5,
                "y_m": 3.1,
                "vx_rel_mps": -4.5,
                "vy_rel_mps": -1.8,
                "speed_kmh": 19.8,
                "category": "vehicle",
                "source": "carla_radar",
            },
            # High-speed rear-left blind-spot overtaker approaching rapidly
            {
                "x_m": -13.5,
                "y_m": -3.3,
                "vx_rel_mps": 6.5,
                "vy_rel_mps": 0.0,
                "speed_kmh": 60.0,
                "category": "vehicle",
                "source": "carla_radar",
            },
        ]

        fused = tracker.fuse(
            aux_vision=aux_vision,
            base_telemetry={"speed_kmh": 36.0, "wet_road": False},
            radar_targets=radar_targets,
            rgb=bright_img,
            dt=0.05,
        )
        self.assertTrue(fused["fusion_active"])
        self.assertTrue(fused["cut_in_detected"])
        self.assertTrue(fused["blind_spot_left"])
        self.assertFalse(fused["blind_spot_right"])
        self.assertLess(fused["lead_distance_m"], 20.0)
        self.assertEqual(fused["hazard_source"], "vehicle")

    def test_multi_sensor_fusion_camera_dropout_resilience(self) -> None:
        tracker = MultiSensorFusionTracker()
        blackout_rgb = np.zeros((196, 392, 3), dtype=np.uint8)
        health = compute_camera_health(blackout_rgb)
        self.assertLess(health["health_score"], 0.30)
        self.assertEqual(health["degradation_reason"], "camera_dropout")

        fused = tracker.fuse(
            aux_vision={
                "pred_lead_distance_m": 55.0,  # Blinded vision failsafe noise
                "pred_ped_distance_m": 100.0,
                "pred_range_rate_mps": 0.0,
                "pred_lane_offset_m": 0.0,
                "pred_traffic_light": "green",
                "pred_hazard_source": "none",
                "pred_blindspot_left": 0.0,
                "pred_blindspot_right": 0.0,
            },
            base_telemetry={"speed_kmh": 40.0, "wet_road": False},
            radar_targets=[
                {
                    "x_m": 14.0,
                    "y_m": 0.1,
                    "vx_rel_mps": -5.5,
                    "vy_rel_mps": 0.0,
                    "speed_kmh": 20.2,
                    "category": "vehicle",
                    "source": "metadrive_3d",
                }
            ],
            rgb=blackout_rgb,
        )
        self.assertLess(fused["sensor_health_score"], 0.30)
        self.assertAlmostEqual(fused["lead_distance_m"], 14.0, delta=0.5)
        self.assertGreater(fused["uncertainty_margin_m"], 1.5)

    def test_actuator_dynamics_and_sotif_faults(self) -> None:
        act = ActuatorDynamicsFilter(
            steer_tau_s=0.10,
            max_steer_rate_rps=2.0,
            steer_deadband=0.01,
            throttle_tau_s=0.10,
            brake_tau_s=0.05,
        )
        (s1, tb1), diag = act.apply(1.0, 1.0, speed_kmh=40.0, dt=0.05)
        # First-order servo lag + speed-dependent rate limit prevents instant jump to 1.0
        self.assertGreater(s1, 0.0)
        self.assertLess(s1, 0.35)
        self.assertGreater(tb1, 0.0)
        self.assertLess(tb1, 0.5)
        self.assertGreater(diag["steer_lag_error"], 0.5)

        injector = SOTIFSensorFaultInjector(seed=7)
        base_rgb = np.full((196, 392, 3), 128, dtype=np.uint8)
        radar_in = [{"x_m": 20.0, "y_m": 0.0, "vx_rel_mps": -2.0, "vy_rel_mps": 0.0}]
        for mode in VALID_FAULT_MODES:
            injector.set_mode(mode)
            fault_rgb = injector.inject_camera(base_rgb, step_idx=24)
            self.assertEqual(fault_rgb.shape, base_rgb.shape)
            self.assertEqual(fault_rgb.dtype, np.uint8)
            fault_rad = injector.inject_radar(radar_in, step_idx=24)
            self.assertGreaterEqual(len(fault_rad), 1)

    def test_safety_comfort_tracker_math(self) -> None:
        tracker = EpisodeSafetyComfortTracker(dt=0.05)
        # Analytical deceleration profile from 36 km/h (10 m/s) closing on 0 km/h obstacle
        speeds = [36.0, 34.0, 30.0, 24.0, 16.0, 8.0, 2.0, 0.0]
        dists = [20.0, 19.5, 19.0, 18.6, 18.3, 18.1, 18.0, 18.0]
        for spd, d in zip(speeds, dists):
            tracker.step(
                telemetry={
                    "speed_kmh": spd,
                    "lead_distance_m": d,
                    "lead_speed_kmh": 0.0,
                    "pedestrian_distance_m": 100.0,
                    "traffic_light": "green",
                    "hazard_source": "vehicle",
                },
                control_action=(0.0, -0.5),
                extra_info={"crash": False, "out_of_road": False},
            )
        summary = tracker.summarize(route_completion=1.0, arrived=True, crashed=False)
        self.assertGreater(summary["min_ttc_s"], 1.5)
        self.assertLess(summary["min_ttc_s"], 4.0)
        self.assertGreater(summary["rms_long_jerk_mps3"], 0.0)
        self.assertEqual(summary["driving_score_pct"], 100.0)

        # Infraction penalty reduces Driving Score
        tracker.vehicle_collisions = 1
        penalized = tracker.summarize(route_completion=1.0, arrived=True, crashed=True)
        self.assertAlmostEqual(penalized["driving_score_pct"], 60.0, delta=0.1)

        agg = aggregate_leaderboard([summary, penalized])
        self.assertEqual(agg["episodes"], 2)
        self.assertAlmostEqual(agg["mean_driving_score_pct"], 80.0, delta=0.1)

    def test_carla_launcher_discovery(self) -> None:
        exe = find_carla_executable()
        stat = CarlaServerManager().status()
        self.assertIn("carla_exe_found", stat)
        self.assertIn("port_open", stat)
        self.assertEqual(stat["carla_exe_found"], exe is not None)

    def test_runtime_engine_vision_radar_fusion_mode(self) -> None:
        engine = DrivingRuntimeEngine(use_mock=True)
        engine.warmup(passes=1)
        blackout = np.zeros((196, 392, 3), dtype=np.uint8)
        resp = engine.step(
            session_id="fusion_test",
            image=blackout,
            telemetry={
                "speed_kmh": 45.0,
                "radar_targets": [
                    {
                        "x_m": 12.0,
                        "y_m": 0.0,
                        "vx_rel_mps": -8.0,
                        "vy_rel_mps": 0.0,
                        "speed_kmh": 16.2,
                        "category": "vehicle",
                        "source": "carla_radar",
                    }
                ],
            },
            perception_mode="vision_radar_fusion",
            shield_enabled=True,
        )
        self.assertEqual(resp.perception_mode, "vision_radar_fusion")
        self.assertTrue(resp.aux_perception.fusion_active)
        self.assertLess(resp.aux_perception.sensor_health_score, 0.30)
        # Safety Shield should trigger braking for 12m closing hazard at 45 km/h
        self.assertTrue(resp.control.shield_triggered)
        self.assertLess(resp.control.throttle_brake, 0.0)

    def test_intersection_crossing_conflict_vs_non_conflicting_diverging_turn(self) -> None:
        tracker = MultiSensorFusionTracker()
        bright_img = np.full((196, 392, 3), 120, dtype=np.uint8)
        bright_img[:60, :] = 180
        aux_clear = {
            "pred_lead_distance_m": 60.0,
            "pred_ped_distance_m": 100.0,
            "pred_range_rate_mps": 0.0,
            "pred_lane_offset_m": 0.0,
            "pred_traffic_light": "green",
            "pred_hazard_source": "none",
            "pred_blindspot_left": 0.0,
            "pred_blindspot_right": 0.0,
        }

        # Diverging concentric intersection vehicle tagged non_conflicting_cross must NOT lock Ego
        diverging_only = tracker.fuse(
            aux_vision=aux_clear,
            base_telemetry={"speed_kmh": 28.0, "wet_road": False},
            radar_targets=[
                {
                    "x_m": 10.5,
                    "y_m": 0.8,
                    "vx_rel_mps": -6.5,
                    "vy_rel_mps": 2.4,
                    "speed_kmh": 25.0,
                    "category": "vehicle",
                    "non_conflicting_cross": True,
                    "source": "metadrive_3d",
                }
            ],
            rgb=bright_img,
            dt=0.05,
        )
        self.assertGreaterEqual(diverging_only["lead_distance_m"], 50.0)
        self.assertEqual(diverging_only["hazard_source"], "none")

        # True spatio-temporal crossing conflict must trigger immediate stopping reference
        tracker.reset()
        conflicting = tracker.fuse(
            aux_vision=aux_clear,
            base_telemetry={"speed_kmh": 28.0, "wet_road": False},
            radar_targets=[
                {
                    "x_m": 11.5,
                    "y_m": 1.1,
                    "vx_rel_mps": -7.5,
                    "vy_rel_mps": 1.5,
                    "speed_kmh": 22.0,
                    "category": "vehicle",
                    "crossing_conflict": True,
                    "source": "metadrive_3d",
                }
            ],
            rgb=bright_img,
            dt=0.05,
        )
        self.assertLessEqual(conflicting["lead_distance_m"], 12.5)
        self.assertLess(conflicting["lead_speed_kmh"], 15.0)
        self.assertEqual(conflicting["hazard_source"], "vehicle")
        # Subsequent frames converge smoothly toward 0.0 km/h stop target
        for _ in range(3):
            conflicting = tracker.fuse(
                aux_vision=aux_clear,
                base_telemetry={"speed_kmh": 20.0, "wet_road": False},
                radar_targets=[
                    {
                        "x_m": 10.5,
                        "y_m": 0.9,
                        "vx_rel_mps": -5.5,
                        "vy_rel_mps": 1.0,
                        "speed_kmh": 22.0,
                        "category": "vehicle",
                        "crossing_conflict": True,
                        "source": "metadrive_3d",
                    }
                ],
                rgb=bright_img,
                dt=0.05,
            )
        self.assertLess(conflicting["lead_speed_kmh"], 2.0)


if __name__ == "__main__":
    unittest.main()
