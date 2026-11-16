"""Numerical 1:1 parity tests between PyO3 Rust core and pure-Python fallback."""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control.rust_backend import (
    RUST_CORE_AVAILABLE,
    python_fallback_module,
    rust_core_module,
)


class TestRustPythonParity(unittest.TestCase):
    """Verify Rust and pure-Python backends match to 1e-6 across randomized closed-loop trajectories."""

    def test_scalar_pid_60_step_randomized_parity(self) -> None:
        self.assertTrue(RUST_CORE_AVAILABLE and rust_core_module is not None)
        rng = random.Random(2026)

        r_pid = rust_core_module.RustScalarPID(0.45, 0.01, 0.18, 5.0)
        p_pid = python_fallback_module.RustScalarPID(0.45, 0.01, 0.18, 5.0)

        for step in range(80):
            # Include large jumps (>1.2) to exercise derivative kick reset
            err = rng.uniform(-3.0, 3.0) if step % 17 == 0 else rng.uniform(-0.8, 0.8)
            dt = rng.choice([0.04, 0.05, 0.10])
            r_out = r_pid.step(err, dt)
            p_out = p_pid.step(err, dt)
            self.assertAlmostEqual(r_out, p_out, delta=1e-9, msg=f"step={step}")

        r_pid.reset()
        p_pid.reset()
        self.assertAlmostEqual(r_pid.step(0.25, 0.05), p_pid.step(0.25, 0.05), delta=1e-9)

        # NaN / Inf guard
        self.assertEqual(r_pid.step(float("nan"), 0.05), 0.0)
        self.assertEqual(p_pid.step(float("nan"), 0.05), 0.0)
        self.assertEqual(r_pid.step(0.5, float("inf")), 0.0)
        self.assertEqual(p_pid.step(0.5, float("inf")), 0.0)

    def test_lead_speed_estimator_irregular_steps_and_cutins(self) -> None:
        self.assertTrue(RUST_CORE_AVAILABLE and rust_core_module is not None)
        rng = random.Random(77)

        r_est = rust_core_module.RustLeadSpeedEstimator(35.0)
        p_est = python_fallback_module.RustLeadSpeedEstimator(35.0)

        dist = 28.0
        for step in range(75):
            if step in (25, 52):
                dist += 9.5  # >6.0m cut-in/target-switch window reset
            else:
                dist = max(4.0, dist + rng.uniform(-0.6, 0.6))
            rr = rng.uniform(-5.0, 2.5)
            ego_v = rng.uniform(0.0, 60.0)
            elapsed = rng.choice([1, 1, 2, 3])
            r_v = r_est.update(dist, rr, ego_v, elapsed, 0.05)
            p_v = p_est.update(dist, rr, ego_v, elapsed, 0.05)
            self.assertAlmostEqual(r_v, p_v, delta=1e-6, msg=f"step={step}")

        # NaN / Inf input resilience
        r_nan = r_est.update(float("nan"), float("inf"), float("-inf"), 1, 0.05)
        p_nan = p_est.update(float("nan"), float("inf"), float("-inf"), 1, 0.05)
        self.assertTrue(math.isfinite(r_nan) and math.isfinite(p_nan))
        self.assertAlmostEqual(r_nan, p_nan, delta=1e-6)

    def test_safety_shield_pid_65_randomized_trajectories_parity(self) -> None:
        self.assertTrue(RUST_CORE_AVAILABLE and rust_core_module is not None)
        rng = random.Random(4242)

        r_ctrl = rust_core_module.RustSafetyShieldPID(use_mock=False)
        p_ctrl = python_fallback_module.RustSafetyShieldPID(use_mock=False)

        maneuvers = ["keep_lane", "change_left", "change_right", "follow_lead"]
        lights = ["green", "yellow", "red"]
        hazards = ["none", "vehicle", "pedestrian", "red_light"]

        for step in range(65):
            maneuver = rng.choice(maneuvers)
            decision = {
                "maneuver": maneuver,
                "maneuver_confidence": rng.uniform(0.55, 0.99),
                "speed_level": rng.choice([0, 1, 2, 3]),
                "emergency_brake": rng.choice([0.05, 0.25, 0.85]),
                "predicted_trajectory": [
                    [rng.uniform(2.0, 14.0), rng.uniform(-1.5, 1.5), rng.uniform(0.0, 65.0)]
                    for _ in range(3)
                ],
            }
            telemetry = {
                "speed_kmh": rng.uniform(0.0, 65.0),
                "lead_distance_m": rng.uniform(3.0, 65.0),
                "lead_speed_kmh": rng.uniform(0.0, 60.0),
                "pedestrian_distance_m": rng.choice([6.0, 14.0, 100.0]),
                "lane_offset_m": rng.uniform(-0.8, 0.8),
                "heading_error": rng.uniform(-0.25, 0.25),
                "road_curvature": rng.uniform(0.0, 0.55),
                "traffic_light": rng.choice(lights),
                "hazard_source": rng.choice(hazards),
                "wet_road": rng.choice([False, True]),
                "a_max_mps2": rng.choice([4.5, 6.0]),
                "uncertainty_margin_m": rng.choice([0.0, 1.8]),
                "left_lane_available": rng.choice([True, False]),
                "right_lane_available": rng.choice([True, False]),
                "blind_spot_left": rng.choice([False, True]),
                "blind_spot_right": rng.choice([False, True]),
            }
            shield_on = (step % 11) != 0
            (r_steer, r_tb), r_info = r_ctrl.compute_control(decision, telemetry, 0.05, shield_on)
            (p_steer, p_tb), p_info = p_ctrl.compute_control(decision, telemetry, 0.05, shield_on)

            self.assertAlmostEqual(r_steer, p_steer, delta=1e-6, msg=f"steer mismatch at step={step}")
            self.assertAlmostEqual(r_tb, p_tb, delta=1e-6, msg=f"tb mismatch at step={step}")
            self.assertEqual(r_info["shield_triggered"], p_info["shield_triggered"])
            self.assertEqual(r_info["shield_reason"], p_info["shield_reason"])
            self.assertEqual(r_info["maneuver"], p_info["maneuver"])
            self.assertEqual(r_info["speed_level"], p_info["speed_level"])
            self.assertAlmostEqual(r_info["mpc_feedforward"], p_info["mpc_feedforward"], delta=1e-6)

        # NaN / Inf telemetry inputs must return finite safe controls
        bad_dec = {
            "maneuver": "keep_lane",
            "maneuver_confidence": float("nan"),
            "speed_level": 2,
            "emergency_brake": float("nan"),
            "predicted_trajectory": [[float("nan"), float("inf"), 10.0]] * 3,
        }
        bad_tel = {
            "speed_kmh": float("nan"),
            "lead_distance_m": float("inf"),
            "lead_speed_kmh": float("-inf"),
            "lane_offset_m": float("nan"),
            "heading_error": float("nan"),
            "road_curvature": float("nan"),
        }
        (rs, rt), _ = r_ctrl.compute_control(bad_dec, bad_tel, 0.05, True)
        (ps, pt), _ = p_ctrl.compute_control(bad_dec, bad_tel, 0.05, True)
        self.assertTrue(all(math.isfinite(x) for x in (rs, rt, ps, pt)))
        self.assertAlmostEqual(rs, ps, delta=1e-6)
        self.assertAlmostEqual(rt, pt, delta=1e-6)

    def test_expert_labeler_and_geometry_functions_parity(self) -> None:
        self.assertTrue(RUST_CORE_AVAILABLE and rust_core_module is not None)
        rng = random.Random(999)

        r_lbl = rust_core_module.RustExpertLabeler()
        p_lbl = python_fallback_module.RustExpertLabeler()
        for idx in range(60):
            tel = {
                "speed_kmh": rng.uniform(0.0, 65.0),
                "left_lane_available": rng.choice([True, False]),
                "right_lane_available": rng.choice([True, False]),
                "nav_command": rng.choice(["go_straight", "turn_left", "turn_right", "merge"]),
                "lead_distance_m": rng.uniform(4.0, 65.0),
                "road_curvature": rng.uniform(0.0, 0.6),
                "traffic_light": rng.choice(["green", "yellow", "red"]),
                "hazard_source": rng.choice(["none", "vehicle", "pedestrian", "red_light"]),
            }
            act = (rng.uniform(-0.6, 0.6), rng.uniform(-1.0, 0.8))
            info = {"scenario_event": rng.choice(["", "sharp_curve"]), "is_real_sim": rng.choice([True, False])}
            self.assertEqual(
                r_lbl.label_step(tel, act, info),
                p_lbl.label_step(tel, act, info),
                msg=f"label mismatch at idx={idx}",
            )

        # Standalone pure-math & polyline helpers
        self.assertAlmostEqual(
            rust_core_module.compute_kinematic_aeb_distance(45.0, 15.0, 4.5, 1.8),
            python_fallback_module.compute_kinematic_aeb_distance(45.0, 15.0, 4.5, 1.8),
            delta=1e-6,
        )
        self.assertAlmostEqual(
            rust_core_module.compute_mpc_feedforward(0.24, 32.0, 0.1),
            python_fallback_module.compute_mpc_feedforward(0.24, 32.0, 0.1),
            delta=1e-6,
        )
        self.assertAlmostEqual(
            rust_core_module.ols_5step_slope_mps([25.0, 24.4, 23.7, 23.0, 22.2], 0.05),
            python_fallback_module.ols_5step_slope_mps([25.0, 24.4, 23.7, 23.0, 22.2], 0.05),
            delta=1e-6,
        )
        ts_list = [0.0, 0.05, 0.15, 0.20]
        dist_list = [20.0, 19.6, 18.8, 18.4]
        self.assertAlmostEqual(
            rust_core_module.timestamp_ols_slope_mps(ts_list, dist_list),
            python_fallback_module.timestamp_ols_slope_mps(ts_list, dist_list),
            delta=1e-6,
        )
        trace = [0.1, -0.6, -0.7, 0.0, -0.8, -0.8, 0.1, 0.2, 0.3, -0.9, -0.9, -0.9]
        self.assertEqual(
            rust_core_module.count_sustained_aeb_events(trace, -0.50, 3, 2),
            python_fallback_module.count_sustained_aeb_events(trace, -0.50, 3, 2),
        )

        route = [(float(i) * 2.0, math.sin(i * 0.2), 0.0) for i in range(30)]
        r_pts = rust_core_module.compute_polyline_tangents(route)
        p_pts = python_fallback_module.compute_polyline_tangents(route)
        for rp, pp in zip(r_pts, p_pts):
            for rv, pv in zip(rp, pp):
                self.assertAlmostEqual(rv, pv, delta=1e-9)

        adj = [(x, y + 3.5, z) for x, y, z in route]
        r_blend = rust_core_module.quintic_s_curve_blend(route, adj, 14)
        p_blend = python_fallback_module.quintic_s_curve_blend(route, adj, 14)
        for rb, pb in zip(r_blend, p_blend):
            for rv, pv in zip(rb, pb):
                self.assertAlmostEqual(rv, pv, delta=1e-9)

        r_pose = rust_core_module.sample_continuous_route_pose(r_pts, 6.2, 0.4, 0, 8.0)
        p_pose = python_fallback_module.sample_continuous_route_pose(p_pts, 6.2, 0.4, 0, 8.0)
        self.assertEqual(r_pose[0], p_pose[0])
        for rv, pv in zip(r_pose[1:], p_pose[1:]):
            self.assertAlmostEqual(rv, pv, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
