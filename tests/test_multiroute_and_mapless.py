"""Unit & integration tests for Multi-Route Dynamic A* Re-Routing, Multi-Town, and Mapless Corridor."""

from __future__ import annotations

import math
import unittest
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import load_validated_config
from src.env import create_driving_env
from src.sim.mapless_corridor import MaplessVisionCorridorEstimator
from src.sim.scenarios import RealisticSimEnvWrapper


def _render_synthetic_corridor_rgb(
    est: MaplessVisionCorridorEstimator,
    c0: float = 0.0,
    c1: float = 0.0,
    c2: float = 0.0,
    draw_lane_lines: bool = True,
    h: int = 196,
    w: int = 392,
) -> np.ndarray:
    """Render a synthetic pinhole-consistent road image for `y_right(x) = c0 + c1*x + 0.5*c2*x^2`."""
    est._ensure_ipm_tables(h, w)
    assert est._roi_rows is not None
    assert est._row_x_m is not None
    assert est._row_m_per_px is not None

    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky + vegetation/sidewalk background
    img[:48, :] = (135, 170, 210)
    img[48:, :] = (55, 105, 50)

    half_cx = 0.5 * float(w)
    half_w_m = 0.5 * est.lane_width_m
    pts_left = []
    pts_right = []
    pts_road_l = []
    pts_road_r = []

    for row, xm, m_per_px in zip(est._roi_rows, est._row_x_m, est._row_m_per_px):
        y_center = float(c0 + c1 * xm + 0.5 * c2 * (xm ** 2))
        col_l = int(round(half_cx + (y_center - half_w_m) / m_per_px))
        col_r = int(round(half_cx + (y_center + half_w_m) / m_per_px))
        col_rl = int(round(half_cx + (y_center - half_w_m - 0.5) / m_per_px))
        col_rr = int(round(half_cx + (y_center + half_w_m + 0.5) / m_per_px))
        pts_left.append((col_l, int(row)))
        pts_right.append((col_r, int(row)))
        pts_road_l.append((col_rl, int(row)))
        pts_road_r.append((col_rr, int(row)))

    poly = np.array(pts_road_l + pts_road_r[::-1], dtype=np.int32)
    cv2.fillPoly(img, [poly], (72, 74, 78))

    if draw_lane_lines:
        for i in range(len(pts_left) - 1):
            cv2.line(img, pts_left[i], pts_left[i + 1], (242, 242, 242), 3, cv2.LINE_AA)
            cv2.line(img, pts_right[i], pts_right[i + 1], (242, 242, 242), 3, cv2.LINE_AA)
    return img


class TestMultiRouteAndMapless(unittest.TestCase):
    def test_mapless_corridor_clothoid_sign_and_offset_accuracy(self) -> None:
        est = MaplessVisionCorridorEstimator(lane_width_m=3.5)

        # 1. Centered straight lane (c0 = 0, c1 = 0, c2 = 0)
        rgb_center = _render_synthetic_corridor_rgb(est, c0=0.0, c1=0.0, c2=0.0)
        diag0 = est.update(rgb_center, speed_kmh=36.0, dt=0.05)
        self.assertEqual(diag0["localization_source"], "mapless_lane_clothoid")
        self.assertLess(abs(float(diag0["lane_offset_m"])), 0.22)
        self.assertGreater(float(diag0["corridor_confidence"]), 0.45)

        # 2. Ego shifted LEFT of lane center -> lane center is at y_right = +0.55m -> lane_offset_m > 0
        est.reset()
        rgb_left = _render_synthetic_corridor_rgb(est, c0=0.55, c1=0.08, c2=0.0)
        for _ in range(3):
            diag_l = est.update(rgb_left, speed_kmh=36.0, dt=0.05)
        self.assertGreater(float(diag_l["lane_offset_m"]), 0.20)
        self.assertGreater(float(diag_l["heading_error"]), 0.02)

        # 3. Ego shifted RIGHT of lane center -> lane center is at y_right = -0.55m -> lane_offset_m < 0
        est.reset()
        rgb_right = _render_synthetic_corridor_rgb(est, c0=-0.55, c1=-0.08, c2=0.0)
        for _ in range(3):
            diag_r = est.update(rgb_right, speed_kmh=36.0, dt=0.05)
        self.assertLess(float(diag_r["lane_offset_m"]), -0.20)
        self.assertLess(float(diag_r["heading_error"]), -0.02)

        # 4. Left-curving lane (y_right curves left: c2 < 0 -> road_curvature > 0)
        est.reset()
        rgb_curve_l = _render_synthetic_corridor_rgb(est, c0=0.0, c1=0.0, c2=-0.012)
        for _ in range(3):
            diag_c = est.update(rgb_curve_l, speed_kmh=36.0, dt=0.05)
        self.assertGreater(float(diag_c["road_curvature"]), 0.02)

        # 5. Unmarked asphalt corridor (no white/yellow paint) -> freespace fallback
        est.reset()
        rgb_unmarked = _render_synthetic_corridor_rgb(
            est, c0=0.35, c1=0.0, c2=0.0, draw_lane_lines=False
        )
        diag_fs = est.update(rgb_unmarked, speed_kmh=30.0, dt=0.05)
        self.assertEqual(diag_fs["localization_source"], "mapless_freespace_imu")
        self.assertTrue(math.isfinite(float(diag_fs["lane_offset_m"])))

        # 6. Total camera blackout -> IMU kinematic bicycle dead-reckoning
        blackout = np.zeros((196, 392, 3), dtype=np.uint8)
        diag_imu = est.update(
            blackout,
            speed_kmh=36.0,
            imu_telemetry={"yaw_rate_rps": 0.12},
            dt=0.05,
        )
        self.assertEqual(diag_imu["localization_source"], "mapless_blackout_hold")
        self.assertTrue(math.isfinite(float(diag_imu["lane_offset_m"])))

    def test_hybrid_fallback_and_pure_mapless_zero_hd_map_leakage(self) -> None:
        cfg = load_validated_config().to_dict()
        base_env = create_driving_env(config=cfg, use_mock=True, simulator="metadrive")
        try:
            env = RealisticSimEnvWrapper(
                base_env=base_env,
                scenario="nominal_urban_cruise",
                localization_mode="hybrid_fallback",
            )
            _rgb, _tel0, info0 = env.reset(seed=5000)
            self.assertEqual(
                info0["localization_diagnostics"]["localization_source"], "hd_map_primary"
            )

            # Inject HD-Map Dropout mid-episode -> must switch to mapless estimator automatically
            env.set_fault_mode("hd_map_dropout")
            _rgb1, _tel1, _rew, _done, info1 = env.step((0.0, 0.35))
            self.assertTrue(bool(info1["localization_diagnostics"]["hd_map_dropped"]))
            self.assertNotEqual(
                info1["localization_diagnostics"]["localization_source"], "hd_map_primary"
            )

            # Switch to pure 100% mapless_vision and poison base_env lateral HD-map ground truth
            env.set_fault_mode("none")
            env.set_localization_mode("mapless_vision")
            orig_step = base_env.step

            def _poisoned_hd_step(action: Tuple[float, float]) -> Any:
                r, t, rw, d, inf = orig_step(action)
                t = dict(t)
                t["lane_offset_m"] = 999.0
                t["heading_error"] = 3.14
                t["road_curvature"] = -99.0
                return r, t, rw, d, inf

            base_env.step = _poisoned_hd_step  # type: ignore[method-assign]
            _rgb2, tel2, _rew2, _done2, info2 = env.step((0.0, 0.35))
            self.assertNotEqual(
                info2["localization_diagnostics"]["localization_source"], "hd_map_primary"
            )
            self.assertLessEqual(abs(float(tel2["lane_offset_m"])), 1.75)
            self.assertLessEqual(abs(float(tel2["heading_error"])), 0.55)
            self.assertLessEqual(abs(float(tel2["road_curvature"])), 1.0)
        finally:
            base_env.close()

    def test_dynamic_destination_reroute_roadblock_and_endless_tour(self) -> None:
        cfg = load_validated_config().to_dict()
        for sim_name in ("metadrive", "carla"):
            base_env = create_driving_env(config=cfg, use_mock=True, simulator=sim_name)
            try:
                env = RealisticSimEnvWrapper(
                    base_env=base_env,
                    scenario="nominal_urban_cruise",
                    endless_tour=True,
                )
                _rgb, _tel, info = env.reset(seed=5001)
                self.assertIn("nav_state", info)
                nodes = env.list_spawn_points()
                self.assertGreaterEqual(len(nodes), 3)

                nav_res1 = env.set_destination(dest_spawn_idx=int(nodes[-1]["index"]))
                self.assertTrue(bool(nav_res1.get("rerouted")))
                self.assertGreaterEqual(int(nav_res1.get("reroute_count", 0)), 1)

                nav_res2 = env.set_destination(dest_xy=(140.0, -18.0))
                self.assertTrue(bool(nav_res2.get("rerouted")))
                self.assertGreaterEqual(int(nav_res2.get("reroute_count", 0)), 2)

                rb = env.spawn_roadblock(distance_ahead_m=22.0)
                self.assertIsInstance(rb, dict)

                # Simulate arrival in endless_tour=True -> must seamlessly chain to next leg without done=True
                orig_step = base_env.step

                def _arrive_step(action: Tuple[float, float]) -> Any:
                    r, t, rw, _d, inf = orig_step(action)
                    inf = dict(inf)
                    inf["arrive_dest"] = True
                    inf["crash"] = False
                    return r, t, rw, True, inf

                base_env.step = _arrive_step  # type: ignore[method-assign]
                _r, _t, _rw, done_tour, info_tour = env.step((0.0, 0.4))
                self.assertFalse(done_tour)
                self.assertEqual(int(info_tour["tour_legs_completed"]), 1)
                self.assertFalse(bool(info_tour["arrive_dest"]))
            finally:
                base_env.close()

    def test_cockpit_navigation_map_and_mapless_api_controls(self) -> None:
        app = create_app(use_mock=True)
        with TestClient(app) as client:
            r_html = client.get("/cockpit")
            self.assertEqual(r_html.status_code, 200)
            self.assertIn("navMapCanvas", r_html.text)
            self.assertIn("hd_map_dropout", r_html.text)
            self.assertIn("mapless_vision", r_html.text)

            r_stat = client.get("/v1/sim/status")
            self.assertEqual(r_stat.status_code, 200)
            stat: Dict[str, Any] = r_stat.json()
            self.assertIn("available_carla_towns", stat)
            self.assertIn("available_localization_modes", stat)

            r_loc = client.post(
                "/v1/sim/control",
                json={"action": "set_localization_mode", "localization_mode": "mapless_vision"},
            )
            self.assertEqual(r_loc.status_code, 200)
            self.assertEqual(r_loc.json()["localization_mode"], "mapless_vision")

            r_tour = client.post(
                "/v1/sim/control",
                json={"action": "set_endless_tour", "enabled": True},
            )
            self.assertEqual(r_tour.status_code, 200)
            self.assertTrue(r_tour.json()["endless_tour"])

            r_dest = client.post(
                "/v1/sim/control",
                json={"action": "set_destination", "dest_xy": [165.0, 12.0]},
            )
            self.assertEqual(r_dest.status_code, 200)
            self.assertIn("nav_update", r_dest.json())
            self.assertTrue(bool(r_dest.json()["nav_update"].get("rerouted")))

            r_step = client.post("/v1/sim/control", json={"action": "step"})
            self.assertEqual(r_step.status_code, 200)
            pkt = r_step.json()["frame"]
            self.assertIn("nav_state", pkt)
            self.assertIn("localization_diagnostics", pkt)
            self.assertEqual(pkt["localization_mode"], "mapless_vision")


if __name__ == "__main__":
    unittest.main()
