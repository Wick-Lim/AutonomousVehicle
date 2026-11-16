"""End-to-end tests for real 3D MetaDrive integration and FastAPI + WebSocket /cockpit streaming."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import create_app
from src.env.metadrive_wrapper import MetaDriveWrapper
from src.runtime.engine import DrivingRuntimeEngine
from src.sim.metrics import EpisodeSafetyComfortTracker
from src.sim.scenarios import RealisticSimEnvWrapper

METADRIVE_INSTALLED = importlib.util.find_spec("metadrive") is not None


class TestRealMetaDriveAndCockpit(unittest.TestCase):
    """Verify live real 3D MetaDrive seed/sensor pipeline and interactive /cockpit + WS /ws/sim."""

    @unittest.skipUnless(METADRIVE_INSTALLED, "MetaDrive 3D package not installed")
    def test_real_metadrive_3d_closed_loop_and_seed_mapping(self) -> None:
        base_env = MetaDriveWrapper(
            use_mock=False,
            start_seed=1000,
            num_scenarios=3,
        )
        env = RealisticSimEnvWrapper(
            base_env=base_env,
            scenario="aggressive_cut_in",
            actuator_realism=True,
        )
        engine = DrivingRuntimeEngine(use_mock=True)
        engine.warmup(passes=1)
        tracker = EpisodeSafetyComfortTracker(dt=0.05)

        try:
            # Seed 5000 exceeds [1000, 1003); fixed wrapper must map cleanly without AssertionError
            rgb, telemetry, info = env.reset(seed=5000)
            self.assertEqual(rgb.shape, (196, 392, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            self.assertGreater(float(rgb.mean()), 5.0)
            self.assertIn("imu_telemetry", info)
            self.assertIn("radar_targets", info)

            for step_i in range(18):
                tel_in = dict(telemetry)
                tel_in["radar_targets"] = info.get("radar_targets", [])
                tel_in["imu_telemetry"] = info.get("imu_telemetry")
                resp = engine.step(
                    session_id="real_md_test",
                    image=rgb,
                    telemetry=tel_in,
                    perception_mode="vision_radar_fusion",
                    shield_enabled=True,
                )
                act = (resp.control.steering, resp.control.throttle_brake)
                rgb, telemetry, _, done, info = env.step(act)
                sc = tracker.step(
                    telemetry=telemetry,
                    control_action=act,
                    extra_info=info,
                )
                self.assertIn("ttc_s", sc)
                if done:
                    break
            summary = tracker.summarize(
                route_completion=float(info.get("route_completion", 0.0)),
                arrived=bool(info.get("arrive_dest", False)),
                crashed=bool(info.get("crash", False)),
            )
            self.assertGreaterEqual(summary["steps"], 1)
            self.assertIn("driving_score_pct", summary)
        finally:
            env.close()

    def test_cockpit_http_and_websocket_streaming(self) -> None:
        app = create_app(use_mock=True)
        with TestClient(app) as client:
            r_html = client.get("/cockpit")
            self.assertEqual(r_html.status_code, 200)
            self.assertIn("text/html", r_html.headers.get("content-type", ""))
            self.assertIn("/ws/sim", r_html.text)
            self.assertIn("bevCanvas", r_html.text)

            r_stat = client.get("/v1/sim/status")
            self.assertEqual(r_stat.status_code, 200)
            stat = r_stat.json()
            self.assertIn("available_scenarios", stat)
            self.assertIn("carla", stat)

            r_ctrl = client.post(
                "/v1/sim/control",
                json={"action": "set_scenario", "scenario": "occluded_jaywalker_dartout"},
            )
            self.assertEqual(r_ctrl.status_code, 200)
            self.assertEqual(r_ctrl.json()["scenario"], "occluded_jaywalker_dartout")

            r_step = client.post("/v1/sim/control", json={"action": "step"})
            self.assertEqual(r_step.status_code, 200)
            frame_pkt = r_step.json()["frame"]
            self.assertTrue(len(frame_pkt["frame_jpeg_b64"]) > 100)
            self.assertIn("radar_targets", frame_pkt)
            self.assertIn("episode_summary", frame_pkt)

            with client.websocket_connect("/ws/sim") as ws:
                init_msg = ws.receive_json()
                self.assertEqual(init_msg["type"], "status")

                ws.send_json({"action": "inject_hazard", "hazard": "cut_in"})
                hz_status = ws.receive_json()
                self.assertEqual(hz_status["type"], "status")

                ws.send_json({"action": "inject_fault", "fault_mode": "sun_glare"})
                flt_status = ws.receive_json()
                self.assertEqual(flt_status["type"], "status")
                self.assertEqual(flt_status["data"]["fault_mode"], "sun_glare")

                ws.send_json({"action": "step"})
                step_status = ws.receive_json()
                self.assertEqual(step_status["type"], "status")
                step_frame = ws.receive_json()
                self.assertEqual(step_frame["type"], "frame")
                self.assertGreater(len(step_frame["data"]["radar_targets"]), 0)
                self.assertEqual(step_frame["data"]["fault_mode"], "sun_glare")

    def test_curvilinear_spatiotemporal_conflict_geometry(self) -> None:
        class _DummyVeh:
            WIDTH = 1.85

        class _ConcentricArcLane:
            def __init__(self, radius: float, center_y: float) -> None:
                self.radius = float(radius)
                self.center_y = float(center_y)
                self.length = 0.5 * math.pi * self.radius

            def position(self, s: float, lat: float = 0.0) -> np.ndarray:
                r_eff = self.radius - float(lat)
                ang = float(s) / self.radius
                return np.array(
                    [r_eff * math.sin(ang), self.center_y + r_eff * math.cos(ang)],
                    dtype=np.float64,
                )

            def heading_theta_at(self, s: float) -> float:
                return -(float(s) / self.radius)

            def local_coordinates(self, pos: np.ndarray) -> Tuple[float, float]:
                dx = float(pos[0])
                dy = float(pos[1]) - self.center_y
                ang = math.atan2(dx, dy)
                rad = math.hypot(dx, dy)
                return (ang * self.radius, self.radius - rad)

        class _DummyOther:
            WIDTH = 1.95

            def __init__(self, lane: Any = None) -> None:
                self.lane = lane

        from typing import Any, Tuple

        ego_lane = _ConcentricArcLane(radius=17.0, center_y=-17.0)
        ego_path = [
            (float(s), float(ego_lane.position(s)[0]), float(ego_lane.position(s)[1]))
            for s in range(2, 32, 2)
        ]

        # Concentric diverging outer lane (R=20.5m, 3.5m radial separation > r_conf=2.4m)
        outer_lane = _ConcentricArcLane(radius=20.5, center_y=-17.0)
        s_outer = 14.0
        pos_outer = outer_lane.position(s_outer, 0.0)
        h_outer = outer_lane.heading_theta_at(s_outer)
        conflict_div, dist_div = MetaDriveWrapper._check_spatiotemporal_conflict(
            vehicle=_DummyVeh(),
            speed_kmh=25.0,
            ego_path_pts=ego_path,
            obj=_DummyOther(lane=outer_lane),
            obj_pos=pos_outer,
            ovx=7.0 * math.cos(h_outer),
            ovy=7.0 * math.sin(h_outer),
            obj_h=h_outer,
        )
        self.assertFalse(conflict_div)
        self.assertEqual(dist_div, 60.0)

        # True perpendicular intersecting trajectory crossing straight ego path at x=14m
        straight_ego = [(float(s), float(s), 0.0) for s in range(2, 32, 2)]
        conflict_cross, dist_cross = MetaDriveWrapper._check_spatiotemporal_conflict(
            vehicle=_DummyVeh(),
            speed_kmh=36.0,
            ego_path_pts=straight_ego,
            obj=_DummyOther(lane=None),
            obj_pos=np.array([14.0, -5.0], dtype=np.float64),
            ovx=0.0,
            ovy=5.0,
            obj_h=0.5 * math.pi,
        )
        self.assertTrue(conflict_cross)
        self.assertLess(dist_cross, 15.0)


if __name__ == "__main__":
    unittest.main()
