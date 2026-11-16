"""Tests for adapter checkpoint v1.1 round-trip, DrivingRuntimeEngine, TorchScript export, and FastAPI."""

from __future__ import annotations

import base64
import concurrent.futures
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import create_app
from src.config import DEFAULT_CONFIG_PATH, load_validated_config
from src.models.laya_pilot import AutonomousDrivingSystem
from src.runtime.engine import DrivingRuntimeEngine, InvalidImageError
from src.runtime.export import export_spatial_projector


class TestRuntimeAndAPI(unittest.TestCase):
    """Verify v1.1 checkpoints, runtime engine thread safety, TorchScript export, and FastAPI endpoints."""

    def test_adapter_checkpoint_v11_round_trip(self) -> None:
        cfg = load_validated_config(DEFAULT_CONFIG_PATH, apply_env_overrides=False).to_dict()
        sys_a = AutonomousDrivingSystem(config=cfg, use_mock=True)
        # Mutate weights deterministically to distinguish from default init
        with torch.no_grad():
            for p in sys_a.projector.parameters():
                p.add_(0.05)
            for p in sys_a.laya.trajectory_head.parameters():
                p.add_(-0.03)
            for p in sys_a.laya.maneuver_head.parameters():
                p.add_(0.02)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "adapter_v11.pt"
            sys_a.save_adapter_checkpoint(
                ckpt_path,
                epoch=3,
                val_metrics={"val_loss": 0.123},
                config=cfg,
            )
            self.assertTrue(ckpt_path.is_file())

            sys_b = AutonomousDrivingSystem(config=cfg, use_mock=True)
            loaded = sys_b.load_adapter_checkpoint(ckpt_path, map_location="cpu", strict=True)
            self.assertEqual(loaded["schema_version"], "1.1")
            self.assertEqual(loaded["epoch"], 3)

            for (na, pa), (nb, pb) in zip(
                sys_a.projector.named_parameters(),
                sys_b.projector.named_parameters(),
            ):
                self.assertTrue(torch.allclose(pa, pb, atol=1e-7), f"Mismatch in projector.{na}")
            for head_name in ("trajectory_head", "maneuver_head", "speed_head", "aeb_head"):
                mod_a = getattr(sys_a.laya, head_name)
                mod_b = getattr(sys_b.laya, head_name)
                for (na, pa), (nb, pb) in zip(
                    mod_a.named_parameters(), mod_b.named_parameters()
                ):
                    self.assertTrue(
                        torch.allclose(pa, pb, atol=1e-7),
                        f"Mismatch in {head_name}.{na}",
                    )

    def test_torchscript_export_and_verification(self) -> None:
        cfg = load_validated_config(DEFAULT_CONFIG_PATH, apply_env_overrides=False).to_dict()
        sys_model = AutonomousDrivingSystem(config=cfg, use_mock=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "spatial_projector_ts.pt"
            meta = export_spatial_projector(
                sys_model,
                output_path=out_path,
                export_format="torchscript",
                verify=True,
            )
            self.assertTrue(out_path.is_file())
            self.assertGreater(meta["size_bytes"], 0)
            self.assertTrue(meta["verified"])
            self.assertLess(meta["max_abs_diff"], 1e-4)

    def test_runtime_engine_sessions_resize_failsafe_and_threads(self) -> None:
        engine = DrivingRuntimeEngine(use_mock=True, max_sessions=2)
        engine.warmup(passes=1)
        self.assertTrue(engine.ready)

        # Auto-resize 720p RGB frame + vision_estimated step
        hd_rgb = np.full((720, 1280, 3), 128, dtype=np.uint8)
        resp1 = engine.step(
            session_id="veh_a",
            image=hd_rgb,
            telemetry={"speed_kmh": 32.0, "nav_command": "go_straight"},
            perception_mode="vision_estimated",
        )
        self.assertEqual(resp1.session_id, "veh_a")
        self.assertEqual(resp1.step_index, 1)
        self.assertFalse(resp1.failsafe_triggered)

        # Corrupt bytes with strict_input=True raises InvalidImageError
        with self.assertRaises(InvalidImageError):
            engine.step(
                session_id="veh_a",
                image=b"not-a-valid-image",
                strict_input=True,
            )

        # Corrupt bytes with strict_input=False triggers safe AEB failsafe response
        failsafe = engine.step(
            session_id="veh_a",
            image=b"not-a-valid-image",
            strict_input=False,
        )
        self.assertTrue(failsafe.failsafe_triggered)
        self.assertEqual(failsafe.control.throttle_brake, -1.0)
        self.assertEqual(failsafe.control.carla_control.brake, 1.0)

        # Concurrent multi-session step execution (LRU capacity 2 tests locking & eviction)
        def _worker(sid: str) -> int:
            r = engine.step(
                session_id=sid,
                image=None,
                telemetry={"speed_kmh": 25.0},
                perception_mode="gt",
            )
            return r.step_index

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_worker, f"concurrent_{i % 3}") for i in range(8)]
            results = [fut.result() for fut in futures]
        self.assertTrue(all(idx >= 1 for idx in results))
        self.assertLessEqual(engine.get_system_info().active_sessions, 2)

    def test_fastapi_endpoints_and_prometheus(self) -> None:
        app = create_app(use_mock=True)
        png_ok, encoded = cv2.imencode(".png", np.full((196, 392, 3), 90, dtype=np.uint8))
        self.assertTrue(png_ok)
        png_bytes = encoded.tobytes()
        img_b64 = base64.b64encode(png_bytes).decode("ascii")

        with TestClient(app) as client:
            r_health = client.get("/healthz")
            self.assertEqual(r_health.status_code, 200)
            self.assertEqual(r_health.json()["status"], "ok")
            self.assertIn("X-Request-ID", r_health.headers)
            self.assertIn("X-Process-Time-Ms", r_health.headers)

            r_ready = client.get("/readyz")
            self.assertEqual(r_ready.status_code, 200)
            self.assertTrue(r_ready.json()["ready"])

            r_info = client.get("/v1/info")
            self.assertEqual(r_info.status_code, 200)
            self.assertEqual(r_info.json()["trainable_parameters"], 4_232_991)

            r_step = client.post(
                "/v1/step",
                json={
                    "session_id": "api_car_1",
                    "image_base64": img_b64,
                    "telemetry": {
                        "speed_kmh": 30.0,
                        "lead_distance_m": 25.0,
                        "nav_command": "go_straight",
                    },
                    "perception_mode": "vision_estimated",
                    "shield_enabled": True,
                },
            )
            self.assertEqual(r_step.status_code, 200, r_step.text)
            step_data = r_step.json()
            self.assertEqual(step_data["step_index"], 1)
            self.assertIn("carla_control", step_data["control"])

            # Corrupt base64 returns HTTP 400
            r_bad = client.post(
                "/v1/step",
                json={"session_id": "api_car_1", "image_base64": "%%%invalid_base64%%%"},
            )
            self.assertEqual(r_bad.status_code, 400)

            # Multipart PNG upload
            r_mp = client.post(
                "/v1/step/multipart",
                files={"image": ("frame.png", png_bytes, "image/png")},
                data={
                    "session_id": "api_car_1",
                    "telemetry_json": json.dumps({"speed_kmh": 28.0, "wet_road": True}),
                    "perception_mode": "gt",
                },
            )
            self.assertEqual(r_mp.status_code, 200, r_mp.text)
            self.assertEqual(r_mp.json()["step_index"], 2)

            # Session reset & delete
            r_reset = client.post("/v1/sessions/api_car_1/reset")
            self.assertEqual(r_reset.status_code, 200)
            self.assertEqual(r_reset.json()["status"], "reset")

            r_del = client.delete("/v1/sessions/api_car_1")
            self.assertEqual(r_del.status_code, 200)
            self.assertEqual(r_del.json()["status"], "deleted")

            r_del_again = client.delete("/v1/sessions/api_car_1")
            self.assertEqual(r_del_again.status_code, 200)
            self.assertEqual(r_del_again.json()["status"], "not_found")

            # Metrics JSON & Prometheus exposition
            r_m_json = client.get("/v1/metrics")
            self.assertEqual(r_m_json.status_code, 200)
            self.assertGreaterEqual(r_m_json.json()["total_steps"], 2)

            r_prom = client.get("/metrics")
            self.assertEqual(r_prom.status_code, 200)
            self.assertIn("av_steps_total", r_prom.text)


if __name__ == "__main__":
    unittest.main()
