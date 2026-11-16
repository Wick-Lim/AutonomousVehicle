"""Unit tests for Pydantic v2 configuration validation and telemetry input sanitization."""

from __future__ import annotations

import copy
import math
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import yaml
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, DEFAULT_CONFIG_PATH, load_validated_config
from src.schemas import TelemetryInput


class TestConfigAndSchemas(unittest.TestCase):
    """Validate AppConfig invariants, env overrides, and TelemetryInput sanitization."""

    def test_default_config_loads_and_matches_invariants(self) -> None:
        cfg = load_validated_config(DEFAULT_CONFIG_PATH, apply_env_overrides=False)
        self.assertEqual(cfg.project.name, "AutonomousVehicle")
        self.assertEqual(cfg.env.camera_height, 196)
        self.assertEqual(cfg.env.camera_width, 392)
        self.assertEqual(cfg.model.dino.patch_size, 14)
        self.assertEqual(cfg.model.dino.num_patches, 392)
        self.assertEqual(cfg.model.projector.num_visual_tokens, 36)
        as_dict = cfg.to_dict()
        self.assertIsInstance(as_dict, dict)
        self.assertEqual(as_dict["control"]["safety_shield"]["aeb_threshold"], 0.70)

    def test_env_overrides_applied_when_enabled(self) -> None:
        env_patch = {
            "AV_DEVICE": "CPU",
            "AV_USE_MOCK": "true",
            "AV_SIMULATOR": "CARLA",
        }
        with mock.patch.dict(os.environ, env_patch, clear=False):
            cfg = load_validated_config(DEFAULT_CONFIG_PATH, apply_env_overrides=True)
            self.assertEqual(cfg.project.device, "cpu")
            self.assertTrue(cfg.project.use_mock)
            self.assertEqual(cfg.env.simulator, "carla")

            cfg_no_env = load_validated_config(DEFAULT_CONFIG_PATH, apply_env_overrides=False)
            self.assertEqual(cfg_no_env.project.device, "auto")
            self.assertFalse(cfg_no_env.project.use_mock)
            self.assertEqual(cfg_no_env.env.simulator, "metadrive")

    def test_invalid_camera_patch_divisibility_raises(self) -> None:
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        bad = copy.deepcopy(raw)
        bad["env"]["camera_height"] = 197
        with self.assertRaises(ValidationError):
            AppConfig.model_validate(bad)

    def test_invalid_pool_grid_mismatch_raises(self) -> None:
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        bad = copy.deepcopy(raw)
        bad["model"]["projector"]["pool_h"] = 5
        with self.assertRaises(ValidationError):
            AppConfig.model_validate(bad)

    def test_invalid_aeb_threshold_out_of_range_raises(self) -> None:
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        bad = copy.deepcopy(raw)
        bad["control"]["safety_shield"]["aeb_threshold"] = 1.5
        with self.assertRaises(ValidationError):
            AppConfig.model_validate(bad)

    def test_invalid_crop_ratio_sum_raises(self) -> None:
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        bad = copy.deepcopy(raw)
        bad["env"]["crop_top_ratio"] = 0.60
        bad["env"]["crop_bottom_ratio"] = 0.30
        with self.assertRaises(ValidationError):
            AppConfig.model_validate(bad)

    def test_telemetry_sanitizes_nan_inf_negative_and_enums(self) -> None:
        tel = TelemetryInput(
            speed_kmh=float("nan"),
            lane_offset_m=float("inf"),
            heading_error=float("-inf"),
            lead_distance_m=-12.5,
            pedestrian_distance_m=float("nan"),
            lead_speed_kmh=-5.0,
            road_curvature=float("nan"),
            nav_command="  TURN_LEFT  ",
            traffic_light=" RED ",
            hazard_source="PEDESTRIAN",
            wet_road=True,
        )
        self.assertEqual(tel.speed_kmh, 0.0)
        self.assertEqual(tel.lane_offset_m, 0.0)
        self.assertEqual(tel.heading_error, 0.0)
        self.assertEqual(tel.lead_distance_m, 0.0)
        self.assertEqual(tel.pedestrian_distance_m, 100.0)
        self.assertEqual(tel.lead_speed_kmh, 0.0)
        self.assertEqual(tel.road_curvature, 0.0)
        self.assertEqual(tel.nav_command, "turn_left")
        self.assertEqual(tel.traffic_light, "red")
        self.assertEqual(tel.hazard_source, "pedestrian")

        d = tel.to_telemetry_dict()
        self.assertTrue(all(math.isfinite(d[k]) for k in ("speed_kmh", "lane_offset_m", "a_max_mps2")))
        self.assertEqual(d["a_max_mps2"], 4.5)
        self.assertEqual(d["uncertainty_margin_m"], 1.8)

    def test_telemetry_unknown_enum_falls_back_safely(self) -> None:
        tel = TelemetryInput(
            nav_command="u_turn_nowhere",
            traffic_light="purple",
            hazard_source="meteor",
            lead_speed_kmh=None,
        )
        self.assertEqual(tel.nav_command, "go_straight")
        self.assertEqual(tel.traffic_light, "green")
        self.assertEqual(tel.hazard_source, "none")
        d = tel.to_telemetry_dict()
        self.assertNotIn("lead_speed_kmh", d)
        self.assertEqual(d["a_max_mps2"], 6.0)
        self.assertEqual(d["uncertainty_margin_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
