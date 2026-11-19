"""CARLA Leaderboard 2.0 / Bench2Drive Official `AutonomousAgent` Adapter (Track 1-B).

Implements the standard `AutonomousAgent` contract:
  - `setup(path_to_conf_file)`
  - `sensors()`
  - `set_global_plan(global_plan_gps, global_plan_world_coord)`
  - `run_step(input_data, timestamp)` -> `carla.VehicleControl` (or compatible dataclass)
  - `destroy()`

Dynamically inherits from `leaderboard.autoagents.autonomous_agent.AutonomousAgent`
when the official CARLA Leaderboard 2.0 / Bench2Drive package is installed, while
remaining 100% runnable standalone in local evaluation environments.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from src.runtime.engine import DrivingRuntimeEngine

try:
    from leaderboard.autoagents.autonomous_agent import (  # type: ignore
        AutonomousAgent as _BaseAutonomousAgent,
        Track as _LeaderboardTrack,
    )

    LEADERBOARD_INSTALLED = True
except Exception:
    LEADERBOARD_INSTALLED = False

    class _LeaderboardTrack(Enum):  # type: ignore[no-redef]
        SENSORS = "SENSORS"
        MAP = "MAP"

    class _BaseAutonomousAgent:  # type: ignore[no-redef]
        """Standalone drop-in base class matching CARLA Leaderboard 2.0 AutonomousAgent."""

        track = _LeaderboardTrack.SENSORS

        def __init__(self, carla_host: str = "127.0.0.1", carla_port: int = 2000, debug: bool = False) -> None:
            self.carla_host = carla_host
            self.carla_port = carla_port
            self.debug = debug
            self._global_plan: List[Any] = []
            self._global_plan_world_coord: List[Any] = []

        def setup(self, path_to_conf_file: str) -> None:
            pass

        def sensors(self) -> List[Dict[str, Any]]:
            return []

        def set_global_plan(
            self,
            global_plan_gps: Sequence[Any],
            global_plan_world_coord: Sequence[Any],
        ) -> None:
            self._global_plan = list(global_plan_gps or [])
            self._global_plan_world_coord = list(global_plan_world_coord or [])

        def run_step(self, input_data: Dict[str, Any], timestamp: float) -> Any:
            raise NotImplementedError

        def destroy(self) -> None:
            pass

        def __call__(self, input_data: Dict[str, Any], timestamp: float = 0.0) -> Any:
            return self.run_step(input_data, timestamp)


@dataclass
class VehicleControlLike:
    """Fallback control structure matching `carla.VehicleControl` attributes."""

    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    hand_brake: bool = False
    reverse: bool = False
    manual_gear_shift: bool = False
    gear: int = 0


def _road_option_to_nav_command(road_opt: Any) -> str:
    """Map CARLA Leaderboard RoadOption enum/int/str into Laya nav_command vocabulary."""
    name = str(getattr(road_opt, "name", road_opt)).strip().upper()
    if "LEFT" in name:
        return "turn_left"
    if "RIGHT" in name:
        return "turn_right"
    if "MERGE" in name or "CHANGELANE" in name:
        return "merge"
    return "go_straight"


def _make_carla_vehicle_control(ctrl_dict: Dict[str, Any]) -> Any:
    """Return a native `carla.VehicleControl` when `carla` is available, else `VehicleControlLike`."""
    steer = float(np.clip(ctrl_dict.get("steer", 0.0), -1.0, 1.0))
    throttle = float(np.clip(ctrl_dict.get("throttle", 0.0), 0.0, 1.0))
    brake = float(np.clip(ctrl_dict.get("brake", 0.0), 0.0, 1.0))
    hand_brake = bool(ctrl_dict.get("hand_brake", False))
    reverse = bool(ctrl_dict.get("reverse", False))
    try:
        import carla  # type: ignore

        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=hand_brake,
            reverse=reverse,
        )
    except Exception:
        return VehicleControlLike(
            steer=steer,
            throttle=throttle,
            brake=brake,
            hand_brake=hand_brake,
            reverse=reverse,
        )


def get_entry_point() -> str:
    """Standard CARLA Leaderboard 2.0 / Bench2Drive agent discovery entrypoint."""
    return "DINOv2LayaLeaderboardAgent"


class DINOv2LayaLeaderboardAgent(_BaseAutonomousAgent):
    """CARLA Leaderboard 2.0 / Bench2Drive compliant DINOv2 + SpatialProjector + Laya Agent."""

    track = _LeaderboardTrack.SENSORS

    def __init__(
        self,
        carla_host: str = "127.0.0.1",
        carla_port: int = 2000,
        debug: bool = False,
    ) -> None:
        super().__init__(carla_host=carla_host, carla_port=carla_port, debug=debug)
        self.engine: Optional[DrivingRuntimeEngine] = None
        self.session_id: str = "carla_leaderboard_ego"
        self.control_mode: str = "pure_e2e_trajectory"
        self.perception_mode: str = "vision_radar_fusion"
        self.shield_enabled: bool = False
        self._last_timestamp: Optional[float] = None
        self._last_response: Any = None
        self._global_plan_gps: List[Tuple[Dict[str, float], Any]] = []
        self._global_plan_world: List[Tuple[Any, Any]] = []

    def setup(self, path_to_conf_file: Optional[str] = None) -> None:
        """Initialize DrivingRuntimeEngine from YAML/JSON agent configuration file."""
        agent_cfg: Dict[str, Any] = {}
        if path_to_conf_file:
            p = Path(path_to_conf_file)
            if p.is_file():
                text = p.read_text(encoding="utf-8")
                if p.suffix.lower() == ".json":
                    agent_cfg = json.loads(text) or {}
                else:
                    agent_cfg = yaml.safe_load(text) or {}

        self.control_mode = str(agent_cfg.get("control_mode", "pure_e2e_trajectory"))
        self.perception_mode = str(agent_cfg.get("perception_mode", "vision_radar_fusion"))
        self.shield_enabled = bool(
            agent_cfg.get("shield_enabled", self.control_mode != "pure_e2e_trajectory")
        )
        use_mock = bool(agent_cfg.get("use_mock", False))
        config_path = agent_cfg.get("config_path")
        proj_ckpt = agent_cfg.get("projector_ckpt")
        temp_ckpt = agent_cfg.get("temperature_ckpt")

        self.engine = DrivingRuntimeEngine(
            config_path=config_path,
            use_mock=use_mock,
            projector_ckpt=proj_ckpt,
            temperature_ckpt=temp_ckpt,
        )

        decision_mode = agent_cfg.get("decision_mode")
        trajectory_mode = agent_cfg.get("trajectory_mode")
        if decision_mode or trajectory_mode:
            self.engine.system.set_policy_modes(
                decision_mode=decision_mode,
                trajectory_mode=trajectory_mode,
            )

        self.engine.warmup(passes=1)
        self._last_timestamp = None

    def sensors(self) -> List[Dict[str, Any]]:
        """Declare CARLA Leaderboard 2.0 / Bench2Drive sensor suite."""
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 1.6,
                "y": 0.0,
                "z": 1.55,
                "roll": 0.0,
                "pitch": -3.0,
                "yaw": 0.0,
                "width": 392,
                "height": 196,
                "fov": 100.0,
                "id": "rgb_front",
            },
            {
                "type": "sensor.other.radar",
                "x": 2.2,
                "y": 0.0,
                "z": 0.65,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "horizontal_fov": 60.0,
                "vertical_fov": 20.0,
                "range": 65.0,
                "id": "radar_front",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "id": "gps",
            },
            {
                "type": "sensor.speedometer",
                "reading_frequency": 20,
                "id": "speed",
            },
        ]

    def set_global_plan(
        self,
        global_plan_gps: Sequence[Any],
        global_plan_world_coord: Sequence[Any],
    ) -> None:
        """Receive high-level route waypoints and RoadOption commands from Leaderboard runner."""
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._global_plan_gps = list(global_plan_gps or [])
        self._global_plan_world = list(global_plan_world_coord or [])

    def _resolve_nav_command(self, gps_data: Any) -> str:
        """Match current GNSS `(lat, lon)` against global plan to select active nav_command."""
        if not self._global_plan_gps:
            return "go_straight"
        if gps_data is None:
            return _road_option_to_nav_command(self._global_plan_gps[0][1])

        try:
            lat = float(gps_data[0])
            lon = float(gps_data[1])
            best_idx = 0
            best_d = float("inf")
            for i, (wp_gps, _opt) in enumerate(self._global_plan_gps):
                w_lat = float(wp_gps.get("lat", 0.0)) if isinstance(wp_gps, dict) else float(wp_gps[0])
                w_lon = float(wp_gps.get("lon", 0.0)) if isinstance(wp_gps, dict) else float(wp_gps[1])
                d = math.hypot(w_lat - lat, w_lon - lon)
                if d < best_d:
                    best_d = d
                    best_idx = i
            look_idx = min(len(self._global_plan_gps) - 1, best_idx + 1)
            return _road_option_to_nav_command(self._global_plan_gps[look_idx][1])
        except Exception:
            return "go_straight"

    @staticmethod
    def _extract_rgb(input_data: Dict[str, Any]) -> np.ndarray:
        raw = input_data.get("rgb_front")
        if raw is None:
            return np.zeros((196, 392, 3), dtype=np.uint8)
        frame = raw[1] if isinstance(raw, (tuple, list)) and len(raw) >= 2 else raw
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[2] == 4:
            # CARLA RGB camera delivers BGRA uint8 -> convert to RGB
            arr = arr[:, :, :3][:, :, ::-1]
        return np.ascontiguousarray(arr, dtype=np.uint8)

    @staticmethod
    def _extract_radar_targets(input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw = input_data.get("radar_front")
        if raw is None:
            return []
        pts = raw[1] if isinstance(raw, (tuple, list)) and len(raw) >= 2 else raw
        if pts is None:
            return []
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 4:
            return []
        # Leaderboard radar format: [velocity, azimuth, altitude, depth]
        targets: List[Dict[str, Any]] = []
        for row in arr[:32]:
            vel, azim, _alt, depth = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            if not (0.5 < depth < 62.0):
                continue
            x_m = depth * math.cos(azim)
            y_m = depth * math.sin(azim)  # +y right
            targets.append(
                {
                    "x_m": round(x_m, 2),
                    "y_m": round(y_m, 2),
                    "vx_rel_mps": round(vel * math.cos(azim), 2),
                    "vy_rel_mps": round(vel * math.sin(azim), 2),
                    "category": "vehicle",
                    "source": "carla_leaderboard_radar",
                }
            )
        return targets

    def run_step(self, input_data: Dict[str, Any], timestamp: float) -> Any:
        """Execute one synchronous Leaderboard 2.0 control tick and return `carla.VehicleControl`."""
        if self.engine is None:
            self.setup(None)
        assert self.engine is not None

        dt = (
            float(np.clip(timestamp - self._last_timestamp, 0.01, 0.25))
            if self._last_timestamp is not None
            else 0.05
        )
        self._last_timestamp = float(timestamp)

        rgb = self._extract_rgb(input_data)

        # Speedometer sensor returns {'speed': m/s}
        speed_raw = input_data.get("speed")
        speed_val = (
            speed_raw[1] if isinstance(speed_raw, (tuple, list)) and len(speed_raw) >= 2 else speed_raw
        )
        if isinstance(speed_val, dict):
            speed_mps = float(speed_val.get("speed", 0.0))
        elif speed_val is not None:
            speed_mps = float(speed_val)
        else:
            speed_mps = 0.0
        speed_kmh = max(0.0, speed_mps * 3.6)

        # IMU sensor returns [ax, ay, az, gx, gy, gz, compass]
        imu_raw = input_data.get("imu")
        imu_val = imu_raw[1] if isinstance(imu_raw, (tuple, list)) and len(imu_raw) >= 2 else imu_raw
        imu_tel: Dict[str, float] = {}
        if imu_val is not None and len(imu_val) >= 6:
            imu_tel = {
                "long_accel_mps2": float(imu_val[0]),
                "lat_accel_mps2": float(imu_val[1]),
                "yaw_rate_rps": float(imu_val[5]),
                "compass_deg": float(math.degrees(imu_val[6])) if len(imu_val) >= 7 else 0.0,
            }

        gps_raw = input_data.get("gps")
        gps_val = gps_raw[1] if isinstance(gps_raw, (tuple, list)) and len(gps_raw) >= 2 else gps_raw
        nav_cmd = self._resolve_nav_command(gps_val)
        radar_targets = self._extract_radar_targets(input_data)

        telemetry: Dict[str, Any] = {
            "speed_kmh": round(speed_kmh, 2),
            "lane_offset_m": 0.0,
            "heading_error": 0.0,
            "left_lane_available": True,
            "right_lane_available": True,
            "nav_command": nav_cmd,
            "radar_targets": radar_targets,
            "imu_telemetry": imu_tel,
        }

        resp = self.engine.step(
            session_id=self.session_id,
            image=rgb,
            telemetry=telemetry,
            perception_mode=self.perception_mode,  # type: ignore[arg-type]
            shield_enabled=self.shield_enabled,
            dt=dt,
            strict_input=False,
            control_mode=self.control_mode,  # type: ignore[arg-type]
        )
        self._last_response = resp
        return _make_carla_vehicle_control(resp.control.carla_control.model_dump())

    def destroy(self) -> None:
        """Clean up active inference session state."""
        if self.engine is not None:
            self.engine.delete_session(self.session_id)
        self._last_timestamp = None


__all__ = [
    "DINOv2LayaLeaderboardAgent",
    "LEADERBOARD_INSTALLED",
    "VehicleControlLike",
    "get_entry_point",
]
