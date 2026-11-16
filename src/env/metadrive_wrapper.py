"""MetaDrive Simulator Wrapper for 196x392 Wide RGB Vision and Structured Telemetry Extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


NAV_COMMANDS = ("go_straight", "turn_left", "turn_right", "merge")


@dataclass
class TelemetryState:
    """Structured vehicle telemetry matching the Section 3.1 I/O specification."""

    speed_kmh: float
    lane_offset_m: float
    heading_error: float
    left_lane_available: bool
    right_lane_available: bool
    nav_command: str
    lead_distance_m: float = 60.0
    lead_speed_kmh: float = 60.0
    pedestrian_distance_m: float = 100.0
    road_curvature: float = 0.0
    expert_maneuver: Optional[str] = None
    traffic_light: str = "green"
    hazard_source: str = "none"
    blind_spot_left: bool = False
    blind_spot_right: bool = False
    wet_road: bool = False
    a_max_mps2: float = 6.0
    uncertainty_margin_m: float = 0.0

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert telemetry to a clean, rounded dictionary for JSON serialization into Laya."""
        return {
            "speed_kmh": round(float(self.speed_kmh), 2),
            "lane_offset_m": round(float(self.lane_offset_m), 3),
            "heading_error": round(float(self.heading_error), 4),
            "left_lane_available": bool(self.left_lane_available),
            "right_lane_available": bool(self.right_lane_available),
            "nav_command": str(self.nav_command),
            "lead_distance_m": round(float(self.lead_distance_m), 2),
            "lead_speed_kmh": round(float(self.lead_speed_kmh), 2),
            "pedestrian_distance_m": round(float(self.pedestrian_distance_m), 2),
            "road_curvature": round(float(self.road_curvature), 4),
            "expert_maneuver": self.expert_maneuver,
            "traffic_light": str(self.traffic_light),
            "hazard_source": str(self.hazard_source),
            "blind_spot_left": bool(self.blind_spot_left),
            "blind_spot_right": bool(self.blind_spot_right),
            "wet_road": bool(self.wet_road),
            "a_max_mps2": round(float(self.a_max_mps2), 2),
            "uncertainty_margin_m": round(float(self.uncertainty_margin_m), 2),
        }


class MetaDriveWrapper:
    """Environment wrapper around MetaDrive providing 196x392 RGB frames and structured telemetry.

    Supports both the real `metadrive-simulator` 3D engine and a deterministic kinematic/visual
    mock simulator (`use_mock=True`) for rapid unit testing and offline pipeline verification.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_mock: bool = False,
        start_seed: Optional[int] = None,
        num_scenarios: int = 100,
    ) -> None:
        self.cfg = config or {}
        env_cfg = self.cfg.get("env", {})

        self.camera_height: int = int(env_cfg.get("camera_height", 196))
        self.camera_width: int = int(env_cfg.get("camera_width", 392))
        self.crop_top_ratio: float = float(env_cfg.get("crop_top_ratio", 0.25))
        self.crop_bottom_ratio: float = float(env_cfg.get("crop_bottom_ratio", 0.10))
        self.decision_repeat: int = int(env_cfg.get("decision_repeat", 5))
        self.traffic_density: float = float(env_cfg.get("traffic_density", 0.18))
        self.map_blocks: int = int(env_cfg.get("map_blocks", 5))
        self.max_episode_steps: int = int(env_cfg.get("max_episode_steps", 400))
        self.lane_width: float = float(env_cfg.get("lane_width", 3.5))
        self.start_seed: int = int(
            start_seed if start_seed is not None else env_cfg.get("start_seed", 1000)
        )
        self.num_scenarios: int = int(num_scenarios)

        self.weather: str = str(env_cfg.get("weather", "clear")).lower()
        self.precipitation: float = float(env_cfg.get("precipitation", 0.0))
        self._is_wet: bool = bool(
            self.weather in ("rain", "wet", "heavy_rain", "hardrain", "rainy")
            or self.precipitation > 0.2
        )

        self.use_mock: bool = bool(
            use_mock or self.cfg.get("project", {}).get("use_mock", False)
        )
        self._raw_env: Any = None
        self._expert_policy: Any = None
        self._routing_target_lane: Any = None
        self._lc_quintic: Optional[Dict[str, Any]] = None
        self._lane_change_hold_steps: int = 0
        self._last_road_curvature: float = 0.0
        self._last_lead_distance_m: float = 60.0
        self._last_ped_distance_m: float = 100.0

        # Internal state for mock/kinematic simulation
        self._rng = np.random.default_rng(self.start_seed)
        self._current_seed: int = self.start_seed
        self._step_count: int = 0
        self._progress: float = 0.0
        self._route_length_m: float = 180.0
        self._num_lanes: int = 3
        self._current_lane_idx: int = 1
        self._target_lane_idx: int = 1
        self._speed_kmh: float = 38.0
        self._lane_offset_m: float = 0.0
        self._heading_error: float = 0.0
        self._road_curvature: float = 0.0
        self._nav_command: str = "go_straight"
        self._lead_distance_m: float = 45.0
        self._lead_speed_kmh: float = 40.0
        self._scenario_event: str = "keep_lane"
        self._prev_vel_xy: Optional[Tuple[float, float]] = None
        self._prev_heading: Optional[float] = None
        self._ema_road_curvature: float = 0.0
        self._reroute_count: int = 0
        self._mock_dest_idx: int = 3
        self._last_imu_telemetry: Dict[str, float] = {
            "long_accel_mps2": 0.0,
            "lat_accel_mps2": 0.0,
            "yaw_rate_rps": 0.0,
            "pitch_rad": 0.0,
        }

        if not self.use_mock:
            self._init_real_metadrive()

    def _init_real_metadrive(self) -> None:
        """Initialize the real MetaDriveEnv with RGBCamera and procedural traffic."""
        try:
            from metadrive.envs.metadrive_env import MetaDriveEnv
            from metadrive.component.sensors.rgb_camera import RGBCamera
            from metadrive.engine.engine_utils import close_engine, engine_initialized
        except ImportError as exc:
            raise ImportError(
                "MetaDrive is not installed. Install `metadrive-simulator` or run with `--mock`."
            ) from exc

        # Panda3D allows a single active engine per process; clean up any stale engine first.
        try:
            if engine_initialized():
                close_engine()
        except Exception:
            pass

        use_render = bool(self.cfg.get("env", {}).get("use_render", False))
        md_config = {
            "use_render": use_render,
            "image_observation": True,
            "stack_size": 1,
            "sensors": {
                "rgb_camera": (RGBCamera, self.camera_width, self.camera_height)
            },
            "vehicle_config": {"image_source": "rgb_camera", "enable_reverse": False},
            "decision_repeat": self.decision_repeat,
            "traffic_density": self.traffic_density,
            "map": self.map_blocks,
            "start_seed": self.start_seed,
            "num_scenarios": max(1, self.num_scenarios),
            "horizon": self.max_episode_steps,
            "out_of_road_penalty": 10.0,
            "crash_vehicle_penalty": 10.0,
        }
        self._raw_env = MetaDriveEnv(md_config)

    @property
    def step_dt(self) -> float:
        """Simulated elapsed seconds per control step in MetaDrive 3D or mock."""
        if not self.use_mock and self._raw_env is not None:
            raw_cfg = getattr(self._raw_env, "config", {}) or {}
            return float(raw_cfg.get("physics_world_step_size", 0.02)) * float(
                raw_cfg.get("decision_repeat", self.decision_repeat)
            )
        return 0.05 * float(self.decision_repeat)

    def set_weather(self, weather: str, precipitation: float = 0.0) -> None:
        """Update weather/wet-road physics friction for real MetaDrive 3D or mock."""
        self.weather = str(weather or "clear").strip().lower()
        self.precipitation = float(precipitation)
        self._is_wet = bool(
            self.weather in ("rain", "wet", "heavy_rain", "hardrain", "rainy", "night_rain")
            or self.precipitation > 0.2
        )
        if self._raw_env is not None:
            try:
                vehicle = getattr(self._raw_env, "agent", None)
                if vehicle is not None:
                    target_friction = 0.62 if "heavy" in self.weather else (0.74 if self._is_wet else 0.90)
                    v_cfg = getattr(vehicle, "config", None)
                    if isinstance(v_cfg, dict):
                        v_cfg["wheel_friction"] = target_friction
                    wheels = getattr(vehicle, "wheels", None)
                    if wheels:
                        for w in wheels:
                            if hasattr(w, "setFrictionSlip"):
                                w.setFrictionSlip(target_friction)
            except Exception:
                pass

    def set_map_blocks(self, map_blocks: int) -> int:
        """Dynamically reconfigure procedural road block count (1..8) and reinitialize MetaDrive."""
        blocks = max(1, min(8, int(map_blocks)))
        if blocks == self.map_blocks:
            return self.map_blocks
        self.map_blocks = blocks
        if not self.use_mock:
            self.close()
            self._init_real_metadrive()
        return self.map_blocks

    def list_spawn_points(self) -> List[Dict[str, Any]]:
        """List available road-network destination nodes on the active MetaDrive map."""
        if not self.use_mock and self._raw_env is not None:
            try:
                vehicle = getattr(self._raw_env, "agent", None)
                rn = getattr(getattr(getattr(vehicle, "navigation", None), "map", None), "road_network", None)
                graph = getattr(rn, "graph", {}) or {}
                nodes: List[Dict[str, Any]] = []
                idx = 0
                for u_node, out_d in graph.items():
                    if str(u_node).startswith("-") or not isinstance(out_d, dict):
                        continue
                    for v_node, lanes in out_d.items():
                        if not lanes:
                            continue
                        lane0 = lanes[0]
                        llen = float(getattr(lane0, "length", 10.0))
                        pt = lane0.position(llen, 0.0)
                        nodes.append(
                            {
                                "index": idx,
                                "x": round(float(pt[0]), 2),
                                "y": round(float(pt[1]), 2),
                                "u_node": str(u_node),
                                "v_node": str(v_node),
                                "lane_index": list(lane0.index),
                            }
                        )
                        idx += 1
                if nodes:
                    return nodes
            except Exception:
                pass

        return [
            {"index": i, "x": round(45.0 * (i + 1), 1), "y": round(((i % 3) - 1) * 18.0, 1)}
            for i in range(6)
        ]

    def set_destination(
        self,
        dest_spawn_idx: Optional[int] = None,
        dest_xy: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Dynamically re-route live MetaDrive navigation to a chosen road-network node."""
        nodes = self.list_spawn_points()
        if not nodes:
            return {"rerouted": False, "reroute_count": self._reroute_count}

        chosen = nodes[-1]
        if dest_spawn_idx is not None:
            chosen = nodes[int(dest_spawn_idx) % len(nodes)]
        elif dest_xy is not None:
            tx, ty = float(dest_xy[0]), float(dest_xy[1])
            chosen = min(
                nodes,
                key=lambda n: math.hypot(float(n["x"]) - tx, float(n["y"]) - ty),
            )

        if not self.use_mock and self._raw_env is not None:
            try:
                vehicle = getattr(self._raw_env, "agent", None)
                nav = getattr(vehicle, "navigation", None)
                cur_lane = getattr(vehicle, "lane", None)
                target_lane_idx = tuple(chosen.get("lane_index", ()))
                if nav is not None and cur_lane is not None and len(target_lane_idx) == 3:
                    nav.set_route(cur_lane.index, target_lane_idx)
                    self._lc_quintic = None
                    self._routing_target_lane = None
                    self._reroute_count += 1
                    return {
                        "rerouted": True,
                        "dest_index": int(chosen["index"]),
                        "dest_xy": [float(chosen["x"]), float(chosen["y"])],
                        "reroute_count": self._reroute_count,
                    }
            except Exception:
                pass

        # Mock mode dynamic route extension / target update
        self._mock_dest_idx = int(chosen["index"])
        self._route_length_m = max(self._progress + 95.0, float(chosen["x"]))
        self._reroute_count += 1
        return {
            "rerouted": True,
            "dest_index": int(chosen["index"]),
            "dest_xy": [float(chosen["x"]), float(chosen["y"])],
            "reroute_count": self._reroute_count,
        }

    def _build_real_nav_state(self, vehicle: Any, route_comp: float) -> Dict[str, Any]:
        """Build lightweight 2D city/procedural map state for live WebSocket cockpit rendering."""
        ego_x = float(vehicle.position[0])
        ego_y = float(vehicle.position[1])
        ego_yaw = float(getattr(vehicle, "heading_theta", 0.0))
        route_xy: List[List[float]] = []
        dest_xy: List[float] = [ego_x + 50.0, ego_y]
        rem_m = 0.0
        try:
            nav = getattr(vehicle, "navigation", None)
            rn = getattr(getattr(nav, "map", None), "road_network", None)
            ckpts = list(getattr(nav, "checkpoints", []) or [])
            if rn is not None and len(ckpts) >= 2:
                for idx_c in range(len(ckpts) - 1):
                    lanes = rn.graph.get(ckpts[idx_c], {}).get(ckpts[idx_c + 1], [])
                    if not lanes:
                        continue
                    ln = lanes[min(1, len(lanes) - 1)]
                    llen = float(getattr(ln, "length", 10.0))
                    rem_m += llen
                    for s_pt in np.linspace(0.0, llen, num=max(2, int(llen // 8) + 2)):
                        p = ln.position(float(s_pt), 0.0)
                        route_xy.append([round(float(p[0]), 1), round(float(p[1]), 1)])
                if route_xy:
                    dest_xy = route_xy[-1]
        except Exception:
            pass
        rem_est = max(0.0, rem_m * (1.0 - float(np.clip(route_comp, 0.0, 1.0))))
        return {
            "map_name": f"MetaDrive-{self.map_blocks}B-S{self._current_seed}",
            "ego_xy_yaw": [round(ego_x, 2), round(ego_y, 2), round(ego_yaw, 3)],
            "dest_xy": dest_xy,
            "active_route_xy": route_xy[:90],
            "spawn_nodes": self.list_spawn_points()[:24],
            "remaining_route_m": round(rem_est, 1),
            "reroute_count": int(self._reroute_count),
        }

    def preprocess_frame(self, raw_img: np.ndarray) -> np.ndarray:
        """Crop top sky / bottom hood and resize to (196, 392, 3) uint8 RGB."""
        if raw_img.ndim == 4:
            raw_img = raw_img[..., -1]
        if raw_img.dtype != np.uint8:
            if raw_img.max() <= 1.0:
                raw_img = (np.clip(raw_img, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                raw_img = np.clip(raw_img, 0, 255).astype(np.uint8)

        h, w = raw_img.shape[:2]
        if (h, w) != (self.camera_height, self.camera_width):
            top = int(h * self.crop_top_ratio)
            bottom = int(h * (1.0 - self.crop_bottom_ratio))
            if bottom > top + 16:
                raw_img = raw_img[top:bottom, :]
            raw_img = cv2.resize(
                raw_img,
                (self.camera_width, self.camera_height),
                interpolation=cv2.INTER_AREA,
            )
        return raw_img

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Reset environment and return (rgb_196x392, telemetry_dict, info_dict)."""
        if seed is not None:
            self._current_seed = int(seed)
        else:
            self._current_seed += 1

        self._prev_vel_xy = None
        self._prev_heading = None
        self._routing_target_lane = None
        self._lc_quintic = None
        self._lane_change_hold_steps = 0
        self._last_road_curvature = 0.0
        self._ema_road_curvature = 0.0
        self._reroute_count = 0
        self._last_lead_distance_m = 60.0
        self._last_ped_distance_m = 100.0
        self._last_gated_maneuver = "keep_lane"
        self._last_imu_telemetry = {
            "long_accel_mps2": 0.0,
            "lat_accel_mps2": 0.0,
            "yaw_rate_rps": 0.0,
            "pitch_rad": 0.0,
        }

        if not self.use_mock and self._raw_env is not None:
            # Map arbitrary external seeds safely into MetaDrive's [start_seed, start_seed + num_scenarios) range.
            md_seed = self.start_seed + (
                (self._current_seed - self.start_seed) % max(1, self.num_scenarios)
            )
            obs, info = self._raw_env.reset(seed=md_seed)
            self._step_count = 0
            self._routing_target_lane = None
            self._prev_routing_lane = None
            self._lc_quintic = None
            self._expert_policy = None
            self._last_gated_maneuver = "keep_lane"
            self.set_weather(self.weather, self.precipitation)
            rgb = self._extract_real_image(obs)
            telemetry, extra_info = self._extract_real_telemetry(info)
            return rgb, telemetry.to_json_dict(), extra_info

        # Mock kinematic scenario initialization
        self._rng = np.random.default_rng(self._current_seed)
        self._step_count = 0
        self._progress = 0.0
        self._route_length_m = float(self._rng.uniform(155.0, 195.0))
        self._num_lanes = 3
        self._current_lane_idx = int(self._rng.integers(0, self._num_lanes))
        self._target_lane_idx = self._current_lane_idx
        self._speed_kmh = float(self._rng.uniform(32.0, 44.0))
        self._lane_offset_m = float(self._rng.normal(0.0, 0.12))
        self._heading_error = float(self._rng.normal(0.0, 0.02))
        self._sample_mock_traffic_event()

        telemetry = self._build_mock_telemetry()
        rgb = self._render_mock_frame()
        info = self._build_mock_info(arrive_dest=False, crash=False, out_of_road=False)
        return rgb, telemetry.to_json_dict(), info

    def step(
        self, action: Tuple[float, float] | np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute [steering, throttle_brake] in [-1, 1] and return next state."""
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))
        self._step_count += 1

        if not self.use_mock and self._raw_env is not None:
            ego_v_raw = float(getattr(self._raw_env.agent, "speed_km_h", 0.0))
            if ego_v_raw < 0.8 and throttle_brake < 0.0:
                throttle_brake = max(-0.04, throttle_brake)
            obs, reward, terminated, truncated, info = self._raw_env.step(
                [steering, throttle_brake]
            )
            done = bool(terminated or truncated)
            rgb = self._extract_real_image(obs)
            telemetry, extra_info = self._extract_real_telemetry(info)
            return rgb, telemetry.to_json_dict(), float(reward), done, extra_info

        # Kinematic bicycle step
        dt = 0.05 * self.decision_repeat
        accel_kmh_s = (
            throttle_brake * 35.0 if throttle_brake >= 0.0 else throttle_brake * 65.0
        )
        self._speed_kmh = float(np.clip(self._speed_kmh + accel_kmh_s * dt, 0.0, 80.0))
        speed_ms = self._speed_kmh / 3.6

        # Update heading and lateral offset with mild curvature disturbance
        self._heading_error = float(
            np.clip(
                0.75 * self._heading_error
                + steering * 0.18
                - self._road_curvature * 0.008,
                -0.45,
                0.45,
            )
        )
        self._lane_offset_m = float(
            self._lane_offset_m + speed_ms * math.sin(self._heading_error) * dt
        )

        # Handle discrete lane transition if offset crosses lane boundary
        half_lane = self.lane_width * 0.52
        if self._lane_offset_m > half_lane and self._current_lane_idx > 0:
            self._current_lane_idx -= 1
            self._lane_offset_m -= self.lane_width
            if self._scenario_event == "change_left":
                self._lead_distance_m = 65.0
                self._lead_speed_kmh = 65.0
                self._scenario_event = "clear_highway"
        elif (
            self._lane_offset_m < -half_lane
            and self._current_lane_idx < self._num_lanes - 1
        ):
            self._current_lane_idx += 1
            self._lane_offset_m += self.lane_width
            if self._scenario_event == "change_right":
                self._lead_distance_m = 65.0
                self._lead_speed_kmh = 65.0
                self._scenario_event = "clear_highway"

        # Update longitudinal progress and lead vehicle distance
        self._progress += speed_ms * dt
        rel_speed_ms = (self._lead_speed_kmh - self._speed_kmh) / 3.6
        self._lead_distance_m = float(
            np.clip(self._lead_distance_m + rel_speed_ms * dt, 1.5, 90.0)
        )

        # If AEB hazard was resolved by braking/slowing down, lead vehicle moves away
        if self._scenario_event == "emergency_brake" and (
            self._speed_kmh < 12.0 or throttle_brake <= -0.5
        ):
            self._lead_speed_kmh = 45.0
            self._lead_distance_m = max(self._lead_distance_m, 28.0)
            self._scenario_event = "keep_lane"

        # Periodically transition traffic scenario in mock mode
        if self._step_count % 18 == 0:
            self._sample_mock_traffic_event()

        # Check termination conditions
        out_of_road = abs(self._lane_offset_m) > (self.lane_width * 0.98)
        crash_lead = self._lead_distance_m < 2.2 and self._speed_kmh > 15.0
        crash = bool(out_of_road or crash_lead)
        arrive_dest = bool(self._progress >= self._route_length_m and not crash)
        timeout = self._step_count >= self.max_episode_steps
        done = bool(crash or arrive_dest or timeout)

        reward = (
            (speed_ms * 0.1)
            - abs(self._lane_offset_m) * 0.15
            - (10.0 if crash else 0.0)
            + (20.0 if arrive_dest else 0.0)
        )

        telemetry = self._build_mock_telemetry()
        rgb = self._render_mock_frame()
        info = self._build_mock_info(
            arrive_dest=arrive_dest, crash=crash, out_of_road=out_of_road
        )
        return rgb, telemetry.to_json_dict(), float(reward), done, info

    def get_expert_action(self, info: Dict[str, Any]) -> Tuple[float, float]:
        """Compute expert physical action [steering, throttle_brake] via IDMPolicy or kinematic expert."""
        if not self.use_mock and self._raw_env is not None:
            if hasattr(self, "_last_expert_act") and self._last_expert_act is not None:
                return self._last_expert_act
            return (0.0, 0.0)

        desired_offset = 0.0
        desired_speed = 40.0
        if self._scenario_event == "emergency_brake":
            desired_speed = 0.0
        elif self._scenario_event == "change_left":
            desired_offset = self.lane_width * 0.85
            desired_speed = 40.0
        elif self._scenario_event == "change_right":
            desired_offset = -self.lane_width * 0.85
            desired_speed = 40.0
        elif self._scenario_event == "follow_lead":
            desired_speed = 20.0
        elif self._scenario_event == "sharp_curve":
            desired_speed = 20.0
        elif self._scenario_event == "clear_highway":
            desired_speed = 65.0

        lat_err = desired_offset - self._lane_offset_m
        steering = np.clip(
            0.45 * lat_err - 1.15 * self._heading_error + 0.08 * self._road_curvature,
            -1.0,
            1.0,
        )
        if self._scenario_event == "emergency_brake":
            throttle_brake = -1.0
        else:
            speed_err = desired_speed - self._speed_kmh
            throttle_brake = np.clip(0.08 * speed_err, -1.0, 1.0)
        return float(steering), float(throttle_brake)

    def _sample_mock_traffic_event(self) -> None:
        """Sample driving scenarios with distinct visual & telemetry signatures."""
        u = float(self._rng.random())
        left_avail = self._current_lane_idx > 0
        right_avail = self._current_lane_idx < (self._num_lanes - 1)

        if u < 0.10:
            self._scenario_event = "emergency_brake"
            self._lead_distance_m = float(self._rng.uniform(10.0, 13.0))
            self._lead_speed_kmh = 0.0
            self._road_curvature = 0.0
            self._nav_command = "go_straight"
        elif u < 0.24 and left_avail:
            self._scenario_event = "change_left"
            self._lead_distance_m = float(self._rng.uniform(20.0, 26.0))
            self._lead_speed_kmh = 25.0
            self._road_curvature = 0.0
            self._nav_command = "turn_left"
        elif u < 0.38 and right_avail:
            self._scenario_event = "change_right"
            self._lead_distance_m = float(self._rng.uniform(20.0, 26.0))
            self._lead_speed_kmh = 25.0
            self._road_curvature = 0.0
            self._nav_command = "turn_right"
        elif u < 0.56:
            self._scenario_event = "follow_lead"
            self._lead_distance_m = float(self._rng.uniform(19.0, 27.0))
            self._lead_speed_kmh = float(self._rng.uniform(25.0, 32.0))
            self._road_curvature = 0.0
            self._nav_command = "merge"
        elif u < 0.72:
            self._scenario_event = "sharp_curve"
            self._lead_distance_m = float(self._rng.uniform(45.0, 70.0))
            self._lead_speed_kmh = 45.0
            self._road_curvature = float(
                self._rng.choice([-1.0, 1.0]) * self._rng.uniform(0.50, 0.75)
            )
            self._nav_command = "go_straight"
        else:
            self._scenario_event = "clear_highway"
            self._lead_distance_m = float(self._rng.uniform(58.0, 85.0))
            self._lead_speed_kmh = 70.0
            self._road_curvature = 0.0
            self._nav_command = "go_straight"

    def _build_mock_telemetry(self) -> TelemetryState:
        return TelemetryState(
            speed_kmh=self._speed_kmh,
            lane_offset_m=self._lane_offset_m,
            heading_error=self._heading_error,
            left_lane_available=bool(self._current_lane_idx > 0),
            right_lane_available=bool(self._current_lane_idx < self._num_lanes - 1),
            nav_command=self._nav_command,
            lead_distance_m=self._lead_distance_m,
            lead_speed_kmh=self._lead_speed_kmh,
            road_curvature=self._road_curvature,
            wet_road=self._is_wet,
            a_max_mps2=4.5 if self._is_wet else 6.0,
            uncertainty_margin_m=1.8 if self._is_wet else 0.0,
        )

    def _build_mock_info(
        self, arrive_dest: bool, crash: bool, out_of_road: bool
    ) -> Dict[str, Any]:
        radar_targets: List[Dict[str, Any]] = []
        if self._lead_distance_m < 58.0:
            vx_rel = (self._lead_speed_kmh - self._speed_kmh) / 3.6
            radar_targets.append(
                {
                    "x_m": round(float(self._lead_distance_m), 2),
                    "y_m": 0.0,
                    "vx_rel_mps": round(float(vx_rel), 2),
                    "vy_rel_mps": 0.0,
                    "speed_kmh": round(float(self._lead_speed_kmh), 2),
                    "category": "vehicle",
                    "source": "mock_radar",
                }
            )
        return {
            "step": self._step_count,
            "seed": self._current_seed,
            "simulator": "metadrive",
            "scenario_event": self._scenario_event,
            "lead_distance_m": round(self._lead_distance_m, 2),
            "lead_speed_kmh": round(self._lead_speed_kmh, 2),
            "road_curvature": round(self._road_curvature, 3),
            "radar_targets": radar_targets,
            "imu_telemetry": dict(self._last_imu_telemetry),
            "lane_invasion": False,
            "wet_road": self._is_wet,
            "weather": self.weather,
            "route_completion": float(
                np.clip(self._progress / max(1.0, self._route_length_m), 0.0, 1.0)
            ),
            "nav_state": {
                "map_name": f"MetaDrive-Mock-S{self._current_seed}",
                "ego_xy_yaw": [round(float(self._progress), 1), round(float(self._lane_offset_m), 2), 0.0],
                "dest_xy": [round(float(self._route_length_m), 1), 0.0],
                "active_route_xy": [[round(float(s), 1), 0.0] for s in np.linspace(self._progress, self._route_length_m, 12)],
                "spawn_nodes": self.list_spawn_points(),
                "remaining_route_m": round(max(0.0, self._route_length_m - self._progress), 1),
                "reroute_count": int(self._reroute_count),
            },
            "arrive_dest": bool(arrive_dest),
            "crash": bool(crash),
            "out_of_road": bool(out_of_road),
        }

    def _render_mock_frame(self) -> np.ndarray:
        """Render a structured 196x392x3 RGB driving scene encoding road geometry & traffic."""
        h, w = self.camera_height, self.camera_width
        img = np.zeros((h, w, 3), dtype=np.uint8)

        horizon_y = int(h * 0.22)
        img[:horizon_y, :] = (135, 175, 215)
        img[horizon_y:, :] = (62, 66, 70)

        vp_x = int(w * 0.5 + self._road_curvature * 85.0 - self._heading_error * 70.0)
        ego_center_x = int(w * 0.5 - (self._lane_offset_m / self.lane_width) * 80.0)

        left_edge_bottom = ego_center_x - 95
        right_edge_bottom = ego_center_x + 95
        cv2.line(img, (vp_x - 18, horizon_y), (left_edge_bottom, h - 1), (235, 235, 235), 3)
        cv2.line(img, (vp_x + 18, horizon_y), (right_edge_bottom, h - 1), (235, 235, 235), 3)

        if self._current_lane_idx > 0:
            cv2.line(
                img,
                (vp_x - 48, horizon_y),
                (left_edge_bottom - 140, h - 1),
                (210, 190, 80),
                2,
            )
        if self._current_lane_idx < self._num_lanes - 1:
            cv2.line(
                img,
                (vp_x + 48, horizon_y),
                (right_edge_bottom + 140, h - 1),
                (210, 190, 80),
                2,
            )

        # Highlight target lane corridor or sharp curve warning for clear spatial visual cue
        if self._scenario_event == "change_left" and self._current_lane_idx > 0:
            cv2.rectangle(img, (20, horizon_y + 20), (w // 3, h - 10), (70, 180, 95), -1)
        elif (
            self._scenario_event == "change_right"
            and self._current_lane_idx < self._num_lanes - 1
        ):
            cv2.rectangle(
                img, (2 * w // 3, horizon_y + 20), (w - 20, h - 10), (70, 180, 95), -1
            )
        elif abs(self._road_curvature) >= 0.40:
            cv2.rectangle(
                img, (w // 4, horizon_y + 4), (3 * w // 4, horizon_y + 42), (240, 190, 30), -1
            )

        dist = max(4.0, self._lead_distance_m)
        if dist < 52.0:
            scale = np.clip(28.0 / dist, 0.20, 1.50)
            car_w = int(72 * scale)
            car_h = int(48 * scale)
            car_y = int(
                horizon_y
                + (h - horizon_y - 14) * np.clip(1.0 - dist / 55.0, 0.10, 0.94)
            )
            car_x = int(
                vp_x
                + (ego_center_x - vp_x) * ((car_y - horizon_y) / max(1, h - horizon_y))
            )

            x1 = max(0, car_x - car_w // 2)
            x2 = min(w - 1, car_x + car_w // 2)
            y1 = max(horizon_y, car_y - car_h)
            y2 = min(h - 1, car_y)

            body_color = (
                (245, 35, 35)
                if self._scenario_event == "emergency_brake"
                else (45, 110, 210)
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), body_color, -1)

        return img

    def _extract_real_image(self, obs: Any) -> np.ndarray:
        """Extract RGB frame from real MetaDrive observation."""
        if isinstance(obs, dict) and "image" in obs:
            raw_img = obs["image"]
        elif self._raw_env is not None and "rgb_camera" in getattr(
            self._raw_env.engine, "sensors", {}
        ):
            raw_img = self._raw_env.engine.sensors["rgb_camera"].perceive(
                to_float=False
            )
        else:
            raw_img = np.asarray(obs)
        return self.preprocess_frame(raw_img)

    @staticmethod
    def _find_upcoming_junction_turn_rad(
        vehicle: Any, nav: Any, road_network: Any, current_lanes: Any
    ) -> Tuple[float, float]:
        """Inspect upcoming road-network graph topology to detect a multi-branch junction turn.

        Distinguishes true multi-branch intersections (`len(graph[u_node]) >= 2`, such as X/T
        junctions) from ordinary continuous curve blocks (`len(graph[u_node]) == 1`) using
        HD-map connectivity only, without inspecting any NPC routes. Returns `(turn_rad, dist_m)`.
        """
        if nav is None or road_network is None or not current_lanes or vehicle is None:
            return (0.0, 999.0)
        cps = list(getattr(nav, "checkpoints", []) or [])
        if len(cps) < 2:
            return (0.0, 999.0)
        cur_idx_tuple = getattr(current_lanes[0], "index", None)
        if not cur_idx_tuple:
            return (0.0, 999.0)
        cur_u = cur_idx_tuple[0]
        try:
            idx = cps.index(cur_u)
        except ValueError:
            return (0.0, 999.0)

        try:
            cur_long = float(current_lanes[0].local_coordinates(vehicle.position)[0])
        except Exception:
            cur_long = 0.0
        cur_len = float(getattr(current_lanes[0], "length", 0.0))
        rem_dist = max(0.0, cur_len - cur_long)
        graph = getattr(road_network, "graph", {}) or {}

        for hop in range(idx + 1, min(len(cps) - 1, idx + 3)):
            u_node = cps[hop]
            v_node = cps[hop + 1]
            out_dict = graph.get(u_node, {})
            lanes_uv = out_dict.get(v_node, []) if isinstance(out_dict, dict) else []
            if not lanes_uv:
                break
            branch = lanes_uv[0]
            blen = float(getattr(branch, "length", 0.0))
            if isinstance(out_dict, dict) and len(out_dict) >= 2 and blen > 1.0:
                try:
                    h0 = float(branch.heading_theta_at(0.0))
                    h1 = float(branch.heading_theta_at(blen))
                    dh = float(math.atan2(math.sin(h1 - h0), math.cos(h1 - h0)))
                    return (dh, float(rem_dist))
                except Exception:
                    return (0.0, 999.0)
            rem_dist += blen
        return (0.0, 999.0)

    @staticmethod
    def _check_spatiotemporal_conflict(
        vehicle: Any,
        speed_kmh: float,
        ego_path_pts: List[Tuple[float, float, float]],
        obj: Any,
        obj_pos: np.ndarray,
        ovx: float,
        ovy: float,
        obj_h: float,
        road_network: Any = None,
    ) -> Tuple[bool, float]:
        """Predict curvilinear spatio-temporal tube conflict (PET/CPA) along Ego's planned path.

        Uses only observable kinematics and local HD-map lane geometry so concentric/diverging
        intersection turns are not flagged as false head-on conflicts while true crossing paths
        yield an accurate stop-line distance.
        """
        if not ego_path_pts:
            return (False, 60.0)

        ospd = float(math.hypot(ovx, ovy))
        ego_w = float(getattr(vehicle, "WIDTH", 1.85))
        obj_w = float(getattr(obj, "WIDTH", 1.95))
        r_conf = 0.5 * ego_w + 0.5 * obj_w + 0.50

        use_lane = False
        ol = 0.0
        olat = 0.0
        llen = 0.0
        obj_lane = getattr(obj, "lane", None)
        if obj_lane is not None and ospd > 0.25:
            try:
                ol_raw, olat_raw = obj_lane.local_coordinates(obj_pos)
                ol = float(ol_raw)
                olat = float(olat_raw)
                llen = float(getattr(obj_lane, "length", 0.0))
                if llen > 1.0 and abs(olat) < 2.2:
                    lh = float(obj_lane.heading_theta_at(min(max(0.0, ol), llen)))
                    if math.cos(obj_h - lh) > 0.68:
                        use_lane = True
            except Exception:
                use_lane = False

        next_obj_lane: Any = None
        if use_lane and obj_lane is not None and road_network is not None:
            try:
                o_idx = getattr(obj_lane, "index", None)
                if o_idx and len(o_idx) >= 2:
                    out_d = getattr(road_network, "graph", {}).get(o_idx[1], {})
                    if isinstance(out_d, dict) and len(out_d) >= 1:
                        all_cands: List[Any] = []
                        for sub_l in out_d.values():
                            if isinstance(sub_l, list):
                                all_cands.extend(sub_l)
                        if all_cands:
                            end_p = obj_lane.position(llen, 0.0)
                            # Match outgoing branch closest to lane end AND aligned with NPC heading
                            next_obj_lane = min(
                                all_cands,
                                key=lambda nl: math.hypot(
                                    float(nl.position(0.0, 0.0)[0]) - float(end_p[0]),
                                    float(nl.position(0.0, 0.0)[1]) - float(end_p[1]),
                                )
                                + 2.5 * (1.0 - math.cos(obj_h - float(nl.heading_theta_at(0.0)))),
                            )
            except Exception:
                next_obj_lane = None

        steps = 20 if ospd > 0.25 else 1
        traj: List[Tuple[float, float, float]] = []
        ox0 = float(obj_pos[0])
        oy0 = float(obj_pos[1])
        if use_lane and obj_lane is not None:
            try:
                end_pt = obj_lane.position(llen, 0.0)
                end_h = float(obj_lane.heading_theta_at(llen))
                end_cos = math.cos(end_h)
                end_sin = math.sin(end_h)
                nl_len = (
                    float(getattr(next_obj_lane, "length", 0.0))
                    if next_obj_lane is not None
                    else 0.0
                )
                for k in range(steps):
                    t = float(k) * 0.25
                    dist = ospd * t
                    s_pos = ol + dist
                    if s_pos <= llen:
                        pt = obj_lane.position(s_pos, olat * math.exp(-dist / 12.0))
                        traj.append((t, float(pt[0]), float(pt[1])))
                    else:
                        extra = s_pos - llen
                        if next_obj_lane is not None and nl_len > 1.0:
                            pt = next_obj_lane.position(min(extra, nl_len), 0.0)
                            traj.append((t, float(pt[0]), float(pt[1])))
                        elif extra <= 24.0:
                            traj.append(
                                (
                                    t,
                                    float(end_pt[0]) + end_cos * extra,
                                    float(end_pt[1]) + end_sin * extra,
                                )
                            )
            except Exception:
                traj = []

        if not traj:
            for k in range(steps):
                t = float(k) * 0.25
                traj.append((t, ox0 + ovx * t, oy0 + ovy * t))

        v_ego_eff = max(float(speed_kmh) / 3.6, 4.0)
        for idx_t, (t_obj, ox, oy) in enumerate(traj):
            for s_ego, ex, ey in ego_path_pts:
                d = float(math.hypot(ex - ox, ey - oy))
                if d < r_conf:
                    if (
                        idx_t == 0
                        and len(traj) > 1
                        and d >= (r_conf - 0.35)
                        and float(math.hypot(ex - traj[1][1], ey - traj[1][2])) > (d + 0.12)
                    ):
                        continue
                    # Reserve a 5.2m intersection stop-line setback when conflict is >=6.5m ahead so
                    # Ego stops cleanly behind the stop line out of opposing left-turners' swept arcs!
                    setback = 5.2 if float(s_ego) >= 6.5 else 2.2
                    if ospd <= 0.85:
                        return (True, float(max(0.8, s_ego - setback)))
                    t_ego = float(s_ego) / v_ego_eff
                    if (t_obj - t_ego) <= 3.2 and (t_ego - t_obj) <= 2.8:
                        return (True, float(max(0.8, s_ego - setback)))

        return (False, 60.0)

    def _extract_real_telemetry(
        self, info: Dict[str, Any]
    ) -> Tuple[TelemetryState, Dict[str, Any]]:
        """Extract TelemetryState and auxiliary metrics from a real MetaDrive vehicle."""
        vehicle = self._raw_env.agent
        speed_kmh = float(
            getattr(vehicle, "speed_km_h", getattr(vehicle, "speed", 0.0) * 3.6)
        )

        # Resolve active routing reference lane (handling road transitions, intersections, and lane drops)
        nav = getattr(vehicle, "navigation", None)
        current_lanes = getattr(nav, "current_ref_lanes", None)
        next_lanes = getattr(nav, "next_ref_lanes", None)
        lane = getattr(vehicle, "lane", None)
        road_network = getattr(getattr(nav, "map", None), "road_network", None)

        routing_lane = lane
        if current_lanes:
            if lane in current_lanes:
                routing_lane = lane
            elif next_lanes and lane in next_lanes:
                routing_lane = lane
            else:
                # Find connected lane in current_ref_lanes or closest reference lane
                connected = None
                for cl in current_lanes:
                    if lane is not None and (
                        lane.is_previous_lane_of(cl)
                        or (
                            road_network is not None
                            and road_network.has_connection(lane.index, cl.index)
                        )
                    ):
                        connected = cl
                        break
                routing_lane = (
                    connected
                    if connected is not None
                    else min(
                        current_lanes,
                        key=lambda l: abs(l.local_coordinates(vehicle.position)[1]),
                    )
                )

            def _arm_quintic_to(tgt_lane: Any) -> None:
                try:
                    s0_q, lat0_q = tgt_lane.local_coordinates(vehicle.position)
                    rem_q = max(16.0, float(getattr(tgt_lane, "length", 24.0)) - float(s0_q) - 2.0)
                    if abs(float(lat0_q)) >= 0.65:
                        self._lc_quintic = {
                            "target_lane": tgt_lane,
                            "s0": float(s0_q),
                            "lat0": float(lat0_q),
                            "L": float(min(24.0, max(16.0, rem_q))),
                        }
                except Exception:
                    pass

            # Honour active SafetyShield-gated lane change command with latched lane-change completion
            gated_maneuver = str(getattr(self, "_last_gated_maneuver", "keep_lane"))
            if getattr(self, "_routing_target_lane", None) is not None:
                if (
                    self._routing_target_lane in current_lanes
                    and getattr(self, "_lane_change_hold_steps", 0) > 0
                ):
                    self._lane_change_hold_steps -= 1
                    routing_lane = self._routing_target_lane
                    try:
                        _, t_lat = routing_lane.local_coordinates(vehicle.position)
                        if abs(float(t_lat)) < 0.45 or (
                            gated_maneuver == "keep_lane"
                            and bool(getattr(self, "_last_shield_blocked", False))
                        ):
                            self._routing_target_lane = None
                            self._lane_change_hold_steps = 0
                    except Exception:
                        pass
                else:
                    self._routing_target_lane = None
                    self._lane_change_hold_steps = 0
            elif (
                routing_lane in current_lanes
                and gated_maneuver in ("change_left", "change_right")
                and abs(float(getattr(self, "_last_road_curvature", 0.0))) < 0.18
                and float(getattr(self, "_last_lead_distance_m", 60.0)) < 35.0
            ):
                lane_len_cur = float(getattr(routing_lane, "length", 0.0))
                try:
                    cur_long_pos = float(routing_lane.local_coordinates(vehicle.position)[0])
                except Exception:
                    cur_long_pos = lane_len_cur
                if lane_len_cur >= 25.0 and cur_long_pos < (lane_len_cur - 15.0):
                    g_idx = current_lanes.index(routing_lane)
                    t_idx = (g_idx - 1) if gated_maneuver == "change_left" else (g_idx + 1)
                    if 0 <= t_idx < len(current_lanes):
                        self._routing_target_lane = current_lanes[t_idx]
                        self._lane_change_hold_steps = 25
                        routing_lane = self._routing_target_lane
                        _arm_quintic_to(routing_lane)

            # Check if approaching end of current road block or lane drop
            merge_wait_creep = False
            try:
                from metadrive.policy.idm_policy import FrontBackObjects

                if routing_lane in current_lanes and next_lanes and len(current_lanes) > len(next_lanes):
                    lane_num_diff = len(current_lanes) - len(next_lanes)
                    if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                        valid_indices = list(range(len(next_lanes)))
                    else:
                        valid_indices = list(
                            range(lane_num_diff, len(current_lanes))
                        )
                    cur_idx = current_lanes.index(routing_lane)
                    if cur_idx not in valid_indices:
                        all_objs_tmp = vehicle.lidar.get_surrounding_objects(vehicle)
                        surr_tmp = FrontBackObjects.get_find_front_back_objs(
                            all_objs_tmp,
                            routing_lane,
                            vehicle.position,
                            max_distance=60.0,
                            ref_lanes=current_lanes,
                        )
                        if cur_idx > valid_indices[-1]:
                            if (
                                surr_tmp.left_lane_exist()
                                and surr_tmp.left_back_min_distance() >= 15.0
                                and surr_tmp.left_front_min_distance() >= 5.0
                            ):
                                routing_lane = current_lanes[cur_idx - 1]
                                _arm_quintic_to(routing_lane)
                            else:
                                merge_wait_creep = True
                        else:
                            if (
                                surr_tmp.right_lane_exist()
                                and surr_tmp.right_back_min_distance() >= 15.0
                                and surr_tmp.right_front_min_distance() >= 5.0
                            ):
                                routing_lane = current_lanes[cur_idx + 1]
                                _arm_quintic_to(routing_lane)
                            else:
                                merge_wait_creep = True
            except Exception:
                pass

            # Pre-intersection tactical turn-lane alignment using HD-map junction topology
            # and a 5th-order Minimum-Jerk Quintic S-Curve (yields to active hazards first).
            if self._lc_quintic is not None:
                if self._lc_quintic["target_lane"] not in current_lanes:
                    self._lc_quintic = None
                elif (
                    float(getattr(self, "_last_lead_distance_m", 60.0)) <= 20.0
                    or float(getattr(self, "_last_ped_distance_m", 100.0)) <= 25.0
                ):
                    try:
                        s_chk = float(
                            self._lc_quintic["target_lane"].local_coordinates(vehicle.position)[0]
                        )
                        tau_chk = (s_chk - float(self._lc_quintic["s0"])) / max(
                            1.0, float(self._lc_quintic["L"])
                        )
                        if tau_chk < 0.25:
                            self._lc_quintic = None
                    except Exception:
                        self._lc_quintic = None

            if (
                self._lc_quintic is None
                and len(current_lanes) >= 2
                and self._routing_target_lane is None
                and not merge_wait_creep
                and abs(float(getattr(self, "_last_road_curvature", 0.0))) < 0.16
                and float(getattr(self, "_last_lead_distance_m", 60.0)) > 28.0
                and float(getattr(self, "_last_ped_distance_m", 100.0)) > 35.0
                and routing_lane in current_lanes
            ):
                dh_turn, dist_junc = self._find_upcoming_junction_turn_rad(
                    vehicle, nav, road_network, current_lanes
                )
                if 22.0 <= dist_junc <= 75.0:
                    try:
                        cur_idx = current_lanes.index(routing_lane)
                        cur_long_c, cur_lat_c = routing_lane.local_coordinates(vehicle.position)
                        rem_c = float(getattr(routing_lane, "length", 0.0)) - float(cur_long_c)
                        desired_dir = 0
                        if dh_turn < -0.50 and cur_idx == 0:
                            desired_dir = 1
                        elif dh_turn > 0.50 and cur_idx == (len(current_lanes) - 1):
                            desired_dir = -1
                        if desired_dir != 0 and abs(float(cur_lat_c)) < 0.60 and rem_c >= 22.0:
                            from metadrive.policy.idm_policy import FrontBackObjects

                            all_objs_lc = vehicle.lidar.get_surrounding_objects(vehicle)
                            surr_lc = FrontBackObjects.get_find_front_back_objs(
                                all_objs_lc,
                                routing_lane,
                                vehicle.position,
                                max_distance=60.0,
                                ref_lanes=current_lanes,
                            )
                            gap_ok = (
                                (
                                    surr_lc.right_lane_exist()
                                    and surr_lc.right_back_min_distance() >= 16.0
                                    and surr_lc.right_front_min_distance() >= 12.0
                                )
                                if desired_dir > 0
                                else (
                                    surr_lc.left_lane_exist()
                                    and surr_lc.left_back_min_distance() >= 16.0
                                    and surr_lc.left_front_min_distance() >= 12.0
                                )
                            )
                            if gap_ok:
                                tgt_lane = current_lanes[cur_idx + desired_dir]
                                s0, lat0 = tgt_lane.local_coordinates(vehicle.position)
                                self._lc_quintic = {
                                    "target_lane": tgt_lane,
                                    "s0": float(s0),
                                    "lat0": float(lat0),
                                    "L": float(min(24.0, max(18.0, rem_c - 4.0))),
                                }
                    except Exception:
                        pass

            if self._lc_quintic is not None:
                routing_lane = self._lc_quintic["target_lane"]

        ego_next_lane: Any = None
        if routing_lane is not None and next_lanes and (not current_lanes or routing_lane in current_lanes):
            try:
                r_len_end = float(getattr(routing_lane, "length", 0.0))
                r_end_pt = routing_lane.position(r_len_end, 0.0)
                ego_next_lane = min(
                    next_lanes,
                    key=lambda nl: math.hypot(
                        float(nl.position(0.0, 0.0)[0]) - float(r_end_pt[0]),
                        float(nl.position(0.0, 0.0)[1]) - float(r_end_pt[1]),
                    ),
                )
            except Exception:
                ego_next_lane = next_lanes[0]

        lane_offset_m = 0.0
        heading_error = 0.0
        road_curvature = 0.0
        if routing_lane is not None:
            try:
                long_coord, lat_coord = routing_lane.local_coordinates(vehicle.position)
                # Guard against unblended lateral reference jumps (>1.4m) across block/routing transitions
                prev_rl = getattr(self, "_prev_routing_lane", None)
                if (
                    self._lc_quintic is None
                    and prev_rl is not None
                    and routing_lane is not prev_rl
                    and 1.40 <= abs(float(lat_coord)) <= 4.40
                ):
                    rem_blk = max(16.0, float(getattr(routing_lane, "length", 24.0)) - float(long_coord) - 2.0)
                    self._lc_quintic = {
                        "target_lane": routing_lane,
                        "s0": float(long_coord),
                        "lat0": float(lat_coord),
                        "L": float(min(24.0, max(16.0, rem_blk))),
                    }
                self._prev_routing_lane = routing_lane
                # In MetaDrive, lat_coord > 0 is right of center; negate so +offset = left of center
                lane_offset_m = -float(lat_coord)
                lane_len = float(getattr(routing_lane, "length", long_coord + 12.0))

                def _heading_at_lookahead(d_lh: float) -> float:
                    s_q = float(long_coord + d_lh)
                    if s_q <= 0.0:
                        return float(routing_lane.heading_theta_at(0.0))
                    if s_q <= lane_len:
                        return float(routing_lane.heading_theta_at(s_q))
                    if ego_next_lane is not None:
                        nl_len = float(getattr(ego_next_lane, "length", 20.0))
                        return float(
                            ego_next_lane.heading_theta_at(
                                min(max(0.2, s_q - lane_len), nl_len)
                            )
                        )
                    return float(routing_lane.heading_theta_at(lane_len))

                # Velocity-scaled cross-boundary heading preview compensates actuator servo lag at speed
                lh_dist = min(3.4, max(1.0, (speed_kmh / 3.6) * 0.16))
                lane_heading = _heading_at_lookahead(lh_dist)
                heading_error = float(
                    math.atan2(
                        math.sin(vehicle.heading_theta - lane_heading),
                        math.cos(vehicle.heading_theta - lane_heading),
                    )
                )
                lc_q = self._lc_quintic
                if lc_q is not None and routing_lane is lc_q["target_lane"]:
                    ds_q = float(long_coord) - float(lc_q["s0"])
                    l_q = max(1.0, float(lc_q["L"]))
                    tau_q = max(0.0, min(1.0, ds_q / l_q))
                    q_val = (10.0 * (tau_q ** 3)) - (15.0 * (tau_q ** 4)) + (6.0 * (tau_q ** 5))
                    q_prime = (30.0 * (tau_q ** 2)) - (60.0 * (tau_q ** 3)) + (30.0 * (tau_q ** 4))
                    lat0_q = float(lc_q["lat0"])
                    lat_ref = lat0_q * (1.0 - q_val)
                    dlat_ds = -lat0_q * q_prime / l_q
                    lane_offset_m = -(float(lat_coord) - lat_ref)
                    ref_heading = float(lane_heading) - math.atan(dlat_ds)
                    heading_error = float(
                        math.atan2(
                            math.sin(vehicle.heading_theta - ref_heading),
                            math.cos(vehicle.heading_theta - ref_heading),
                        )
                    )
                    if tau_q >= 0.98 and abs(float(lat_coord)) < 0.45:
                        self._lc_quintic = None

                # Multi-horizon curvature envelope (under-axle local + 12m + 24m + 36m preview)
                s_lo = max(0.0, float(long_coord) - 3.0)
                s_hi = min(lane_len, float(long_coord) + 5.0)
                curv_loc = 0.0
                if (s_hi - s_lo) >= 2.0:
                    dh_loc = math.atan2(
                        math.sin(
                            float(routing_lane.heading_theta_at(s_hi))
                            - float(routing_lane.heading_theta_at(s_lo))
                        ),
                        math.cos(
                            float(routing_lane.heading_theta_at(s_hi))
                            - float(routing_lane.heading_theta_at(s_lo))
                        ),
                    )
                    curv_loc = float(dh_loc * (12.0 / (s_hi - s_lo)))

                h_12 = _heading_at_lookahead(12.0)
                h_24 = _heading_at_lookahead(24.0)
                h_36 = _heading_at_lookahead(36.0)
                curv_12 = float(
                    math.atan2(
                        math.sin(h_12 - lane_heading),
                        math.cos(h_12 - lane_heading),
                    )
                )
                curv_24 = 0.78 * float(
                    math.atan2(
                        math.sin(h_24 - lane_heading),
                        math.cos(h_24 - lane_heading),
                    )
                )
                curv_36 = 0.56 * float(
                    math.atan2(
                        math.sin(h_36 - lane_heading),
                        math.cos(h_36 - lane_heading),
                    )
                )
                raw_curv = max((curv_loc, curv_12, curv_24, curv_36), key=abs)
                if abs(raw_curv) >= abs(self._ema_road_curvature):
                    self._ema_road_curvature = float(raw_curv)
                else:
                    # Smooth curve-exit settling decay prevents premature full-throttle surge
                    # and discontinuous shy-away offset snaps while steering servo unwinds
                    self._ema_road_curvature = float(
                        0.88 * self._ema_road_curvature + 0.12 * raw_curv
                    )
                road_curvature = float(self._ema_road_curvature)
            except Exception:
                pass
        self._last_road_curvature = float(road_curvature)

        # Extract surrounding vehicle distances, 360-deg radar targets & BSM via MetaDrive Lidar / SideDetector
        lead_distance_m = 60.0
        lead_speed_kmh = max(40.0, speed_kmh)
        pedestrian_distance_m = 100.0
        hazard_source = "none"
        blind_spot_left = False
        blind_spot_right = False
        left_avail = False
        right_avail = False
        nav_command = "go_straight"
        radar_targets: List[Dict[str, Any]] = []

        # Compute real IMU kinematics from Bullet vehicle velocity & heading deltas
        dt_step = max(1e-3, self.step_dt)
        ego_h_now = float(getattr(vehicle, "heading_theta", 0.0))
        hx_now, hy_now = math.cos(ego_h_now), math.sin(ego_h_now)
        raw_vel = getattr(vehicle, "velocity", (0.0, 0.0))
        vx_world = float(raw_vel[0]) if len(raw_vel) >= 2 else (speed_kmh / 3.6) * hx_now
        vy_world = float(raw_vel[1]) if len(raw_vel) >= 2 else (speed_kmh / 3.6) * hy_now
        if self._prev_vel_xy is not None and self._prev_heading is not None:
            dvx = (vx_world - self._prev_vel_xy[0]) / dt_step
            dvy = (vy_world - self._prev_vel_xy[1]) / dt_step
            long_acc = dvx * hx_now + dvy * hy_now
            lat_acc = dvx * hy_now - dvy * hx_now
            dyaw = math.atan2(
                math.sin(ego_h_now - self._prev_heading),
                math.cos(ego_h_now - self._prev_heading),
            )
            yaw_rate = dyaw / dt_step
            self._last_imu_telemetry = {
                "long_accel_mps2": round(float(np.clip(long_acc, -15.0, 15.0)), 3),
                "lat_accel_mps2": round(float(np.clip(lat_acc, -15.0, 15.0)), 3),
                "yaw_rate_rps": round(float(np.clip(yaw_rate, -5.0, 5.0)), 4),
                "pitch_rad": 0.0,
            }
        self._prev_vel_xy = (vx_world, vy_world)
        self._prev_heading = ego_h_now

        try:
            from metadrive.policy.idm_policy import FrontBackObjects

            all_objects = vehicle.lidar.get_surrounding_objects(vehicle)
            # Also query MetaDrive 3D engine objects within 65m so rear/side vehicles outside FrontBackObjects are captured
            engine_objs: List[Any] = []
            try:
                engine = getattr(self._raw_env, "engine", None)
                if engine is not None and hasattr(engine, "get_objects"):
                    engine_objs = list(engine.get_objects().values())
            except Exception:
                engine_objs = []

            merged_objs: List[Any] = []
            seen_ids = set()
            for obj in list(all_objects) + engine_objs:
                oid = id(obj)
                if oid not in seen_ids:
                    seen_ids.add(oid)
                    merged_objs.append(obj)

            active_ref_lanes = (
                current_lanes
                if (current_lanes and routing_lane in current_lanes)
                else (
                    next_lanes
                    if (next_lanes and routing_lane in next_lanes)
                    else None
                )
            )
            if active_ref_lanes and routing_lane in active_ref_lanes:
                lane_id = active_ref_lanes.index(routing_lane)
                num_lanes = len(active_ref_lanes)
                left_avail = lane_id > 0
                right_avail = lane_id < (num_lanes - 1)

                surrounding = FrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    routing_lane,
                    vehicle.position,
                    max_distance=60.0,
                    ref_lanes=active_ref_lanes,
                )
                if surrounding.has_front_object():
                    lead_distance_m = float(surrounding.front_min_distance())
                    front_obj = surrounding.front_object()
                    lead_speed_kmh = float(getattr(front_obj, "speed_km_h", 0.0))
                    hazard_source = "vehicle"

                if left_avail:
                    if (
                        surrounding.left_back_min_distance() < 15.0
                        or surrounding.left_front_min_distance() < 15.0
                    ):
                        left_avail = False
                        blind_spot_left = True
                if right_avail:
                    if (
                        surrounding.right_back_min_distance() < 15.0
                        or surrounding.right_front_min_distance() < 15.0
                    ):
                        right_avail = False
                        blind_spot_right = True

            # Curved-lane, 360-degree Radar target & BSM extraction (MetaDrive XY: +y is LEFT, so +y_body_right = rel*h_right)
            ego_pos = np.asarray(vehicle.position, dtype=np.float64)
            ego_h = float(vehicle.heading_theta)
            hx, hy = math.cos(ego_h), math.sin(ego_h)
            ego_long, ego_lat = (
                routing_lane.local_coordinates(vehicle.position)
                if routing_lane is not None
                else (0.0, 0.0)
            )
            ego_path_pts: List[Tuple[float, float, float]] = []
            if routing_lane is not None:
                try:
                    r_len = float(getattr(routing_lane, "length", 0.0))
                    for s_ahead in [1.0] + [float(s_step) for s_step in range(2, 34, 2)]:
                        if float(ego_long) + s_ahead <= r_len:
                            pt = routing_lane.position(
                                float(ego_long) + s_ahead,
                                float(ego_lat) * math.exp(-s_ahead / 8.0),
                            )
                            ego_path_pts.append((s_ahead, float(pt[0]), float(pt[1])))
                        elif ego_next_lane is not None:
                            s_nl = float(ego_long) + s_ahead - r_len
                            nl_len = float(getattr(ego_next_lane, "length", 0.0))
                            if 0.0 <= s_nl <= nl_len:
                                pt = ego_next_lane.position(s_nl, 0.0)
                                ego_path_pts.append((s_ahead, float(pt[0]), float(pt[1])))
                except Exception:
                    ego_path_pts = []

            for obj in merged_objs:
                if obj is vehicle or not hasattr(obj, "position"):
                    continue
                obj_cls_name = type(obj).__name__.lower()
                if "lane" in obj_cls_name or "block" in obj_cls_name or "light" in obj_cls_name:
                    continue
                if not (hasattr(obj, "speed_km_h") or hasattr(obj, "velocity") or "pedestrian" in obj_cls_name or "obstacle" in obj_cls_name):
                    continue

                rel = np.asarray(obj.position, dtype=np.float64) - ego_pos
                euclid_dist = float(math.hypot(rel[0], rel[1]))
                if euclid_dist > 65.0 or euclid_dist < 0.3:
                    continue

                # Standard ego body frame: +x forward, +y right
                x_body = float(rel[0] * hx + rel[1] * hy)
                y_body = float(rel[0] * hy - rel[1] * hx)

                obj_vel = getattr(obj, "velocity", None)
                if obj_vel is not None and len(obj_vel) >= 2:
                    ovx, ovy = float(obj_vel[0]), float(obj_vel[1])
                else:
                    ospd = float(getattr(obj, "speed_km_h", 0.0)) / 3.6
                    oh = float(getattr(obj, "heading_theta", ego_h))
                    ovx, ovy = ospd * math.cos(oh), ospd * math.sin(oh)

                dvx_w = ovx - vx_world
                dvy_w = ovy - vy_world
                vx_rel = float(dvx_w * hx + dvy_w * hy)
                vy_rel = float(dvx_w * hy - dvy_w * hx)
                obj_speed_kmh = float(getattr(obj, "speed_km_h", 3.6 * math.hypot(ovx, ovy)))

                if "pedestrian" in obj_cls_name or "walker" in obj_cls_name:
                    cat = "pedestrian"
                elif "cone" in obj_cls_name or "barrier" in obj_cls_name or "obstacle" in obj_cls_name:
                    cat = "obstacle"
                else:
                    cat = "vehicle"

                obj_h = float(getattr(obj, "heading_theta", ego_h))
                heading_cos = float(math.cos(obj_h - ego_h))
                crossing_conflict = False
                conf_dist = x_body
                if cat == "vehicle" and heading_cos < 0.55:
                    if ego_path_pts:
                        crossing_conflict, conf_dist = self._check_spatiotemporal_conflict(
                            vehicle=vehicle,
                            speed_kmh=speed_kmh,
                            ego_path_pts=ego_path_pts,
                            obj=obj,
                            obj_pos=np.asarray(obj.position, dtype=np.float64),
                            ovx=ovx,
                            ovy=ovy,
                            obj_h=obj_h,
                            road_network=road_network,
                        )
                    if not crossing_conflict and 0.5 < x_body < 26.0 and vx_rel < -0.8:
                        t_cpa = x_body / max(1.0, -vx_rel)
                        y_cpa = y_body + vy_rel * min(t_cpa, 3.0)
                        if t_cpa <= 3.0 and (
                            abs(y_body) < 1.95
                            or (heading_cos > -0.65 and abs(y_cpa) < 2.25 and y_body * vy_rel < -0.35)
                        ):
                            crossing_conflict = True
                            conf_dist = max(0.8, x_body - 3.2)
                    if not crossing_conflict and heading_cos < -0.45:
                        # Reject non-intersecting oncoming opposite-lane vehicles
                        continue

                non_conflicting_cross = bool(
                    cat == "vehicle" and heading_cos < 0.55 and not crossing_conflict
                )

                d_long: Optional[float] = None
                d_lat: Optional[float] = None
                if (
                    not crossing_conflict
                    and not non_conflicting_cross
                    and routing_lane is not None
                ):
                    try:
                        r_len_f = float(getattr(routing_lane, "length", 0.0))
                        o_long, o_lat = routing_lane.local_coordinates(obj.position)
                        if float(o_long) <= (r_len_f + 1.0):
                            if abs(float(o_lat)) < 12.0:
                                cand_long = float(o_long) - float(ego_long)
                                if -25.0 <= cand_long <= 65.0:
                                    d_long = cand_long
                                    d_lat = float(o_lat) - float(ego_lat)
                        elif ego_next_lane is not None:
                            nl_long, nl_lat = ego_next_lane.local_coordinates(obj.position)
                            nl_len_f = float(getattr(ego_next_lane, "length", 0.0))
                            if (
                                -2.0 <= float(nl_long) <= (nl_len_f + 2.0)
                                and abs(float(nl_lat)) < 12.0
                            ):
                                cand_long = (r_len_f - float(ego_long)) + float(nl_long)
                                if -25.0 <= cand_long <= 65.0:
                                    d_long = cand_long
                                    d_lat = float(nl_lat)
                    except Exception:
                        d_long = None
                        d_lat = None

                rad_entry: Dict[str, Any] = {
                    "x_m": round(conf_dist if crossing_conflict else x_body, 2),
                    "y_m": round(y_body, 2),
                    "vx_rel_mps": round(vx_rel, 2),
                    "vy_rel_mps": round(vy_rel, 2),
                    "speed_kmh": round(obj_speed_kmh, 2),
                    "category": cat,
                    "source": "metadrive_3d",
                }
                if crossing_conflict:
                    rad_entry["crossing_conflict"] = True
                elif non_conflicting_cross:
                    rad_entry["non_conflicting_cross"] = True
                elif d_long is not None and d_lat is not None:
                    rad_entry["frenet_x_m"] = round(d_long, 2)
                    rad_entry["frenet_y_m"] = round(d_lat, 2)
                radar_targets.append(rad_entry)

                # Side/rear blind-spot check (+y_body / +d_lat is right, negative is left)
                if not crossing_conflict:
                    eff_x = d_long if d_long is not None else x_body
                    eff_y = d_lat if d_lat is not None else y_body
                    if (-10.5 <= eff_x <= 7.5 and 1.6 <= abs(eff_y) <= 5.0) or (
                        -18.0 <= eff_x < -10.5 and 1.6 <= abs(eff_y) <= 5.0 and vx_rel > 2.5
                    ):
                        if eff_y < 0.0:
                            blind_spot_left = True
                        else:
                            blind_spot_right = True

                forward_proj = x_body
                lateral_proj = abs(y_body)
                if cat == "pedestrian" and 0.5 < forward_proj < pedestrian_distance_m and lateral_proj < 3.0:
                    pedestrian_distance_m = float(forward_proj)

                if crossing_conflict:
                    phys_conf_dist = max(float(euclid_dist), float(conf_dist) + 2.0)
                    if 1.2 < phys_conf_dist < lead_distance_m:
                        lead_distance_m = float(phys_conf_dist)
                        lead_speed_kmh = 0.0
                        hazard_source = "vehicle"
                elif not non_conflicting_cross:
                    in_cone = (
                        d_long is None
                        and (cat != "vehicle" or heading_cos >= 0.55)
                        and 0.5 < forward_proj < lead_distance_m
                        and lateral_proj < 2.6
                    )

                    in_curved_lane = False
                    curve_dist = euclid_dist
                    if d_long is not None and d_lat is not None and 0.5 < d_long < lead_distance_m and abs(d_lat) < 2.6:
                        in_curved_lane = True
                        curve_dist = d_long

                    if in_cone or in_curved_lane:
                        cand_dist = min(
                            forward_proj if in_cone else 60.0,
                            curve_dist if in_curved_lane else 60.0,
                        )
                        if cand_dist < lead_distance_m:
                            lead_distance_m = float(cand_dist)
                            lead_speed_kmh = float(obj_speed_kmh)
                            hazard_source = cat if cat in ("pedestrian", "vehicle") else "vehicle"

            if merge_wait_creep:
                lead_distance_m = min(lead_distance_m, 14.0)
                lead_speed_kmh = min(lead_speed_kmh, 5.0)
                if hazard_source == "none":
                    hazard_source = "vehicle"

            dh_junc_nav, dist_junc_nav = self._find_upcoming_junction_turn_rad(
                vehicle, nav, road_network, current_lanes
            )
            if abs(road_curvature) > 0.22:
                nav_command = "turn_left" if road_curvature > 0 else "turn_right"
            elif dist_junc_nav <= 36.0 and abs(dh_junc_nav) > 0.30:
                nav_command = "turn_left" if dh_junc_nav > 0 else "turn_right"
            elif next_lanes and current_lanes and len(current_lanes) > len(next_lanes):
                nav_command = "merge"
        except Exception:
            pass

        if blind_spot_left or self._lc_quintic is not None:
            left_avail = False
        if blind_spot_right or self._lc_quintic is not None:
            right_avail = False

        expert_maneuver: Optional[str] = None
        self._last_expert_act = (0.0, 0.0)
        try:
            from metadrive.policy.idm_policy import IDMPolicy

            if (
                self._expert_policy is None
                or self._expert_policy.control_object != vehicle
            ):
                self._expert_policy = IDMPolicy(
                    control_object=vehicle,
                    random_seed=self._current_seed,
                )
            act = self._expert_policy.act(agent_id="default_agent")
            self._last_expert_act = (
                float(np.clip(act[0], -1.0, 1.0)),
                float(np.clip(act[1], -1.0, 1.0)),
            )
            target_lane = getattr(self._expert_policy, "routing_target_lane", None)
            ego_lane = getattr(vehicle, "lane", None)
            if (
                target_lane is not None
                and ego_lane is not None
                and target_lane is not ego_lane
                and len(getattr(target_lane, "index", ())) == 3
                and len(getattr(ego_lane, "index", ())) == 3
                and target_lane.index[0] == ego_lane.index[0]
                and target_lane.index[1] == ego_lane.index[1]
            ):
                if target_lane.index[2] < ego_lane.index[2]:
                    expert_maneuver = "change_left"
                elif target_lane.index[2] > ego_lane.index[2]:
                    expert_maneuver = "change_right"
        except Exception:
            pass

        is_wet = bool(
            self._is_wet
            or str(info.get("weather", self.weather)).lower()
            in ("rain", "wet", "heavy_rain", "hardrain", "rainy", "night_rain")
            or float(info.get("precipitation", self.precipitation)) > 0.2
        )
        telemetry = TelemetryState(
            speed_kmh=speed_kmh,
            lane_offset_m=lane_offset_m,
            heading_error=heading_error,
            left_lane_available=left_avail,
            right_lane_available=right_avail,
            nav_command=nav_command,
            lead_distance_m=lead_distance_m,
            lead_speed_kmh=lead_speed_kmh,
            pedestrian_distance_m=pedestrian_distance_m,
            road_curvature=road_curvature,
            expert_maneuver=expert_maneuver,
            hazard_source=hazard_source,
            blind_spot_left=blind_spot_left,
            blind_spot_right=blind_spot_right,
            wet_road=is_wet,
            a_max_mps2=4.5 if is_wet else 6.0,
            uncertainty_margin_m=1.8 if is_wet else 0.0,
        )
        extra_info = {
            "step": self._step_count,
            "seed": self._current_seed,
            "simulator": "metadrive",
            "scenario_event": "real_metadrive",
            "lead_distance_m": round(lead_distance_m, 2),
            "lead_speed_kmh": round(lead_speed_kmh, 2),
            "pedestrian_distance_m": round(pedestrian_distance_m, 2),
            "road_curvature": round(road_curvature, 4),
            "expert_maneuver": expert_maneuver,
            "hazard_source": hazard_source,
            "blind_spot_left": blind_spot_left,
            "blind_spot_right": blind_spot_right,
            "radar_targets": radar_targets,
            "imu_telemetry": dict(self._last_imu_telemetry),
            "lane_invasion": bool(info.get("crash_sidewalk", False)),
            "wet_road": is_wet,
            "weather": self.weather,
            "route_completion": float(info.get("route_completion", 0.0)),
            "nav_state": self._build_real_nav_state(
                vehicle, float(info.get("route_completion", 0.0))
            ),
            "arrive_dest": bool(info.get("arrive_dest", False)),
            "crash": bool(
                info.get("crash", False)
                or info.get("crash_vehicle", False)
                or info.get("crash_object", False)
                or info.get("out_of_road", False)
            ),
            "out_of_road": bool(info.get("out_of_road", False)),
        }
        self._last_lead_distance_m = float(lead_distance_m)
        self._last_ped_distance_m = float(pedestrian_distance_m)
        return telemetry, extra_info

    def close(self) -> None:
        if self._raw_env is not None:
            try:
                self._raw_env.close()
            except Exception:
                pass
            try:
                from metadrive.engine.engine_utils import close_engine, engine_initialized

                if engine_initialized():
                    close_engine()
            except Exception:
                pass
            self._raw_env = None
