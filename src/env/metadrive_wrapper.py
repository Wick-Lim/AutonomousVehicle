"""MetaDrive Simulator Wrapper for 196x392 Wide RGB Vision and Structured Telemetry Extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
    road_curvature: float = 0.0
    expert_maneuver: Optional[str] = None
    traffic_light: str = "green"
    hazard_source: str = "none"
    blind_spot_left: bool = False
    blind_spot_right: bool = False

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
            "road_curvature": round(float(self.road_curvature), 4),
            "expert_maneuver": self.expert_maneuver,
            "traffic_light": str(self.traffic_light),
            "hazard_source": str(self.hazard_source),
            "blind_spot_left": bool(self.blind_spot_left),
            "blind_spot_right": bool(self.blind_spot_right),
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

        self.use_mock: bool = bool(
            use_mock or self.cfg.get("project", {}).get("use_mock", False)
        )
        self._raw_env: Any = None
        self._expert_policy: Any = None
        self._routing_target_lane: Any = None

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

        if not self.use_mock:
            self._init_real_metadrive()

    def _init_real_metadrive(self) -> None:
        """Initialize the real MetaDriveEnv with RGBCamera and procedural traffic."""
        try:
            from metadrive.envs.metadrive_env import MetaDriveEnv
            from metadrive.component.sensors.rgb_camera import RGBCamera
        except ImportError as exc:
            raise ImportError(
                "MetaDrive is not installed. Install `metadrive-simulator` or run with `--mock`."
            ) from exc

        use_render = bool(self.cfg.get("env", {}).get("use_render", False))
        md_config = {
            "use_render": use_render,
            "image_observation": True,
            "stack_size": 1,
            "sensors": {
                "rgb_camera": (RGBCamera, self.camera_width, self.camera_height)
            },
            "vehicle_config": {"image_source": "rgb_camera"},
            "decision_repeat": self.decision_repeat,
            "traffic_density": self.traffic_density,
            "map": self.map_blocks,
            "start_seed": self.start_seed,
            "num_scenarios": self.num_scenarios,
            "horizon": self.max_episode_steps,
            "out_of_road_penalty": 10.0,
            "crash_vehicle_penalty": 10.0,
        }
        self._raw_env = MetaDriveEnv(md_config)

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

        if not self.use_mock and self._raw_env is not None:
            obs, info = self._raw_env.reset(seed=self._current_seed)
            self._step_count = 0
            self._routing_target_lane = None
            self._expert_policy = None
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
        )

    def _build_mock_info(
        self, arrive_dest: bool, crash: bool, out_of_road: bool
    ) -> Dict[str, Any]:
        return {
            "step": self._step_count,
            "seed": self._current_seed,
            "scenario_event": self._scenario_event,
            "lead_distance_m": round(self._lead_distance_m, 2),
            "lead_speed_kmh": round(self._lead_speed_kmh, 2),
            "road_curvature": round(self._road_curvature, 3),
            "route_completion": float(
                np.clip(self._progress / max(1.0, self._route_length_m), 0.0, 1.0)
            ),
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
                            else:
                                merge_wait_creep = True
                        else:
                            if (
                                surr_tmp.right_lane_exist()
                                and surr_tmp.right_back_min_distance() >= 15.0
                                and surr_tmp.right_front_min_distance() >= 5.0
                            ):
                                routing_lane = current_lanes[cur_idx + 1]
                            else:
                                merge_wait_creep = True
            except Exception:
                pass

        lane_offset_m = 0.0
        heading_error = 0.0
        road_curvature = 0.0
        if routing_lane is not None:
            try:
                long_coord, lat_coord = routing_lane.local_coordinates(vehicle.position)
                # In MetaDrive, lat_coord > 0 is right of center; negate so +offset = left of center
                lane_offset_m = -float(lat_coord)
                lookahead_long = min(
                    float(long_coord + 1.0),
                    float(getattr(routing_lane, "length", long_coord + 1.0)),
                )
                lane_heading = routing_lane.heading_theta_at(lookahead_long)
                heading_error = float(
                    math.atan2(
                        math.sin(vehicle.heading_theta - lane_heading),
                        math.cos(vehicle.heading_theta - lane_heading),
                    )
                )
                ahead_long = min(
                    float(long_coord + 12.0),
                    float(getattr(routing_lane, "length", long_coord + 12.0)),
                )
                lane_heading_ahead = routing_lane.heading_theta_at(ahead_long)
                road_curvature = float(
                    math.atan2(
                        math.sin(lane_heading_ahead - lane_heading),
                        math.cos(lane_heading_ahead - lane_heading),
                    )
                )
            except Exception:
                pass

        # Extract surrounding vehicle distances via MetaDrive FrontBackObjects + lidar
        lead_distance_m = 60.0
        lead_speed_kmh = max(40.0, speed_kmh)
        left_avail = False
        right_avail = False
        nav_command = "go_straight"

        try:
            from metadrive.policy.idm_policy import FrontBackObjects

            all_objects = vehicle.lidar.get_surrounding_objects(vehicle)
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

                if left_avail:
                    if (
                        surrounding.left_back_min_distance() < 15.0
                        or surrounding.left_front_min_distance() < 15.0
                    ):
                        left_avail = False
                if right_avail:
                    if (
                        surrounding.right_back_min_distance() < 15.0
                        or surrounding.right_front_min_distance() < 15.0
                    ):
                        right_avail = False

            # Curved-lane & forward-cone radar check (tracks lead vehicles around curves and intersections)
            ego_pos = np.asarray(vehicle.position, dtype=np.float64)
            ego_h = float(vehicle.heading_theta)
            hx, hy = math.cos(ego_h), math.sin(ego_h)
            ego_long, ego_lat = (
                routing_lane.local_coordinates(vehicle.position)
                if routing_lane is not None
                else (0.0, 0.0)
            )
            for obj in all_objects:
                if obj is vehicle or not hasattr(obj, "position"):
                    continue
                obj_h = float(getattr(obj, "heading_theta", ego_h))
                if math.cos(obj_h - ego_h) < -0.65:
                    # Ignore oncoming traffic moving in the opposite lane direction (~180 deg)
                    continue
                rel = np.asarray(obj.position, dtype=np.float64) - ego_pos
                euclid_dist = float(math.hypot(rel[0], rel[1]))
                if euclid_dist > 55.0:
                    continue
                forward_proj = rel[0] * hx + rel[1] * hy
                lateral_proj = abs(-rel[0] * hy + rel[1] * hx)
                in_cone = 0.5 < forward_proj < lead_distance_m and lateral_proj < 2.8

                in_curved_lane = False
                curve_dist = euclid_dist
                if routing_lane is not None:
                    try:
                        o_long, o_lat = routing_lane.local_coordinates(obj.position)
                        d_long = float(o_long - ego_long)
                        if (
                            0.5 < d_long < lead_distance_m
                            and abs(float(o_lat - ego_lat)) < 2.6
                        ):
                            in_curved_lane = True
                            curve_dist = d_long
                    except Exception:
                        pass

                if in_cone or in_curved_lane:
                    cand_dist = min(
                        forward_proj if in_cone else 60.0,
                        curve_dist if in_curved_lane else 60.0,
                    )
                    if cand_dist < lead_distance_m:
                        lead_distance_m = float(cand_dist)
                        lead_speed_kmh = float(getattr(obj, "speed_km_h", 0.0))

            if merge_wait_creep:
                lead_distance_m = min(lead_distance_m, 14.0)
                lead_speed_kmh = min(lead_speed_kmh, 5.0)

            if abs(road_curvature) > 0.22:
                nav_command = "turn_left" if road_curvature > 0 else "turn_right"
            elif next_lanes and current_lanes and len(current_lanes) > len(next_lanes):
                nav_command = "merge"
        except Exception:
            pass

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

        telemetry = TelemetryState(
            speed_kmh=speed_kmh,
            lane_offset_m=lane_offset_m,
            heading_error=heading_error,
            left_lane_available=left_avail,
            right_lane_available=right_avail,
            nav_command=nav_command,
            lead_distance_m=lead_distance_m,
            lead_speed_kmh=lead_speed_kmh,
            road_curvature=road_curvature,
            expert_maneuver=expert_maneuver,
        )
        extra_info = {
            "step": self._step_count,
            "seed": self._current_seed,
            "scenario_event": "real_metadrive",
            "lead_distance_m": round(lead_distance_m, 2),
            "lead_speed_kmh": round(lead_speed_kmh, 2),
            "road_curvature": round(road_curvature, 4),
            "expert_maneuver": expert_maneuver,
            "route_completion": float(info.get("route_completion", 0.0)),
            "arrive_dest": bool(info.get("arrive_dest", False)),
            "crash": bool(
                info.get("crash", False)
                or info.get("crash_vehicle", False)
                or info.get("crash_object", False)
                or info.get("out_of_road", False)
            ),
            "out_of_road": bool(info.get("out_of_road", False)),
        }
        return telemetry, extra_info

    def close(self) -> None:
        if self._raw_env is not None:
            self._raw_env.close()
            self._raw_env = None
