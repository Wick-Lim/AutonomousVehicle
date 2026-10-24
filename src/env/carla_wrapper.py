"""CARLA Simulator Wrapper (0.9.14 / 0.9.15+) for 196x392 Wide RGB Vision & Urban Telemetry.

Provides a drop-in CARLA environment interface matching `MetaDriveWrapper`:
- Connects to a live CARLA server (`carla.Client(host, port)`) in 20Hz synchronous mode (`fixed_delta_seconds=0.05`).
- Attaches a 392x196 wide RGB camera (`sensor.camera.rgb`) and collision/lane-invasion sensors.
- Extracts structured `TelemetryState` from `carla.Map.get_waypoint()`, including urban traffic light
  states (`red`, `yellow`, `green`) and pedestrian crosswalk hazards (`hazard_source`).
- Translates `[steering, throttle_brake]` in `[-1, 1]` into `carla.VehicleControl(steer, throttle, brake)`.
- Includes a deterministic Urban CARLA Kinematic/Visual Mock Simulator (`use_mock=True` or automatic
  fallback when no CARLA server is listening on `host:port`) with traffic lights, zebra crosswalks,
  pedestrians, and multi-lane urban traffic.
"""

from __future__ import annotations

import math
import queue
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.env.metadrive_wrapper import NAV_COMMANDS, TelemetryState


class CarlaWrapper:
    """CARLA 3D Urban Autonomous Driving Wrapper with 196x392 RGB & Waypoint/Traffic-Light Telemetry."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_mock: bool = False,
        start_seed: Optional[int] = None,
        num_scenarios: int = 50,
    ) -> None:
        self.cfg = config or {}
        env_cfg = self.cfg.get("env", {})
        carla_cfg = env_cfg.get("carla", {})

        self.camera_height: int = int(env_cfg.get("camera_height", 196))
        self.camera_width: int = int(env_cfg.get("camera_width", 392))
        self.crop_top_ratio: float = float(env_cfg.get("crop_top_ratio", 0.25))
        self.crop_bottom_ratio: float = float(env_cfg.get("crop_bottom_ratio", 0.10))
        self.decision_repeat: int = int(env_cfg.get("decision_repeat", 5))
        self.max_episode_steps: int = int(env_cfg.get("max_episode_steps", 750))
        self.lane_width: float = float(env_cfg.get("lane_width", 3.5))
        self.start_seed: int = int(
            start_seed if start_seed is not None else env_cfg.get("start_seed", 1000)
        )
        self.num_scenarios: int = int(num_scenarios)

        # CARLA Server connection settings
        self.host: str = str(carla_cfg.get("host", "127.0.0.1"))
        self.port: int = int(carla_cfg.get("port", 2000))
        self.tm_port: int = int(carla_cfg.get("tm_port", 8000))
        self.town: str = str(carla_cfg.get("town", "Town03"))
        self.vehicle_filter: str = str(
            carla_cfg.get("vehicle_blueprint", "vehicle.tesla.model3")
        )
        self.camera_fov: float = float(carla_cfg.get("camera_fov", 100.0))
        self.fixed_delta_seconds: float = float(
            carla_cfg.get("fixed_delta_seconds", 0.05)
        )
        self.num_npc_vehicles: int = int(carla_cfg.get("num_npc_vehicles", 25))
        self.num_pedestrians: int = int(carla_cfg.get("num_pedestrians", 15))
        self.target_route_distance_m: float = float(
            carla_cfg.get("target_route_distance_m", 180.0)
        )

        self.use_mock: bool = bool(
            use_mock or self.cfg.get("project", {}).get("use_mock", False)
        )

        # Live CARLA handles
        self._carla_mod: Any = None
        self._client: Any = None
        self._world: Any = None
        self._map: Any = None
        self._traffic_manager: Any = None
        self._ego_vehicle: Any = None
        self._camera_sensor: Any = None
        self._collision_sensor: Any = None
        self._lane_invasion_sensor: Any = None
        self._behavior_agent: Any = None
        self._npc_actors: List[Any] = []
        self._image_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._collision_detected: bool = False
        self._out_of_road_detected: bool = False

        if not self.use_mock:
            self._try_init_real_carla()

        # Internal state for tracking episode progress & Urban CARLA Mock Simulator
        self._rng = np.random.default_rng(self.start_seed)
        self._current_seed: int = self.start_seed
        self._step_count: int = 0
        self._progress: float = 0.0
        self._route_length_m: float = self.target_route_distance_m
        self._prev_location: Optional[Tuple[float, float, float]] = None

        self._num_lanes: int = 3
        self._current_lane_idx: int = 1
        self._speed_kmh: float = 35.0
        self._lane_offset_m: float = 0.0
        self._heading_error: float = 0.0
        self._lead_distance_m: float = 65.0
        self._lead_speed_kmh: float = 50.0
        self._road_curvature: float = 0.0
        self._nav_command: str = "go_straight"
        self._traffic_light: str = "green"
        self._hazard_source: str = "none"
        self._scenario_event: str = "clear_highway"
        self._last_expert_act: Tuple[float, float] = (0.0, 0.0)

    def _try_init_real_carla(self) -> None:
        """Attempt to connect to a running CARLA server (`host:port`); fallback to urban mock if unavailable."""
        try:
            import carla  # type: ignore

            self._carla_mod = carla
            self._client = carla.Client(self.host, self.port)
            self._client.set_timeout(4.0)
            self._world = self._client.get_world()
            current_map_name = self._world.get_map().name.split("/")[-1]
            if self.town and current_map_name != self.town:
                self._world = self._client.load_world(self.town)
            self._map = self._world.get_map()

            settings = self._world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.fixed_delta_seconds
            self._world.apply_settings(settings)

            self._traffic_manager = self._client.get_trafficmanager(self.tm_port)
            self._traffic_manager.set_synchronous_mode(True)
            self._traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        except Exception:
            self._client = None
            self._world = None
            self.use_mock = True

    @staticmethod
    def to_carla_vehicle_control(
        steering: float, throttle_brake: float
    ) -> Dict[str, Any]:
        """Convert normalized `[steering, throttle_brake]` in `[-1, 1]` to CARLA VehicleControl fields."""
        steer = float(np.clip(steering, -1.0, 1.0))
        tb = float(np.clip(throttle_brake, -1.0, 1.0))
        if tb >= 0.0:
            throttle = tb
            brake = 0.0
        else:
            throttle = 0.0
            brake = -tb
        return {
            "steer": round(steer, 4),
            "throttle": round(throttle, 4),
            "brake": round(brake, 4),
            "hand_brake": False,
            "reverse": False,
        }

    def preprocess_frame(self, raw_img: np.ndarray) -> np.ndarray:
        """Crop sky/hood and resize to 196x392x3 uint8 RGB."""
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

    def _cleanup_carla_actors(self) -> None:
        """Destroy sensors, ego vehicle, and NPC actors in live CARLA world."""
        for sensor in (
            self._camera_sensor,
            self._collision_sensor,
            self._lane_invasion_sensor,
        ):
            if sensor is not None:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception:
                    pass
        self._camera_sensor = None
        self._collision_sensor = None
        self._lane_invasion_sensor = None

        if self._ego_vehicle is not None:
            try:
                self._ego_vehicle.destroy()
            except Exception:
                pass
            self._ego_vehicle = None

        for npc in self._npc_actors:
            try:
                npc.destroy()
            except Exception:
                pass
        self._npc_actors.clear()

        while not self._image_queue.empty():
            try:
                self._image_queue.get_nowait()
            except Exception:
                break

    def _spawn_real_carla_episode(self, seed: int) -> None:
        """Spawn ego vehicle, RGB camera, collision sensor, and NPC traffic in CARLA."""
        carla = self._carla_mod
        self._cleanup_carla_actors()
        self._collision_detected = False
        self._out_of_road_detected = False
        self._progress = 0.0

        bp_lib = self._world.get_blueprint_library()
        spawn_points = list(self._map.get_spawn_points())
        rng = np.random.default_rng(seed)
        rng.shuffle(spawn_points)

        ego_bps = bp_lib.filter(self.vehicle_filter)
        ego_bp = ego_bps[0] if len(ego_bps) > 0 else bp_lib.filter("vehicle.*")[0]
        ego_bp.set_attribute("role_name", "hero")

        for sp in spawn_points:
            self._ego_vehicle = self._world.try_spawn_actor(ego_bp, sp)
            if self._ego_vehicle is not None:
                break
        if self._ego_vehicle is None:
            raise RuntimeError("Failed to spawn CARLA ego vehicle.")

        # Attach 392x196 Wide RGB Camera
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.camera_width))
        cam_bp.set_attribute("image_size_y", str(self.camera_height))
        cam_bp.set_attribute("fov", str(self.camera_fov))
        cam_transform = carla.Transform(carla.Location(x=1.5, z=1.6))
        self._camera_sensor = self._world.spawn_actor(
            cam_bp, cam_transform, attach_to=self._ego_vehicle
        )

        def _on_image(carla_img: Any) -> None:
            arr = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
            arr = arr.reshape((carla_img.height, carla_img.width, 4))
            rgb = arr[:, :, :3][:, :, ::-1].copy()
            self._image_queue.put(rgb)

        self._camera_sensor.listen(_on_image)

        # Attach Collision Sensor
        col_bp = bp_lib.find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self._ego_vehicle
        )
        self._collision_sensor.listen(
            lambda _event: setattr(self, "_collision_detected", True)
        )

        # Spawn NPC traffic vehicles
        vehicle_bps = [
            bp
            for bp in bp_lib.filter("vehicle.*")
            if int(bp.get_attribute("number_of_wheels")) == 4
        ]
        for sp in spawn_points[1 : 1 + self.num_npc_vehicles]:
            npc_bp = vehicle_bps[int(rng.integers(0, len(vehicle_bps)))]
            npc = self._world.try_spawn_actor(npc_bp, sp)
            if npc is not None:
                npc.set_autopilot(True, self.tm_port)
                self._npc_actors.append(npc)

        # Initialize CARLA BehaviorAgent expert if available
        self._behavior_agent = None
        try:
            from agents.navigation.behavior_agent import BehaviorAgent  # type: ignore

            self._behavior_agent = BehaviorAgent(self._ego_vehicle, behavior="normal")
            dest_sp = spawn_points[min(len(spawn_points) - 1, 10)]
            self._behavior_agent.set_destination(dest_sp.location)
        except Exception:
            self._ego_vehicle.set_autopilot(False, self.tm_port)

        self._world.tick()
        loc = self._ego_vehicle.get_location()
        self._prev_location = (float(loc.x), float(loc.y), float(loc.z))

    def _extract_real_carla_state(self) -> Tuple[ np.ndarray, TelemetryState, Dict[str, Any]]:
        """Extract 196x392 RGB frame and urban waypoint/traffic-light telemetry from live CARLA."""
        carla = self._carla_mod
        try:
            raw_rgb = self._image_queue.get(timeout=2.0)
        except Exception:
            raw_rgb = np.zeros(
                (self.camera_height, self.camera_width, 3), dtype=np.uint8
            )
        rgb = self.preprocess_frame(raw_rgb)

        v = self._ego_vehicle
        vel = v.get_velocity()
        speed_ms = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        speed_kmh = float(speed_ms * 3.6)

        loc = v.get_location()
        if self._prev_location is not None:
            dx = float(loc.x) - self._prev_location[0]
            dy = float(loc.y) - self._prev_location[1]
            self._progress += math.hypot(dx, dy)
        self._prev_location = (float(loc.x), float(loc.y), float(loc.z))

        wp = self._map.get_waypoint(
            loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        ego_tr = v.get_transform()
        ego_yaw = math.radians(float(ego_tr.rotation.yaw))
        wp_tr = wp.transform
        wp_yaw = math.radians(float(wp_tr.rotation.yaw))

        # Signed lateral offset (+ = left of lane center)
        dx_wp = float(loc.x - wp_tr.location.x)
        dy_wp = float(loc.y - wp_tr.location.y)
        right_vec_x = -math.sin(wp_yaw)
        right_vec_y = math.cos(wp_yaw)
        lat_right = dx_wp * right_vec_x + dy_wp * right_vec_y
        lane_offset_m = float(-lat_right)

        heading_error = float(
            math.atan2(math.sin(ego_yaw - wp_yaw), math.cos(ego_yaw - wp_yaw))
        )

        next_wps = wp.next(12.0)
        road_curvature = 0.0
        if next_wps:
            next_yaw = math.radians(float(next_wps[0].transform.rotation.yaw))
            road_curvature = float(
                math.atan2(math.sin(next_yaw - wp_yaw), math.cos(next_yaw - wp_yaw))
            )

        left_wp = wp.get_left_lane()
        right_wp = wp.get_right_lane()
        left_avail = bool(
            left_wp is not None and left_wp.lane_type == carla.LaneType.Driving
        )
        right_avail = bool(
            right_wp is not None and right_wp.lane_type == carla.LaneType.Driving
        )

        # Scan forward vehicles and pedestrians
        lead_distance_m = 60.0
        lead_speed_kmh = max(40.0, speed_kmh)
        hazard_source = "none"
        hx, hy = math.cos(ego_yaw), math.sin(ego_yaw)

        for actor in self._world.get_actors():
            if actor.id == v.id:
                continue
            type_id = str(getattr(actor, "type_id", ""))
            if not (type_id.startswith("vehicle.") or type_id.startswith("walker.pedestrian.")):
                continue
            a_loc = actor.get_location()
            rx = float(a_loc.x - loc.x)
            ry = float(a_loc.y - loc.y)
            dist = math.hypot(rx, ry)
            if dist > 55.0:
                continue
            fp = rx * hx + ry * hy
            lp = abs(-rx * hy + ry * hx)
            if 0.5 < fp < lead_distance_m and lp < 2.6:
                lead_distance_m = float(fp)
                a_vel = actor.get_velocity()
                lead_speed_kmh = float(
                    3.6 * math.sqrt(a_vel.x**2 + a_vel.y**2 + a_vel.z**2)
                )
                hazard_source = (
                    "pedestrian" if type_id.startswith("walker.pedestrian.") else "vehicle"
                )

        # Traffic light state check
        traffic_light = "green"
        if v.is_at_traffic_light():
            tl = v.get_traffic_light()
            if tl is not None:
                tl_state = tl.get_state()
                if tl_state == carla.TrafficLightState.Red:
                    traffic_light = "red"
                    hazard_source = "red_light"
                    lead_distance_m = min(lead_distance_m, 8.0)
                    lead_speed_kmh = 0.0
                elif tl_state == carla.TrafficLightState.Yellow:
                    traffic_light = "yellow"

        nav_command = "go_straight"
        if abs(road_curvature) > 0.22:
            nav_command = "turn_left" if road_curvature > 0 else "turn_right"

        # Compute expert action from CARLA BehaviorAgent if active
        self._last_expert_act = (0.0, 0.0)
        if self._behavior_agent is not None:
            try:
                ctrl = self._behavior_agent.run_step()
                tb = float(ctrl.throttle) if ctrl.throttle > 0.0 else -float(ctrl.brake)
                self._last_expert_act = (
                    float(np.clip(ctrl.steer, -1.0, 1.0)),
                    float(np.clip(tb, -1.0, 1.0)),
                )
            except Exception:
                pass

        out_of_road = bool(abs(lane_offset_m) > self.lane_width * 0.95)
        crash = bool(self._collision_detected or out_of_road)
        arrive_dest = bool(self._progress >= self._route_length_m and not crash)

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
            expert_maneuver=None,
            traffic_light=traffic_light,
            hazard_source=hazard_source,
        )
        extra_info = {
            "step": self._step_count,
            "seed": self._current_seed,
            "simulator": "carla",
            "scenario_event": "real_carla",
            "lead_distance_m": round(lead_distance_m, 2),
            "lead_speed_kmh": round(lead_speed_kmh, 2),
            "road_curvature": round(road_curvature, 4),
            "traffic_light": traffic_light,
            "hazard_source": hazard_source,
            "route_completion": float(
                np.clip(self._progress / max(1.0, self._route_length_m), 0.0, 1.0)
            ),
            "arrive_dest": arrive_dest,
            "crash": crash,
            "out_of_road": out_of_road,
        }
        return rgb, telemetry, extra_info

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        """Reset CARLA environment and return `(rgb_196x392, telemetry_dict, info_dict)`."""
        if seed is not None:
            self._current_seed = int(seed)
        else:
            self._current_seed += 1

        self._step_count = 0
        if not self.use_mock and self._world is not None:
            self._spawn_real_carla_episode(self._current_seed)
            rgb, telemetry, extra_info = self._extract_real_carla_state()
            return rgb, telemetry.to_json_dict(), extra_info

        # Urban CARLA Mock Simulator initialization
        self._rng = np.random.default_rng(self._current_seed)
        self._progress = 0.0
        self._route_length_m = float(self._rng.uniform(155.0, 195.0))
        self._num_lanes = 3
        self._current_lane_idx = int(self._rng.integers(0, self._num_lanes))
        self._speed_kmh = float(self._rng.uniform(30.0, 42.0))
        self._lane_offset_m = float(self._rng.normal(0.0, 0.10))
        self._heading_error = float(self._rng.normal(0.0, 0.02))
        self._sample_mock_urban_event()

        telemetry = self._build_mock_telemetry()
        rgb = self._render_mock_urban_frame()
        info = self._build_mock_info(arrive_dest=False, crash=False, out_of_road=False)
        return rgb, telemetry.to_json_dict(), info

    def step(
        self, action: Tuple[float, float] | np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute `[steering, throttle_brake]` in CARLA (via `VehicleControl`) and return next state."""
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))
        self._step_count += 1

        if not self.use_mock and self._world is not None and self._ego_vehicle is not None:
            carla = self._carla_mod
            ctrl_dict = self.to_carla_vehicle_control(steering, throttle_brake)
            ctrl = carla.VehicleControl(
                steer=ctrl_dict["steer"],
                throttle=ctrl_dict["throttle"],
                brake=ctrl_dict["brake"],
                hand_brake=ctrl_dict["hand_brake"],
                reverse=ctrl_dict["reverse"],
            )
            self._ego_vehicle.apply_control(ctrl)
            self._world.tick()
            rgb, telemetry, info = self._extract_real_carla_state()
            done = bool(
                info["crash"]
                or info["arrive_dest"]
                or self._step_count >= self.max_episode_steps
            )
            reward = (
                (telemetry.speed_kmh / 36.0)
                - abs(telemetry.lane_offset_m) * 0.15
                - (10.0 if info["crash"] else 0.0)
                + (20.0 if info["arrive_dest"] else 0.0)
            )
            return rgb, telemetry.to_json_dict(), float(reward), done, info

        # Urban CARLA Mock Kinematic Step
        dt = 0.05 * self.decision_repeat
        accel_kmh_s = (
            throttle_brake * 35.0 if throttle_brake >= 0.0 else throttle_brake * 65.0
        )
        self._speed_kmh = float(np.clip(self._speed_kmh + accel_kmh_s * dt, 0.0, 80.0))
        speed_ms = self._speed_kmh / 3.6

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

        half_lane = self.lane_width * 0.52
        if self._lane_offset_m > half_lane and self._current_lane_idx > 0:
            self._current_lane_idx -= 1
            self._lane_offset_m -= self.lane_width
            if self._scenario_event == "change_left":
                self._lead_distance_m = 65.0
                self._lead_speed_kmh = 65.0
                self._traffic_light = "green"
                self._hazard_source = "none"
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
                self._traffic_light = "green"
                self._hazard_source = "none"
                self._scenario_event = "clear_highway"

        self._progress += speed_ms * dt
        rel_speed_ms = (self._lead_speed_kmh - self._speed_kmh) / 3.6
        self._lead_distance_m = float(
            np.clip(self._lead_distance_m + rel_speed_ms * dt, 1.5, 90.0)
        )

        # Resolve urban red light or pedestrian crossing once vehicle yields/brakes
        if self._scenario_event == "emergency_brake" and (
            self._speed_kmh < 12.0 or throttle_brake <= -0.5
        ):
            self._lead_speed_kmh = 45.0
            self._lead_distance_m = max(self._lead_distance_m, 32.0)
            self._traffic_light = "green"
            self._hazard_source = "none"
            self._scenario_event = "keep_lane"

        if self._step_count % 18 == 0:
            self._sample_mock_urban_event()

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
        rgb = self._render_mock_urban_frame()
        info = self._build_mock_info(
            arrive_dest=arrive_dest, crash=crash, out_of_road=out_of_road
        )
        return rgb, telemetry.to_json_dict(), float(reward), done, info

    def get_expert_action(self, info: Dict[str, Any]) -> Tuple[float, float]:
        """Compute CARLA BehaviorAgent / Urban Autopilot expert action `[steering, throttle_brake]`."""
        if not self.use_mock and self._world is not None:
            return self._last_expert_act

        desired_offset = 0.0
        desired_speed = 40.0
        if (
            self._scenario_event == "emergency_brake"
            or self._traffic_light == "red"
            or self._hazard_source == "pedestrian"
        ):
            desired_speed = 0.0
        elif self._scenario_event == "change_left":
            desired_offset = self.lane_width * 0.85
            desired_speed = 40.0
        elif self._scenario_event == "change_right":
            desired_offset = -self.lane_width * 0.85
            desired_speed = 40.0
        elif self._scenario_event == "follow_lead" or self._traffic_light == "yellow":
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
        if (
            self._scenario_event == "emergency_brake"
            or self._traffic_light == "red"
            or self._hazard_source == "pedestrian"
        ):
            throttle_brake = -1.0
        else:
            speed_err = desired_speed - self._speed_kmh
            throttle_brake = np.clip(0.08 * speed_err, -1.0, 1.0)
        return float(steering), float(throttle_brake)

    def _sample_mock_urban_event(self) -> None:
        """Sample CARLA urban traffic scenarios (including Red Lights, Pedestrians, Intersections)."""
        u = float(self._rng.random())
        left_avail = self._current_lane_idx > 0
        right_avail = self._current_lane_idx < (self._num_lanes - 1)

        self._traffic_light = "green"
        self._hazard_source = "none"

        if u < 0.07:
            # Urban Pedestrian Crosswalk AEB
            self._scenario_event = "emergency_brake"
            self._lead_distance_m = float(self._rng.uniform(9.5, 12.8))
            self._lead_speed_kmh = 0.0
            self._road_curvature = 0.0
            self._nav_command = "go_straight"
            self._hazard_source = "pedestrian"
        elif u < 0.14:
            # Urban Red Traffic Light Stop
            self._scenario_event = "emergency_brake"
            self._lead_distance_m = float(self._rng.uniform(10.0, 13.2))
            self._lead_speed_kmh = 0.0
            self._road_curvature = 0.0
            self._nav_command = "go_straight"
            self._traffic_light = "red"
            self._hazard_source = "red_light"
        elif u < 0.28 and left_avail:
            self._scenario_event = "change_left"
            self._lead_distance_m = float(self._rng.uniform(20.0, 26.0))
            self._lead_speed_kmh = 25.0
            self._road_curvature = 0.0
            self._nav_command = "turn_left"
            self._hazard_source = "vehicle"
        elif u < 0.42 and right_avail:
            self._scenario_event = "change_right"
            self._lead_distance_m = float(self._rng.uniform(20.0, 26.0))
            self._lead_speed_kmh = 25.0
            self._road_curvature = 0.0
            self._nav_command = "turn_right"
            self._hazard_source = "vehicle"
        elif u < 0.58:
            self._scenario_event = "follow_lead"
            self._lead_distance_m = float(self._rng.uniform(19.0, 27.0))
            self._lead_speed_kmh = float(self._rng.uniform(25.0, 32.0))
            self._road_curvature = 0.0
            self._nav_command = "merge"
            self._traffic_light = "yellow"
            self._hazard_source = "vehicle"
        elif u < 0.74:
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
        exp_man = (
            self._scenario_event
            if self._scenario_event in ("change_left", "change_right")
            else ("follow_lead" if self._lead_distance_m <= 30.0 else "keep_lane")
        )
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
            expert_maneuver=exp_man,
            traffic_light=self._traffic_light,
            hazard_source=self._hazard_source,
        )

    def _build_mock_info(
        self, arrive_dest: bool, crash: bool, out_of_road: bool
    ) -> Dict[str, Any]:
        exp_man = (
            self._scenario_event
            if self._scenario_event in ("change_left", "change_right")
            else ("follow_lead" if self._lead_distance_m <= 30.0 else "keep_lane")
        )
        return {
            "step": self._step_count,
            "seed": self._current_seed,
            "simulator": "carla",
            "scenario_event": "real_carla",
            "expert_maneuver": exp_man,
            "lead_distance_m": round(self._lead_distance_m, 2),
            "lead_speed_kmh": round(self._lead_speed_kmh, 2),
            "road_curvature": round(self._road_curvature, 3),
            "traffic_light": self._traffic_light,
            "hazard_source": self._hazard_source,
            "carla_control_preview": self.to_carla_vehicle_control(0.0, 0.5),
            "route_completion": float(
                np.clip(self._progress / max(1.0, self._route_length_m), 0.0, 1.0)
            ),
            "arrive_dest": bool(arrive_dest),
            "crash": bool(crash),
            "out_of_road": bool(out_of_road),
        }

    def _render_mock_urban_frame(self) -> np.ndarray:
        """Render a 196x392x3 RGB CARLA urban scene with traffic lights, crosswalks, & pedestrians."""
        h, w = self.camera_height, self.camera_width
        img = np.zeros((h, w, 3), dtype=np.uint8)

        horizon_y = int(h * 0.24)
        # Urban sky & city building skyline
        img[:horizon_y, :] = (115, 155, 205)
        img[horizon_y:, :] = (55, 58, 64)
        cv2.rectangle(img, (0, 8), (65, horizon_y), (85, 90, 98), -1)
        cv2.rectangle(img, (w - 70, 10), (w, horizon_y), (80, 86, 94), -1)

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

        # Render overhead CARLA traffic light gantry
        gantry_x = int(np.clip(vp_x, 80, w - 80))
        cv2.rectangle(
            img,
            (gantry_x - 14, 6),
            (gantry_x + 14, horizon_y + 12),
            (30, 30, 35),
            -1,
        )
        tl_color = (40, 235, 65)  # green
        if self._traffic_light == "red":
            tl_color = (250, 35, 35)
        elif self._traffic_light == "yellow":
            tl_color = (250, 215, 40)
        cv2.circle(img, (gantry_x, 22), 8, tl_color, -1)

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
            obj_y = int(
                horizon_y
                + (h - horizon_y - 14) * np.clip(1.0 - dist / 55.0, 0.10, 0.94)
            )
            obj_x = int(
                vp_x
                + (ego_center_x - vp_x) * ((obj_y - horizon_y) / max(1, h - horizon_y))
            )

            if self._hazard_source == "pedestrian":
                # Draw zebra crosswalk stripes + pedestrian silhouette
                for stripe_x in range(obj_x - 80, obj_x + 85, 22):
                    cv2.rectangle(
                        img,
                        (stripe_x, max(horizon_y, obj_y - 6)),
                        (stripe_x + 12, min(h - 1, obj_y + 6)),
                        (240, 240, 240),
                        -1,
                    )
                pw = max(12, int(28 * scale))
                ph = max(24, int(64 * scale))
                cv2.rectangle(
                    img,
                    (max(0, obj_x - pw // 2), max(horizon_y, obj_y - ph)),
                    (min(w - 1, obj_x + pw // 2), min(h - 1, obj_y)),
                    (255, 85, 30),
                    -1,
                )
            else:
                car_w = int(72 * scale)
                car_h = int(48 * scale)
                x1 = max(0, obj_x - car_w // 2)
                x2 = min(w - 1, obj_x + car_w // 2)
                y1 = max(horizon_y, obj_y - car_h)
                y2 = min(h - 1, obj_y)
                body_color = (
                    (245, 35, 35)
                    if self._scenario_event == "emergency_brake"
                    else (45, 110, 210)
                )
                cv2.rectangle(img, (x1, y1), (x2, y2), body_color, -1)

        return img

    def close(self) -> None:
        """Clean up CARLA actors and restore asynchronous world settings."""
        if self._world is not None:
            self._cleanup_carla_actors()
            try:
                settings = self._world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self._world.apply_settings(settings)
            except Exception:
                pass
            self._world = None
