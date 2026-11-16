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
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.control.rust_backend import get_backend
from src.env.metadrive_wrapper import NAV_COMMANDS, TelemetryState
from src.sim.carla_launcher import get_carla_server_manager

FSD_SCENARIO_NAMES: Tuple[str, ...] = (
    "FSD-1_Intersection_Turn",
    "FSD-2_Dynamic_Overtake_Left",
    "FSD-3_Pedestrian_Crosswalk_AEB",
    "FSD-4_Urban_ACC_Follow",
    "FSD-5_MultiBlock_Overtake_Right",
    "FSD-6_StopAndGo_Lead_Braking",
    "FSD-7_BlindSpot_Blocked_Overtake",
    "FSD-8_Construction_Zone_Bypass_Right",
    "FSD-9_Night_Headlight_Pedestrian_AEB",
    "FSD-10_HeavyRain_Cornering_ACC",
)

FSD_WEATHER_PRESETS: Tuple[str, ...] = (
    "ClearNoon",
    "CloudyNoon",
    "WetNoon",
    "SoftRainNoon",
    "ClearSunset",
    "WetCloudyNoon",
    "MidRainyNoon",
    "WetSunset",
    "ClearNight",
    "HardRainNoon",
)

AVAILABLE_CARLA_TOWNS: Tuple[str, ...] = (
    "Town10HD_Opt",
    "Town03_Opt",
    "Town04_Opt",
    "Town05_Opt",
    "Town01_Opt",
    "Town02_Opt",
)


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
        self.auto_launch_server: bool = bool(
            carla_cfg.get("auto_launch_server", False)
        )
        self.carla_offscreen: bool = bool(carla_cfg.get("offscreen", True))

        self.use_mock: bool = bool(
            use_mock or self.cfg.get("project", {}).get("use_mock", False)
        )

        # Live CARLA handles
        self._carla_mod: Any = None
        self._client: Any = None
        self._world: Any = None
        self._map: Any = None
        self._grp: Any = None
        self._traffic_manager: Any = None
        self._ego_vehicle: Any = None
        self._camera_sensor: Any = None
        self._hd_camera_sensor: Any = None
        self._collision_sensor: Any = None
        self._lane_invasion_sensor: Any = None
        self._radar_front_sensor: Any = None
        self._radar_bsm_left_sensor: Any = None
        self._radar_bsm_right_sensor: Any = None
        self._imu_sensor: Any = None
        self._gnss_sensor: Any = None
        self._sensor_lock = threading.Lock()
        self._latest_radar_front: List[Dict[str, Any]] = []
        self._latest_radar_left: List[Dict[str, Any]] = []
        self._latest_radar_right: List[Dict[str, Any]] = []
        self._latest_imu: Dict[str, float] = {
            "long_accel_mps2": 0.0,
            "lat_accel_mps2": 0.0,
            "yaw_rate_rps": 0.0,
            "pitch_rad": 0.0,
            "compass_deg": 0.0,
        }
        self._latest_gnss: Dict[str, float] = {
            "latitude": 0.0,
            "longitude": 0.0,
            "altitude": 0.0,
        }
        self._lane_invasion_count: int = 0
        self._step_lane_invasion: bool = False
        self._external_scenario_control: bool = False
        self._navigation_mode: str = "segment"
        self._dest_location: Optional[Tuple[float, float, float]] = None
        self._dest_spawn_idx: Optional[int] = None
        self._route_options: List[str] = []
        self._reroute_count: int = 0
        self._last_reroute_step: int = -100
        self._blocked_hold_steps: int = 0
        self._roadblock_actors: List[Any] = []
        self._behavior_agent: Any = None
        self._npc_actors: List[Any] = []
        self._scenario_actor: Any = None
        self._fsd_scenario_name: str = FSD_SCENARIO_NAMES[0]
        self._weather_preset: str = "ClearNoon"
        self._fsd_event_resolved: bool = False
        self._fsd_stop_hold_steps: int = 0
        self._fsd_lane_changed: bool = False
        self._fsd_metrics: Dict[str, int] = {
            "red_light_stops": 0,
            "lane_changes_completed": 0,
            "pedestrian_aeb_stops": 0,
            "acc_follow_episodes": 0,
            "stop_and_go_recoveries": 0,
            "blindspot_interlocks": 0,
            "night_rain_episodes": 0,
            "red_light_infractions": 0,
            "pedestrian_collisions": 0,
            "vehicle_collisions": 0,
            "off_road_infractions": 0,
        }
        self._image_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._hd_image_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._last_hd_rgb: Optional[np.ndarray] = None
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

    @property
    def step_dt(self) -> float:
        """Simulated elapsed seconds per synchronous control step in CARLA 3D or mock."""
        return float(self.fixed_delta_seconds)

    def _try_init_real_carla(self) -> None:
        """Attempt to connect to a running CARLA server (`host:port`); optionally auto-launch or fallback to urban mock."""
        if self.auto_launch_server:
            get_carla_server_manager().ensure_running(
                host=self.host,
                port=self.port,
                offscreen=self.carla_offscreen,
                quality_level="Low",
                timeout_s=60.0,
            )
        try:
            import carla  # type: ignore

            self._carla_mod = carla
            self._client = carla.Client(self.host, self.port)
            self._client.set_timeout(15.0)
            self._world = self._client.get_world()
            current_map_name = self._world.get_map().name.split("/")[-1]
            if (
                self.town
                and self.town.lower() != "current"
                and current_map_name != self.town
            ):
                self._world = self._client.load_world(self.town)
            self._map = self._world.get_map()

            settings = self._world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.fixed_delta_seconds
            self._world.apply_settings(settings)

            self._traffic_manager = self._client.get_trafficmanager(self.tm_port)
            self._traffic_manager.set_synchronous_mode(True)
            self._traffic_manager.set_global_distance_to_leading_vehicle(4.0)

            # Auto-discover CARLA PythonAPI for A* GlobalRoutePlanner
            for cand_path in (
                Path(r"C:\Users\ijh69\Downloads\CARLA_0.9.16\PythonAPI\carla"),
                Path.home() / "Downloads" / "CARLA_0.9.16" / "PythonAPI" / "carla",
            ):
                if cand_path.exists() and str(cand_path) not in sys.path:
                    sys.path.insert(0, str(cand_path))
            try:
                from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore

                self._grp = GlobalRoutePlanner(self._map, 2.0)
            except Exception:
                self._grp = None

            print(
                f"[CarlaWrapper] Connected to live CARLA server at {self.host}:{self.port} "
                f"(Map: {current_map_name}, GlobalRoutePlanner={'ON' if self._grp else 'OFF'})"
            )
        except Exception as exc:
            print(
                f"[CarlaWrapper] Live CARLA server not detected at {self.host}:{self.port} ({exc}); using fallback."
            )
            self._client = None
            self._world = None
            self.use_mock = True

    @staticmethod
    def to_carla_vehicle_control(
        steering: float, throttle_brake: float
    ) -> Dict[str, Any]:
        """Convert normalized `[steering (+left), throttle_brake]` in `[-1, 1]` to CARLA VehicleControl (`+steer = right`)."""
        steer = float(np.clip(-steering, -1.0, 1.0))
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
        sensors = (
            self._camera_sensor,
            self._hd_camera_sensor,
            self._collision_sensor,
            self._lane_invasion_sensor,
            self._radar_front_sensor,
            self._radar_bsm_left_sensor,
            self._radar_bsm_right_sensor,
            self._imu_sensor,
            self._gnss_sensor,
        )
        for sensor in sensors:
            if sensor is not None:
                try:
                    sensor.stop()
                except Exception:
                    pass
        if self._world is not None:
            try:
                self._world.tick()
            except Exception:
                pass
        for sensor in sensors:
            if sensor is not None:
                try:
                    sensor.destroy()
                except Exception:
                    pass
        self._camera_sensor = None
        self._hd_camera_sensor = None
        self._collision_sensor = None
        self._lane_invasion_sensor = None
        self._radar_front_sensor = None
        self._radar_bsm_left_sensor = None
        self._radar_bsm_right_sensor = None
        self._imu_sensor = None
        self._gnss_sensor = None
        with self._sensor_lock:
            self._latest_radar_front = []
            self._latest_radar_left = []
            self._latest_radar_right = []
        self._lane_invasion_count = 0
        self._step_lane_invasion = False

        if self._ego_vehicle is not None:
            try:
                self._ego_vehicle.destroy()
            except Exception:
                pass
            self._ego_vehicle = None

        if self._scenario_actor is not None:
            try:
                self._scenario_actor.destroy()
            except Exception:
                pass
            self._scenario_actor = None

        self._npc_actors.clear()
        self._roadblock_actors.clear()

        # Sweep all vehicles or walkers in the persistent CARLA world
        if self._world is not None:
            try:
                for actor in self._world.get_actors():
                    t_id = str(getattr(actor, "type_id", ""))
                    if t_id.startswith("vehicle.") or t_id.startswith("walker.pedestrian."):
                        try:
                            actor.destroy()
                        except Exception:
                            pass
            except Exception:
                pass

        for q in (self._image_queue, self._hd_image_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break

    def _update_spectator_chase_cam(self) -> None:
        """Position the main CarlaUE4.exe 3D Spectator camera behind the ego vehicle with gimbal smoothing."""
        if self._world is None or self._ego_vehicle is None:
            return
        try:
            carla = self._carla_mod
            ego_tf = self._ego_vehicle.get_transform()
            ego_yaw_deg = float(ego_tf.rotation.yaw)
            prev_cam = getattr(self, "_spectator_cam_state", None)
            if prev_cam is None:
                sm_yaw_deg = ego_yaw_deg
                fwd = ego_tf.get_forward_vector()
                sm_x = float(ego_tf.location.x - fwd.x * 6.5)
                sm_y = float(ego_tf.location.y - fwd.y * 6.5)
                sm_z = float(ego_tf.location.z + 2.8)
            else:
                px, py, pz, pyaw_deg = prev_cam
                # Wrap angle difference in degrees for smooth yaw interpolation
                dyaw = ((ego_yaw_deg - pyaw_deg + 180.0) % 360.0) - 180.0
                sm_yaw_deg = pyaw_deg + 0.22 * dyaw
                sm_yaw_rad = math.radians(sm_yaw_deg)
                target_x = float(ego_tf.location.x - math.cos(sm_yaw_rad) * 6.5)
                target_y = float(ego_tf.location.y - math.sin(sm_yaw_rad) * 6.5)
                target_z = float(ego_tf.location.z + 2.8)
                sm_x = px + 0.35 * (target_x - px)
                sm_y = py + 0.35 * (target_y - py)
                sm_z = pz + 0.35 * (target_z - pz)

            self._spectator_cam_state = (sm_x, sm_y, sm_z, sm_yaw_deg)
            cam_loc = carla.Location(x=sm_x, y=sm_y, z=sm_z)
            cam_rot = carla.Rotation(pitch=-12.0, yaw=sm_yaw_deg, roll=0.0)
            self._world.get_spectator().set_transform(carla.Transform(cam_loc, cam_rot))
        except Exception:
            pass

    @staticmethod
    def _compute_polyline_tangents(
        xyz_list: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float, float]]:
        """Attach continuous forward tangent yaw_rad to a list of (x, y, z) 3D points via Rust core."""
        return list(get_backend().compute_polyline_tangents(xyz_list))

    @staticmethod
    def _smooth_parallel_lane_jumps(
        xyz: List[List[float]], wp_yaws: List[float]
    ) -> None:
        """In-place Minimum-Jerk 5th-Order Quintic smoothing of raw ~3.5m parallel-lane jumps."""
        n_pts = len(xyz)
        if n_pts < 4 or len(wp_yaws) != n_pts:
            return
        for k in range(1, n_pts):
            dx_k = xyz[k][0] - xyz[k - 1][0]
            dy_k = xyz[k][1] - xyz[k - 1][1]
            yaw_prev = wp_yaws[k - 1]
            yaw_curr = wp_yaws[k]
            lat_jump = -dx_k * math.sin(yaw_prev) + dy_k * math.cos(yaw_prev)
            if abs(lat_jump) >= 1.30 and math.cos(yaw_curr - yaw_prev) >= 0.72:
                m_blend = min(14, n_pts - k)
                for step_j in range(m_blend):
                    idx_b = k + step_j
                    tau = float(step_j + 1) / float(m_blend + 1)
                    q_tau = (10.0 * (tau ** 3)) - (15.0 * (tau ** 4)) + (6.0 * (tau ** 5))
                    rem_shift = -(1.0 - q_tau) * lat_jump
                    yb = wp_yaws[idx_b]
                    xyz[idx_b][0] += -math.sin(yb) * rem_shift
                    xyz[idx_b][1] += math.cos(yb) * rem_shift

    def _rebuild_route_pts(self) -> None:
        """Build C2-continuous 3D polyline `self._route_pts` from `self._route_waypoints`.

        CARLA's `GlobalRoutePlanner.trace_route()` encodes lane changes as discontinuous
        ~3.5m perpendicular jumps between adjacent waypoints. We detect any parallel-lane
        step jump and blend it over a ~28m Minimum-Jerk 5th-Order Quintic S-curve so
        downstream path-tracking never encounters artificial 90-degree tangent kinks.
        """
        wps = getattr(self, "_route_waypoints", [])
        xyz: List[List[float]] = [
            [
                float(w.transform.location.x),
                float(w.transform.location.y),
                float(w.transform.location.z),
            ]
            for w in wps
        ]
        if len(xyz) >= 4:
            wp_yaws = [math.radians(float(w.transform.rotation.yaw)) for w in wps]
            self._smooth_parallel_lane_jumps(xyz, wp_yaws)

        xyz_tuples = [(p[0], p[1], p[2]) for p in xyz]
        self._route_pts: List[Tuple[float, float, float, float]] = (
            self._compute_polyline_tangents(xyz_tuples)
        )

    def _trace_astar_waypoints(
        self, start_loc: Any, end_loc: Any
    ) -> Tuple[List[Any], List[str]]:
        """Trace global A* route `(waypoints, road_options)` using CARLA GlobalRoutePlanner."""
        if self._grp is None:
            return [], []
        try:
            traced = self._grp.trace_route(start_loc, end_loc)
        except Exception:
            return [], []
        wps: List[Any] = []
        opts: List[str] = []
        for wp, opt in traced:
            loc = wp.transform.location
            if wps:
                prev_loc = wps[-1].transform.location
                if math.hypot(float(loc.x - prev_loc.x), float(loc.y - prev_loc.y)) < 0.45:
                    continue
            wps.append(wp)
            opts.append(getattr(opt, "name", str(opt)).split(".")[-1].lower())
        return wps, opts

    @staticmethod
    def _polyline_length_m(pts: List[Tuple[float, float, float, float]], start_i: int = 0) -> float:
        total = 0.0
        for idx in range(max(1, start_i + 1), len(pts)):
            total += math.hypot(pts[idx][0] - pts[idx - 1][0], pts[idx][1] - pts[idx - 1][1])
        return float(total)

    def set_town(self, town_name: str) -> str:
        """Switch CARLA Unreal map at runtime (`Town01_Opt`..`Town10HD_Opt`) and rebuild A* planner."""
        target = str(town_name or "Town10HD_Opt").strip()
        self.town = target
        self._dest_location = None
        self._dest_spawn_idx = None
        self._navigation_mode = "fsd"
        self._reroute_count = 0
        if not self.use_mock and self._client is not None:
            self._cleanup_carla_actors()
            try:
                cur_name = self._world.get_map().name.split("/")[-1] if self._world is not None else ""
                if target.lower() != "current" and cur_name != target:
                    self._world = self._client.load_world(target)
                self._map = self._world.get_map()
                settings = self._world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.fixed_delta_seconds
                self._world.apply_settings(settings)
                if self._traffic_manager is not None:
                    self._traffic_manager.set_synchronous_mode(True)
                    self._traffic_manager.set_global_distance_to_leading_vehicle(4.0)
                try:
                    from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore

                    self._grp = GlobalRoutePlanner(self._map, 2.0)
                except Exception:
                    self._grp = None
                return str(self._world.get_map().name.split("/")[-1])
            except Exception:
                pass
        return self.town

    def list_spawn_points(self) -> List[Dict[str, Any]]:
        """Return road spawn/destination nodes `[{index, x, y, z, yaw}, ...]` on the active CARLA map."""
        if not self.use_mock and self._map is not None:
            try:
                sps = list(self._map.get_spawn_points())
                return [
                    {
                        "index": idx,
                        "x": round(float(sp.location.x), 1),
                        "y": round(float(sp.location.y), 1),
                        "z": round(float(sp.location.z), 1),
                        "yaw": round(float(sp.rotation.yaw), 1),
                    }
                    for idx, sp in enumerate(sps)
                ]
            except Exception:
                pass
        # Deterministic 12-node urban grid in mock mode
        mock_nodes: List[Dict[str, Any]] = []
        for idx in range(12):
            gx = float((idx % 4) * 95.0)
            gy = float(((idx // 4) - 1) * 75.0)
            mock_nodes.append({"index": idx, "x": gx, "y": gy, "z": 0.0, "yaw": 0.0})
        return mock_nodes

    def set_destination(
        self,
        dest_spawn_idx: Optional[int] = None,
        dest_xy: Optional[Tuple[float, float]] = None,
        min_distance_m: float = 170.0,
        enable_point_to_point: bool = True,
    ) -> Dict[str, Any]:
        """Set or dynamically update the global destination and re-plan live A* route immediately."""
        if enable_point_to_point:
            self._navigation_mode = "point_to_point"

        nodes = self.list_spawn_points()
        if not nodes:
            return {"rerouted": False, "reroute_count": self._reroute_count}

        ego_xy = (0.0, 0.0)
        if not self.use_mock and self._ego_vehicle is not None:
            loc = self._ego_vehicle.get_location()
            ego_xy = (float(loc.x), float(loc.y))
        else:
            ego_xy = (float(self._progress), float(self._lane_offset_m))

        chosen = nodes[-1]
        if dest_spawn_idx is not None:
            chosen = nodes[int(dest_spawn_idx) % len(nodes)]
        elif dest_xy is not None:
            tx, ty = float(dest_xy[0]), float(dest_xy[1])
            chosen = min(
                nodes,
                key=lambda n: math.hypot(float(n["x"]) - tx, float(n["y"]) - ty),
            )
        else:
            distant = [
                n
                for n in nodes
                if math.hypot(float(n["x"]) - ego_xy[0], float(n["y"]) - ego_xy[1]) >= float(min_distance_m)
            ]
            if distant:
                base_pick = (self._current_seed + self._reroute_count + 3) % len(distant)
                chosen = distant[base_pick]
                # Prefer a destination whose actual A* route distance is [min_distance_m, 340m]
                # rather than a 2km+ opposite-highway cloverleaf loop
                if (
                    not self.use_mock
                    and self._world is not None
                    and self._ego_vehicle is not None
                    and self._grp is not None
                    and self._carla_mod is not None
                ):
                    carla = self._carla_mod
                    ego_tr_c = self._ego_vehicle.get_transform()
                    ego_loc_c = ego_tr_c.location
                    ego_yaw_c = math.radians(float(ego_tr_c.rotation.yaw))
                    fwd_x, fwd_y = math.cos(ego_yaw_c), math.sin(ego_yaw_c)

                    # Prioritize forward-aligned destinations within reasonable Euclidean range first
                    def _cand_rank(idx_d: int) -> Tuple[int, float]:
                        nd = distant[idx_d]
                        rx_d = float(nd["x"]) - ego_xy[0]
                        ry_d = float(nd["y"]) - ego_xy[1]
                        euc = math.hypot(rx_d, ry_d)
                        fwd_dot = (rx_d * fwd_x + ry_d * fwd_y) / max(1.0, euc)
                        in_sweet = 0 if (fwd_dot > 0.15 and euc <= 330.0) else (1 if euc <= 360.0 else 2)
                        return (in_sweet, abs(euc - 210.0))

                    rot_indices = [(base_pick + j) % len(distant) for j in range(len(distant))]
                    rot_indices.sort(key=_cand_rank)

                    best_fallback = None
                    best_fb_len = 1e9
                    found_in_range = False
                    for idx_d in rot_indices[: min(160, len(rot_indices))]:
                        cand_n = distant[idx_d]
                        c_loc = carla.Location(
                            x=float(cand_n["x"]),
                            y=float(cand_n["y"]),
                            z=float(cand_n.get("z", 0.0)),
                        )
                        c_wps, _c_opts = self._trace_astar_waypoints(ego_loc_c, c_loc)
                        est_len = float(len(c_wps)) * 2.0
                        if float(min_distance_m) * 0.82 <= est_len <= 340.0:
                            chosen = cand_n
                            found_in_range = True
                            break
                        if float(min_distance_m) * 0.72 <= est_len < best_fb_len:
                            best_fb_len = est_len
                            best_fallback = cand_n
                    if not found_in_range and best_fallback is not None:
                        chosen = best_fallback
            else:
                chosen = max(
                    nodes,
                    key=lambda n: math.hypot(float(n["x"]) - ego_xy[0], float(n["y"]) - ego_xy[1]),
                )

        self._dest_spawn_idx = int(chosen["index"])
        self._dest_location = (
            float(chosen["x"]),
            float(chosen["y"]),
            float(chosen.get("z", 0.0)),
        )

        if not self.use_mock and self._world is not None and self._ego_vehicle is not None:
            return self.replan_from_current_pose(reason="set_destination")

        # Mock mode live destination re-plan
        dist_goal = max(
            90.0,
            math.hypot(self._dest_location[0] - ego_xy[0], self._dest_location[1] - ego_xy[1]),
        )
        self._route_length_m = float(self._progress + dist_goal)
        self._reroute_count += 1
        return {
            "rerouted": True,
            "reason": "set_destination",
            "dest_index": self._dest_spawn_idx,
            "dest_xy": [self._dest_location[0], self._dest_location[1]],
            "remaining_route_m": round(dist_goal, 1),
            "reroute_count": self._reroute_count,
        }

    def replan_from_current_pose(self, reason: str = "dynamic_replan") -> Dict[str, Any]:
        """Re-compute A* route from Ego's live CARLA pose to `self._dest_location` without respawning."""
        if (
            self.use_mock
            or self._world is None
            or self._ego_vehicle is None
            or self._carla_mod is None
            or self._dest_location is None
        ):
            return {"rerouted": False, "reason": reason, "reroute_count": self._reroute_count}

        carla = self._carla_mod
        ego_loc = self._ego_vehicle.get_location()
        dest_loc = carla.Location(
            x=float(self._dest_location[0]),
            y=float(self._dest_location[1]),
            z=float(self._dest_location[2]),
        )
        wps, opts = self._trace_astar_waypoints(ego_loc, dest_loc)
        if len(wps) < 4 and self._map is not None:
            start_wp = self._map.get_waypoint(
                ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if start_wp is not None:
                wps = [start_wp]
                curr = start_wp
                for _ in range(90):
                    nxt = curr.next(2.0)
                    if not nxt:
                        break
                    curr = nxt[0]
                    wps.append(curr)
                opts = ["lanefollow"] * len(wps)

        if len(wps) >= 4:
            self._route_waypoints = wps
            self._route_options = opts
            self._route_wp_idx = 0
            self._fsd_lane_changed = False
            self._rebuild_route_pts()
            rem_m = max(18.0, self._polyline_length_m(self._route_pts, 0))
            self._route_length_m = float(self._progress + rem_m)
            self._reroute_count += 1
            self._last_reroute_step = self._step_count
            return {
                "rerouted": True,
                "reason": reason,
                "dest_index": self._dest_spawn_idx,
                "dest_xy": [float(self._dest_location[0]), float(self._dest_location[1])],
                "remaining_route_m": round(rem_m, 1),
                "reroute_count": self._reroute_count,
            }
        return {"rerouted": False, "reason": reason, "reroute_count": self._reroute_count}

    def spawn_roadblock(self, distance_ahead_m: float = 24.0) -> Dict[str, Any]:
        """Spawn a physical stalled vehicle blocking Ego's active lane to test live dynamic bypass & A* re-routing."""
        if not self.use_mock and self._world is not None and self._ego_vehicle is not None and self._carla_mod is not None:
            carla = self._carla_mod
            wps = getattr(self, "_route_waypoints", [])
            step_ahead = max(5, int(round(float(distance_ahead_m) / 2.0)))
            target_idx = min(len(wps) - 1, self._route_wp_idx + step_ahead)
            if wps and target_idx > self._route_wp_idx:
                obs_wp = None
                # Select the nearest multi-lane waypoint ahead so lane-change bypass is physically possible
                for idx_c in range(target_idx, min(len(wps) - 3, target_idx + 45)):
                    w_c = wps[idx_c]
                    lw = w_c.get_left_lane()
                    rw = w_c.get_right_lane()
                    has_adj = bool(
                        (lw is not None and lw.lane_type == carla.LaneType.Driving and lw.lane_id * w_c.lane_id > 0)
                        or (rw is not None and rw.lane_type == carla.LaneType.Driving and rw.lane_id * w_c.lane_id > 0)
                    )
                    if has_adj:
                        obs_wp = w_c
                        break
                if obs_wp is None:
                    return {"spawned": False, "reason": "single_lane_corridor"}
                obs_tf = carla.Transform(
                    carla.Location(
                        x=obs_wp.transform.location.x,
                        y=obs_wp.transform.location.y,
                        z=obs_wp.transform.location.z + 0.35,
                    ),
                    obs_wp.transform.rotation,
                )
                bp_lib = self._world.get_blueprint_library()
                cands = [
                    bp
                    for bp in bp_lib.filter("vehicle.*")
                    if bp.has_attribute("number_of_wheels")
                    and int(bp.get_attribute("number_of_wheels")) == 4
                ]
                if cands:
                    actor = self._world.try_spawn_actor(cands[0], obs_tf)
                    if actor is not None:
                        actor.apply_control(
                            carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
                        )
                        self._roadblock_actors.append(actor)
                        return {
                            "spawned": True,
                            "x": round(float(obs_tf.location.x), 1),
                            "y": round(float(obs_tf.location.y), 1),
                        }
            return {"spawned": False}

        # Mock roadblock injection
        self._lead_distance_m = float(distance_ahead_m)
        self._lead_speed_kmh = 0.0
        self._hazard_source = "vehicle"
        self._scenario_event = "change_left" if self._current_lane_idx > 0 else "change_right"
        return {"spawned": True, "x": round(self._progress + distance_ahead_m, 1), "y": 0.0}

    def _build_real_nav_state(self, ego_tr: Any, best_i: int) -> Dict[str, Any]:
        """Build lightweight 2D city map state (`<= 100` route points) for live WebSocket cockpit UI."""
        pts = getattr(self, "_route_pts", [])
        rem_pts = pts[max(0, best_i) :]
        stride = max(1, len(rem_pts) // 85)
        route_xy = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in rem_pts[::stride]]
        if rem_pts and (not route_xy or route_xy[-1] != [round(float(rem_pts[-1][0]), 1), round(float(rem_pts[-1][1]), 1)]):
            route_xy.append([round(float(rem_pts[-1][0]), 1), round(float(rem_pts[-1][1]), 1)])
        rem_m = self._polyline_length_m(pts, max(0, best_i))
        dest_xy = (
            [round(float(self._dest_location[0]), 1), round(float(self._dest_location[1]), 1)]
            if self._dest_location is not None
            else (route_xy[-1] if route_xy else [round(float(ego_tr.location.x), 1), round(float(ego_tr.location.y), 1)])
        )
        map_name = (
            self._world.get_map().name.split("/")[-1]
            if self._world is not None
            else str(self.town)
        )
        return {
            "map_name": map_name,
            "navigation_mode": self._navigation_mode,
            "ego_xy_yaw": [
                round(float(ego_tr.location.x), 2),
                round(float(ego_tr.location.y), 2),
                round(math.radians(float(ego_tr.rotation.yaw)), 3),
            ],
            "dest_xy": dest_xy,
            "dest_index": self._dest_spawn_idx,
            "active_route_xy": route_xy,
            "spawn_nodes": self.list_spawn_points()[:60],
            "remaining_route_m": round(rem_m, 1),
            "reroute_count": int(self._reroute_count),
        }

    def _switch_route_to_adjacent_lane(self, direction: str, blend_steps: int = 14) -> bool:
        """Switch route to adjacent lane ('left' or 'right') via Minimum-Jerk Quintic S-Curve + A* splice."""
        if not getattr(self, "_route_waypoints", None) or self._carla_mod is None:
            return False
        carla = self._carla_mod
        start_idx = min(len(self._route_waypoints) - 1, self._route_wp_idx + 1)
        pivot_wp = self._route_waypoints[start_idx]
        adj_wp = (
            pivot_wp.get_left_lane()
            if direction == "left"
            else pivot_wp.get_right_lane()
        )
        if (
            adj_wp is None
            or adj_wp.lane_type != carla.LaneType.Driving
            or (adj_wp.lane_id * pivot_wp.lane_id <= 0)
        ):
            return False

        # Generate adjacent lane chain starting from pivot_wp
        adj_chain = [adj_wp]
        curr_wp = adj_wp
        for _ in range(80):
            nxt = curr_wp.next(2.0)
            if not nxt:
                break
            curr_wp = nxt[0]
            adj_chain.append(curr_wp)

        orig_chain = self._route_waypoints[start_idx:]
        n_blend = min(max(6, int(blend_steps)), len(orig_chain), len(adj_chain))
        n_hold = min(len(adj_chain), n_blend + 18)

        # In Point-to-Point navigation mode, hold the adjacent bypass lane for ~36m past the
        # S-curve so Ego completely clears any stalled roadblock before splicing A* back to _dest_location!
        spliced_opts: List[str] = []
        if (
            self._navigation_mode == "point_to_point"
            and self._dest_location is not None
            and self._grp is not None
            and n_hold >= 2
        ):
            s_end_wp = adj_chain[n_hold - 1]
            dest_loc = carla.Location(
                x=float(self._dest_location[0]),
                y=float(self._dest_location[1]),
                z=float(self._dest_location[2]),
            )
            tail_wps, tail_opts = self._trace_astar_waypoints(
                s_end_wp.transform.location, dest_loc
            )
            if len(tail_wps) >= 4:
                adj_chain = list(adj_chain[:n_hold]) + tail_wps[1:]
                spliced_opts = ["lanefollow"] * n_hold + tail_opts[1:]
                self._reroute_count += 1
                self._last_reroute_step = self._step_count

        prefix_xyz = [
            (p[0], p[1], p[2]) for p in getattr(self, "_route_pts", [])[:start_idx]
        ]
        orig_xyz = [
            (
                float(w.transform.location.x),
                float(w.transform.location.y),
                float(w.transform.location.z),
            )
            for w in orig_chain[:n_blend]
        ]
        adj_xyz_mut: List[List[float]] = [
            [
                float(w.transform.location.x),
                float(w.transform.location.y),
                float(w.transform.location.z),
            ]
            for w in adj_chain
        ]
        if len(adj_xyz_mut) >= 4:
            adj_yaws = [math.radians(float(w.transform.rotation.yaw)) for w in adj_chain]
            self._smooth_parallel_lane_jumps(adj_xyz_mut, adj_yaws)
        adj_xyz = [(p[0], p[1], p[2]) for p in adj_xyz_mut]
        blended_xyz = list(
            get_backend().quintic_s_curve_blend(orig_xyz, adj_xyz, n_blend)
        )

        self._route_pts = self._compute_polyline_tangents(prefix_xyz + blended_xyz)
        self._route_waypoints = list(self._route_waypoints[:start_idx]) + adj_chain
        if spliced_opts:
            prev_opts = list(getattr(self, "_route_options", []))[:start_idx]
            self._route_options = prev_opts + spliced_opts
            rem_m = max(18.0, self._polyline_length_m(self._route_pts, start_idx))
            self._route_length_m = float(self._progress + rem_m)
        else:
            self._route_options = ["lanefollow"] * len(self._route_waypoints)
        self._lane_change_end_idx = start_idx + n_blend
        self._fsd_lane_changed = True
        self._fsd_metrics["lane_changes_completed"] += 1
        return True

    def _sample_continuous_route_pose(
        self,
        x: float,
        y: float,
        start_idx: int,
        lookahead_m: float = 8.0,
    ) -> Tuple[int, float, float, float, float, float, float]:
        """Project (x, y) onto `self._route_pts` and interpolate a continuous lookahead point via Rust core.

        Returns `(best_idx, proj_x, proj_y, proj_yaw, look_x, look_y, ref_yaw)`.
        """
        pts = getattr(self, "_route_pts", [])
        if not pts:
            return 0, x, y, 0.0, x + lookahead_m, y, 0.0
        return tuple(  # type: ignore[return-value]
            get_backend().sample_continuous_route_pose(
                pts, float(x), float(y), max(0, int(start_idx)), float(lookahead_m)
            )
        )

    def _step_fsd4_lead_vehicle(self) -> None:
        """Actively drive the ACC/Stop-and-Go lead vehicle along the route (FSD-4, FSD-6, FSD-10)."""
        if (
            self._fsd_scenario_name
            not in (
                "FSD-4_Urban_ACC_Follow",
                "FSD-6_StopAndGo_Lead_Braking",
                "FSD-10_HeavyRain_Cornering_ACC",
            )
            or self._scenario_actor is None
            or self._carla_mod is None
        ):
            return
        try:
            carla = self._carla_mod
            lead = self._scenario_actor
            if lead.is_at_traffic_light():
                tl = lead.get_traffic_light()
                if tl is not None:
                    try:
                        tl.set_state(carla.TrafficLightState.Green)
                    except Exception:
                        pass
            a_tr = lead.get_transform()
            a_loc = a_tr.location
            a_vel = lead.get_velocity()
            lead_speed_kmh = float(3.6 * math.hypot(a_vel.x, a_vel.y))
            lead_idx = int(getattr(self, "_lead_route_idx", 8))
            new_idx, _, _, _, lx, ly, _ = self._sample_continuous_route_pose(
                float(a_loc.x), float(a_loc.y), lead_idx, lookahead_m=7.5
            )
            self._lead_route_idx = new_idx
            a_yaw = math.radians(float(a_tr.rotation.yaw))
            target_yaw = math.atan2(ly - float(a_loc.y), lx - float(a_loc.x))
            yaw_err = math.atan2(
                math.sin(target_yaw - a_yaw), math.cos(target_yaw - a_yaw)
            )
            lead_steer = float(np.clip(yaw_err * 0.90, -0.45, 0.45))

            # In FSD-6, lead vehicle performs an abrupt mid-block stop, then re-accelerates
            if (
                self._fsd_scenario_name == "FSD-6_StopAndGo_Lead_Braking"
                and not self._fsd_event_resolved
                and 14.0 <= self._progress <= 24.0
            ):
                target_lead_kmh = 0.0
            else:
                target_lead_kmh = 24.0

            speed_err = target_lead_kmh - lead_speed_kmh
            if target_lead_kmh <= 0.5:
                lead_throttle = 0.0
                lead_brake = 1.0
            elif speed_err >= -0.8:
                lead_throttle = float(np.clip(0.44 + 0.05 * speed_err, 0.18, 0.68))
                lead_brake = 0.0
            else:
                lead_throttle = 0.0
                lead_brake = float(np.clip(-0.08 * speed_err, 0.0, 0.35))
            lead.apply_control(
                carla.VehicleControl(
                    throttle=round(lead_throttle, 4),
                    steer=round(lead_steer, 4),
                    brake=round(lead_brake, 4),
                )
            )
        except Exception:
            pass

    def _spawn_real_carla_episode(self, seed: int) -> None:
        """Spawn ego vehicle, FSD scenario actors, HD chase cam, and A* route in CARLA."""
        carla = self._carla_mod
        self._cleanup_carla_actors()
        self._collision_detected = False
        self._out_of_road_detected = False
        self._progress = 0.0
        self._last_hd_rgb = None
        self._fsd_event_resolved = False
        self._fsd_stop_hold_steps = 0
        self._fsd_lane_changed = False
        self._lane_change_ramp_m = 0.0
        self._lane_change_end_idx = 0
        self._lead_route_idx = 8
        self._spectator_cam_state = None
        self._carla_steer_prev = 0.0
        self._last_gated_maneuver = "keep_lane"
        self._step_lane_invasion = False
        self._reroute_count = 0
        self._last_reroute_step = -100
        self._blocked_hold_steps = 0

        ep_offset = int(abs(seed - 5000))
        ext_ctrl = bool(getattr(self, "_external_scenario_control", False))
        if ext_ctrl:
            self._fsd_scenario_name = "EXTERNAL_STRESS_DIRECTOR"
            self._weather_preset = str(getattr(self, "_weather_preset", "ClearNoon"))
            if any(k in self._weather_preset for k in ("Night", "Rain", "HardRain", "MidRainy")):
                self._fsd_metrics["night_rain_episodes"] += 1
            self.set_weather(self._weather_preset)
        else:
            scenario_idx = ep_offset % len(FSD_SCENARIO_NAMES)
            weather_shift = (ep_offset // len(FSD_SCENARIO_NAMES)) * 2
            self._fsd_scenario_name = FSD_SCENARIO_NAMES[scenario_idx]
            self._weather_preset = FSD_WEATHER_PRESETS[
                (scenario_idx + weather_shift) % len(FSD_WEATHER_PRESETS)
            ]
            if self._weather_preset in ("ClearNight", "HardRainNoon", "MidRainyNoon"):
                self._fsd_metrics["night_rain_episodes"] += 1
            try:
                weather_param = getattr(carla.WeatherParameters, self._weather_preset, None)
                if weather_param is not None:
                    self._world.set_weather(weather_param)
                elif self._weather_preset == "ClearNight":
                    w_night = carla.WeatherParameters(
                        cloudiness=10.0,
                        precipitation=0.0,
                        sun_altitude_angle=-30.0,
                    )
                    self._world.set_weather(w_night)
            except Exception:
                pass

        bp_lib = self._world.get_blueprint_library()
        all_sps = list(self._map.get_spawn_points())
        rng = np.random.default_rng(seed)

        four_wheel_bps = [
            bp
            for bp in bp_lib.filter("vehicle.*")
            if bp.has_attribute("number_of_wheels")
            and int(bp.get_attribute("number_of_wheels")) == 4
        ]
        ego_bps = bp_lib.filter(self.vehicle_filter)
        if len(ego_bps) == 0:
            for preferred in (
                "vehicle.lincoln.mkz_2020",
                "vehicle.tesla.model3",
                "vehicle.audi.etron",
                "vehicle.ford.mustang",
            ):
                cand = bp_lib.filter(preferred)
                if len(cand) > 0:
                    ego_bps = cand
                    break
        ego_bp = (
            ego_bps[0]
            if len(ego_bps) > 0
            else (four_wheel_bps[0] if len(four_wheel_bps) > 0 else bp_lib.filter("vehicle.*")[0])
        )
        if ego_bp.has_attribute("role_name"):
            ego_bp.set_attribute("role_name", "hero")

        # Select scenario-specific spawn point indices in Town10HD_Opt
        scenario_sp_map = {
            "FSD-1_Intersection_Turn": [1, 12, 14, 15, 2, 17, 21],
            "FSD-2_Dynamic_Overtake_Left": [4, 7, 9, 10, 20, 23, 1],
            "FSD-3_Pedestrian_Crosswalk_AEB": [5, 8, 22, 6, 19, 4],
            "FSD-4_Urban_ACC_Follow": [7, 10, 4, 9, 20, 23],
            "FSD-5_MultiBlock_Overtake_Right": [6, 19, 22, 5, 8, 3],
            "FSD-6_StopAndGo_Lead_Braking": [7, 10, 4, 9, 20, 23],
            "FSD-7_BlindSpot_Blocked_Overtake": [4, 7, 9, 10, 20, 23, 1],
            "FSD-8_Construction_Zone_Bypass_Right": [6, 19, 22, 5, 8, 3],
            "FSD-9_Night_Headlight_Pedestrian_AEB": [5, 8, 22, 6, 19, 4],
            "FSD-10_HeavyRain_Cornering_ACC": [7, 10, 4, 9, 20, 23],
        }
        cur_map_short = self._map.name.split("/")[-1] if self._map is not None else ""
        if ext_ctrl:
            if "Town10" in cur_map_short:
                urban_sps = [4, 7, 10, 1, 5, 6, 9, 20, 22, 23]
                rot = ep_offset % len(urban_sps)
                pref_indices = urban_sps[rot:] + urban_sps[:rot]
                ordered_sps = [all_sps[i] for i in pref_indices if i < len(all_sps)] + all_sps
            else:
                # On Town01..Town05, prioritize multi-lane arterial/highway corridors (not 1-lane dead-end alleys)
                multilane_sps = []
                singlelane_sps = []
                for cand_sp in all_sps:
                    w0 = self._map.get_waypoint(
                        cand_sp.location, project_to_road=True, lane_type=carla.LaneType.Driving
                    )
                    if w0 is None:
                        continue
                    lw0 = w0.get_left_lane()
                    rw0 = w0.get_right_lane()
                    if (
                        (lw0 is not None and lw0.lane_type == carla.LaneType.Driving and lw0.lane_id * w0.lane_id > 0)
                        or (rw0 is not None and rw0.lane_type == carla.LaneType.Driving and rw0.lane_id * w0.lane_id > 0)
                    ):
                        multilane_sps.append(cand_sp)
                    else:
                        singlelane_sps.append(cand_sp)
                pool = multilane_sps if multilane_sps else singlelane_sps
                if pool:
                    rot_p = ep_offset % len(pool)
                    ordered_sps = pool[rot_p:] + pool[:rot_p] + singlelane_sps
                else:
                    ordered_sps = all_sps
        else:
            pref_indices = scenario_sp_map.get(self._fsd_scenario_name, [1, 4, 5, 7, 6])
            ordered_sps = [all_sps[i] for i in pref_indices if i < len(all_sps)] + all_sps

        for sp in ordered_sps:
            start_wp = self._map.get_waypoint(
                sp.location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if start_wp is None:
                continue
            if self._fsd_scenario_name in (
                "FSD-2_Dynamic_Overtake_Left",
                "FSD-7_BlindSpot_Blocked_Overtake",
            ):
                l_wp = start_wp.get_left_lane()
                if not (
                    l_wp is not None
                    and l_wp.lane_type == carla.LaneType.Driving
                    and l_wp.lane_id * start_wp.lane_id > 0
                ):
                    continue
            elif self._fsd_scenario_name in (
                "FSD-5_MultiBlock_Overtake_Right",
                "FSD-8_Construction_Zone_Bypass_Right",
            ):
                r_wp = start_wp.get_right_lane()
                if not (
                    r_wp is not None
                    and r_wp.lane_type == carla.LaneType.Driving
                    and r_wp.lane_id * start_wp.lane_id > 0
                ):
                    continue

            wp_chain = [start_wp]
            wp_opts = ["lanefollow"]
            curr_wp = start_wp
            for _ in range(95):
                nxt = curr_wp.next(2.0)
                if not nxt:
                    break
                curr_wp = nxt[0]
                wp_chain.append(curr_wp)
                wp_opts.append("lanefollow")
            if len(wp_chain) < 42:
                continue

            if self._navigation_mode == "point_to_point" and self._grp is not None and all_sps:
                # Select an arbitrary distant destination (>= 170m away) across the city graph
                if self._dest_spawn_idx is not None and 0 <= self._dest_spawn_idx < len(all_sps):
                    dest_idx = self._dest_spawn_idx
                else:
                    far_indices = [
                        idx_s
                        for idx_s, cand_sp in enumerate(all_sps)
                        if math.hypot(
                            float(cand_sp.location.x - sp.location.x),
                            float(cand_sp.location.y - sp.location.y),
                        )
                        >= 170.0
                    ]
                    if far_indices:
                        dest_idx = far_indices[(ep_offset + 3) % len(far_indices)]
                    else:
                        dest_idx = max(
                            range(len(all_sps)),
                            key=lambda idx_s: math.hypot(
                                float(all_sps[idx_s].location.x - sp.location.x),
                                float(all_sps[idx_s].location.y - sp.location.y),
                            ),
                        )
                dest_sp = all_sps[dest_idx]
                traced_wps, traced_opts = self._trace_astar_waypoints(sp.location, dest_sp.location)
                if len(traced_wps) >= 35:
                    wp_chain = traced_wps
                    wp_opts = traced_opts
                    self._dest_spawn_idx = int(dest_idx)
                    self._dest_location = (
                        float(dest_sp.location.x),
                        float(dest_sp.location.y),
                        float(dest_sp.location.z),
                    )
            elif self._grp is not None and len(wp_chain) >= 40:
                # Segment mode: refine 90m local route with A* trace_route
                try:
                    dest_seg = wp_chain[min(len(wp_chain) - 1, 45)].transform.location
                    traced_wps, traced_opts = self._trace_astar_waypoints(sp.location, dest_seg)
                    if len(traced_wps) >= 25:
                        wp_chain = traced_wps
                        wp_opts = traced_opts
                except Exception:
                    pass

            self._ego_vehicle = self._world.try_spawn_actor(ego_bp, sp)
            if self._ego_vehicle is not None:
                self._route_waypoints = wp_chain
                self._route_options = wp_opts
                self._route_wp_idx = 0
                self._rebuild_route_pts()
                if self._navigation_mode == "point_to_point":
                    full_len = self._polyline_length_m(self._route_pts, 0)
                    self._route_length_m = max(68.0, full_len)
                    if self._dest_location is None and wp_chain:
                        dl = wp_chain[-1].transform.location
                        self._dest_location = (float(dl.x), float(dl.y), float(dl.z))
                else:
                    self._route_length_m = min(self.target_route_distance_m, 68.0)
                    if wp_chain:
                        dl = wp_chain[-1].transform.location
                        self._dest_location = (float(dl.x), float(dl.y), float(dl.z))
                break

        if self._ego_vehicle is None:
            raise RuntimeError("Failed to spawn CARLA ego vehicle.")

        # Turn on ego vehicle headlights during night, sunset, or rain scenarios
        try:
            if hasattr(carla, "VehicleLightState"):
                light_mask = (
                    carla.VehicleLightState.Position
                    | carla.VehicleLightState.LowBeam
                )
                if "Night" in self._weather_preset or "Rain" in self._weather_preset:
                    light_mask |= carla.VehicleLightState.HighBeam
                self._ego_vehicle.set_light_state(
                    carla.VehicleLightState(light_mask)
                )
        except Exception:
            pass

        # Spawn Scenario-Specific Dynamic Actor (Stalled Car, Jaywalking Pedestrian, or ACC Lead Car)
        vehicle_bps = four_wheel_bps if len(four_wheel_bps) > 0 else list(bp_lib.filter("vehicle.*"))
        if (
            self._fsd_scenario_name
            in (
                "FSD-2_Dynamic_Overtake_Left",
                "FSD-5_MultiBlock_Overtake_Right",
                "FSD-7_BlindSpot_Blocked_Overtake",
                "FSD-8_Construction_Zone_Bypass_Right",
            )
            and len(self._route_waypoints) > 20
        ):
            obs_wp = self._route_waypoints[18]  # ~36m ahead in ego's initial lane
            obs_tf = carla.Transform(
                carla.Location(
                    x=obs_wp.transform.location.x,
                    y=obs_wp.transform.location.y,
                    z=obs_wp.transform.location.z + 0.35,
                ),
                obs_wp.transform.rotation,
            )
            obs_bp = vehicle_bps[0]
            self._scenario_actor = self._world.try_spawn_actor(obs_bp, obs_tf)
            if self._scenario_actor is not None:
                self._scenario_actor.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
                )
        elif (
            self._fsd_scenario_name
            in ("FSD-3_Pedestrian_Crosswalk_AEB", "FSD-9_Night_Headlight_Pedestrian_AEB")
            and len(self._route_waypoints) > 15
        ):
            ped_wp = self._route_waypoints[13]  # ~26m ahead on crosswalk
            ped_tf = carla.Transform(
                carla.Location(
                    x=ped_wp.transform.location.x,
                    y=ped_wp.transform.location.y,
                    z=ped_wp.transform.location.z + 0.85,
                ),
                carla.Rotation(yaw=ped_wp.transform.rotation.yaw - 90.0),
            )
            walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
            if walker_bps:
                self._scenario_actor = self._world.try_spawn_actor(walker_bps[0], ped_tf)
        elif (
            self._fsd_scenario_name
            in (
                "FSD-4_Urban_ACC_Follow",
                "FSD-6_StopAndGo_Lead_Braking",
                "FSD-10_HeavyRain_Cornering_ACC",
            )
            and len(self._route_waypoints) > 14
        ):
            lead_wp = self._route_waypoints[8]  # ~16m ahead in same lane for steady ACC tracking
            lead_tf = carla.Transform(
                carla.Location(
                    x=lead_wp.transform.location.x,
                    y=lead_wp.transform.location.y,
                    z=lead_wp.transform.location.z + 0.35,
                ),
                lead_wp.transform.rotation,
            )
            self._scenario_actor = self._world.try_spawn_actor(vehicle_bps[1 % len(vehicle_bps)], lead_tf)
            if self._scenario_actor is not None:
                self._lead_route_idx = 8
                self._fsd_metrics["acc_follow_episodes"] += 1

        # 1. Attach 392x196 Wide RGB Camera (for DINOv2 + Laya)
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

        # 2. Attach 960x540 HD 3rd-Person Chase Camera (for crisp HUD & MP4 recording)
        hd_bp = bp_lib.find("sensor.camera.rgb")
        hd_bp.set_attribute("image_size_x", "960")
        hd_bp.set_attribute("image_size_y", "540")
        hd_bp.set_attribute("fov", "90.0")
        hd_transform = carla.Transform(
            carla.Location(x=-6.2, z=2.6),
            carla.Rotation(pitch=-12.0),
        )
        self._hd_camera_sensor = self._world.spawn_actor(
            hd_bp, hd_transform, attach_to=self._ego_vehicle
        )

        def _on_hd_image(carla_img: Any) -> None:
            arr = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
            arr = arr.reshape((carla_img.height, carla_img.width, 4))
            rgb = arr[:, :, :3][:, :, ::-1].copy()
            self._hd_image_queue.put(rgb)

        self._hd_camera_sensor.listen(_on_hd_image)

        # Attach Collision & Lane-Invasion Sensors
        col_bp = bp_lib.find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self._ego_vehicle
        )
        self._collision_sensor.listen(
            lambda _event: setattr(self, "_collision_detected", True)
        )
        try:
            lane_bp = bp_lib.find("sensor.other.lane_invasion")
            self._lane_invasion_sensor = self._world.spawn_actor(
                lane_bp, carla.Transform(), attach_to=self._ego_vehicle
            )

            def _on_lane_invasion(event: Any) -> None:
                crossed = getattr(event, "crossed_lane_markings", None) or []
                if crossed:
                    illegal = any(
                        str(getattr(m, "type", "")).split(".")[-1].lower()
                        in ("solid", "solidsolid", "curb", "grass")
                        for m in crossed
                    )
                    if not illegal:
                        return
                self._lane_invasion_count += 1
                self._step_lane_invasion = True

            self._lane_invasion_sensor.listen(_on_lane_invasion)
        except Exception:
            self._lane_invasion_sensor = None

        # Attach 3 Native CARLA Millimeter-Wave Radars (Front + Left/Right Rear BSM)
        def _make_radar_callback(
            target_attr: str,
            mount_x: float,
            mount_y: float,
            mount_yaw_deg: float,
        ):
            yaw_rad = math.radians(mount_yaw_deg)

            def _on_radar(radar_data: Any) -> None:
                ego_spd = 0.0
                v_bx, v_by, v_bz = 0.0, 0.0, 0.0
                try:
                    if self._ego_vehicle is not None:
                        ev = self._ego_vehicle.get_velocity()
                        etf = self._ego_vehicle.get_transform()
                        fwd_v = etf.get_forward_vector()
                        rgt_v = etf.get_right_vector()
                        up_v = etf.get_up_vector()
                        v_bx = float(ev.x * fwd_v.x + ev.y * fwd_v.y + ev.z * fwd_v.z)
                        v_by = float(ev.x * rgt_v.x + ev.y * rgt_v.y + ev.z * rgt_v.z)
                        v_bz = float(ev.x * up_v.x + ev.y * up_v.y + ev.z * up_v.z)
                        ego_spd = math.sqrt(float(ev.x) ** 2 + float(ev.y) ** 2 + float(ev.z) ** 2)
                except Exception:
                    ego_spd = 0.0
                pts: List[Dict[str, Any]] = []
                for det in radar_data:
                    if det.altitude < -0.015 or det.altitude > 0.055 or det.depth < 1.5:
                        continue
                    cos_alt = math.cos(float(det.altitude))
                    sin_alt = math.sin(float(det.altitude))
                    az = float(det.azimuth)
                    x_s = float(det.depth) * cos_alt * math.cos(az)
                    y_s = float(det.depth) * cos_alt * math.sin(az)
                    x_body = mount_x + x_s * math.cos(yaw_rad) - y_s * math.sin(yaw_rad)
                    y_body = mount_y + x_s * math.sin(yaw_rad) + y_s * math.cos(yaw_rad)
                    ray_ang = yaw_rad + az
                    ux, uy = math.cos(ray_ang), math.sin(ray_ang)
                    r_vel = float(det.velocity)
                    v_ray_static = -(v_bx * cos_alt * ux + v_by * cos_alt * uy + v_bz * sin_alt)
                    # Exact 3D MTI Doppler clutter filter: reject static hills, tunnels, curbs & buildings
                    if abs(r_vel - v_ray_static) < 2.25:
                        continue
                    vx_rel = r_vel * ux
                    vy_rel = r_vel * uy
                    pts.append(
                        {
                            "x_m": round(x_body, 2),
                            "y_m": round(y_body, 2),
                            "vx_rel_mps": round(vx_rel, 2),
                            "vy_rel_mps": round(vy_rel, 2),
                            "speed_kmh": round(max(0.0, (ego_spd + vx_rel) * 3.6), 2),
                            "category": "vehicle",
                            "source": "carla_radar",
                        }
                    )
                pts.sort(key=lambda d: abs(d["x_m"]) + abs(d["y_m"]))
                with self._sensor_lock:
                    setattr(self, target_attr, pts[:24])

            return _on_radar

        for attr_name, buf_attr, loc_args, rot_yaw, hfov, vfov, rng_m in (
            ("_radar_front_sensor", "_latest_radar_front", (2.2, 0.0, 0.8), 0.0, "35", "10", "75"),
            ("_radar_bsm_left_sensor", "_latest_radar_left", (-1.8, -0.9, 0.8), -150.0, "70", "15", "25"),
            ("_radar_bsm_right_sensor", "_latest_radar_right", (-1.8, 0.9, 0.8), 150.0, "70", "15", "25"),
        ):
            try:
                r_bp = bp_lib.find("sensor.other.radar")
                r_bp.set_attribute("horizontal_fov", hfov)
                r_bp.set_attribute("vertical_fov", vfov)
                r_bp.set_attribute("points_per_second", "1500")
                r_bp.set_attribute("range", rng_m)
                r_tf = carla.Transform(
                    carla.Location(x=loc_args[0], y=loc_args[1], z=loc_args[2]),
                    carla.Rotation(pitch=1.5, yaw=rot_yaw),
                )
                r_actor = self._world.spawn_actor(r_bp, r_tf, attach_to=self._ego_vehicle)
                r_actor.listen(_make_radar_callback(buf_attr, loc_args[0], loc_args[1], rot_yaw))
                setattr(self, attr_name, r_actor)
            except Exception:
                setattr(self, attr_name, None)

        # Attach Native IMU & GNSS Sensors
        try:
            imu_bp = bp_lib.find("sensor.other.imu")
            self._imu_sensor = self._world.spawn_actor(
                imu_bp, carla.Transform(), attach_to=self._ego_vehicle
            )

            def _on_imu(imu_data: Any) -> None:
                with self._sensor_lock:
                    self._latest_imu = {
                        "long_accel_mps2": round(float(imu_data.accelerometer.x), 3),
                        "lat_accel_mps2": round(float(imu_data.accelerometer.y), 3),
                        "yaw_rate_rps": round(float(imu_data.gyroscope.z), 4),
                        "pitch_rad": round(float(imu_data.gyroscope.y) * 0.05, 4),
                        "compass_deg": round(math.degrees(float(imu_data.compass)), 2),
                    }

            self._imu_sensor.listen(_on_imu)
        except Exception:
            self._imu_sensor = None

        try:
            gnss_bp = bp_lib.find("sensor.other.gnss")
            self._gnss_sensor = self._world.spawn_actor(
                gnss_bp, carla.Transform(), attach_to=self._ego_vehicle
            )

            def _on_gnss(gnss_data: Any) -> None:
                with self._sensor_lock:
                    self._latest_gnss = {
                        "latitude": round(float(gnss_data.latitude), 6),
                        "longitude": round(float(gnss_data.longitude), 6),
                        "altitude": round(float(gnss_data.altitude), 2),
                    }

            self._gnss_sensor.listen(_on_gnss)
        except Exception:
            self._gnss_sensor = None

        # Sync PhysX Wheel Tire Friction with Active Weather Preset
        self._sync_physx_tire_friction(self._weather_preset)

        if self._traffic_manager is not None:
            try:
                self._traffic_manager.set_random_device_seed(int(seed))
            except Exception:
                pass

        # Spawn NPC traffic vehicles at least 28m away from ego route corridor
        vehicle_bps = four_wheel_bps if len(four_wheel_bps) > 0 else list(bp_lib.filter("vehicle.*"))
        spawned_npcs = 0
        route_locs = [w.transform.location for w in self._route_waypoints[:45]]
        for sp in all_sps:
            if spawned_npcs >= self.num_npc_vehicles:
                break
            too_close = any(
                math.hypot(float(sp.location.x - rl.x), float(sp.location.y - rl.y)) < 18.0
                for rl in route_locs
            )
            if too_close:
                continue
            npc_bp = vehicle_bps[int(rng.integers(0, len(vehicle_bps)))]
            npc = self._world.try_spawn_actor(npc_bp, sp)
            if npc is not None:
                npc.set_autopilot(True, self.tm_port)
                self._npc_actors.append(npc)
                spawned_npcs += 1

        # Initialize CARLA BehaviorAgent expert if available
        self._behavior_agent = None
        try:
            from agents.navigation.behavior_agent import BehaviorAgent  # type: ignore

            self._behavior_agent = BehaviorAgent(self._ego_vehicle, behavior="normal")
            dest_sp = all_sps[min(len(all_sps) - 1, 10)]
            self._behavior_agent.set_destination(dest_sp.location)
        except Exception:
            self._ego_vehicle.set_autopilot(False, self.tm_port)

        # Warm up 10 physics ticks so vehicle settles on suspension and engages 1st gear smoothly
        for _ in range(10):
            self._ego_vehicle.apply_control(
                carla.VehicleControl(throttle=0.45, steer=0.0, brake=0.0)
            )
            self._step_fsd4_lead_vehicle()
            self._world.tick()
        while self._image_queue.qsize() > 1:
            try:
                self._image_queue.get_nowait()
            except Exception:
                break
        self._update_spectator_chase_cam()
        loc = self._ego_vehicle.get_location()
        self._prev_location = (float(loc.x), float(loc.y), float(loc.z))
        self._progress = 0.0

    def _render_fsd_ar_overlay(
        self,
        ego_tr: Any,
        best_i: int,
        exp_man: str,
        traffic_light: str,
        hazard_source: str,
        lead_distance_m: float,
        speed_kmh: float,
    ) -> Optional[np.ndarray]:
        """Draw 3D AR Trajectory Carpet in CarlaUE4 and project FSD AR + BEV Radar onto HD frame."""
        carla = self._carla_mod
        pts = getattr(self, "_route_pts", [])
        if self._world is not None and len(pts) > best_i + 2:
            try:
                is_stop = traffic_light == "red" or hazard_source == "pedestrian"
                if is_stop:
                    col = carla.Color(255, 35, 35)
                elif exp_man in ("change_left", "change_right"):
                    col = carla.Color(255, 185, 20)
                else:
                    col = carla.Color(20, 235, 220)
                end_idx = min(len(pts) - 1, best_i + 15)
                for idx in range(best_i, end_idx):
                    p1 = pts[idx]
                    p2 = pts[idx + 1]
                    self._world.debug.draw_line(
                        carla.Location(p1[0], p1[1], p1[2] + 0.18),
                        carla.Location(p2[0], p2[1], p2[2] + 0.18),
                        thickness=0.14,
                        color=col,
                        life_time=0.07,
                    )
            except Exception:
                pass

        if self._last_hd_rgb is None:
            return None

        hd = self._last_hd_rgb.copy()
        h, w = hd.shape[:2]  # 540, 960
        ego_loc = ego_tr.location
        ego_yaw = math.radians(float(ego_tr.rotation.yaw))
        cos_y, sin_y = math.cos(ego_yaw), math.sin(ego_yaw)

        # Camera intrinsic & extrinsic (attached at x=-6.2, z=2.6, pitch=-12 deg, FOV=90 deg)
        focal = w / 2.0  # tan(45 deg) = 1.0
        cx, cy = w / 2.0, h / 2.0
        pitch_rad = math.radians(-12.0)
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)

        def _world_to_pixel(wx: float, wy: float, wz: float) -> Optional[Tuple[int, int]]:
            # Ego local coordinates (+X forward, +Y right, +Z up)
            dx = wx - float(ego_loc.x)
            dy = wy - float(ego_loc.y)
            dz = wz - float(ego_loc.z)
            ex = dx * cos_y + dy * sin_y
            ey = -dx * sin_y + dy * cos_y
            ez = dz
            # Relative to HD camera at (-6.2, 0.0, 2.6) pitched by -12 deg
            rx = ex - (-6.2)
            ry = ey
            rz = ez - 2.6
            cam_z = rx * cos_p + rz * sin_p
            cam_y_up = -rx * sin_p + rz * cos_p
            if cam_z <= 1.2:
                return None
            u = int(cx + (ry / cam_z) * focal)
            v_px = int(cy - (cam_y_up / cam_z) * focal)
            if -200 <= u <= w + 200 and -200 <= v_px <= h + 200:
                return (u, v_px)
            return None

        # 1. Project 3D FSD Trajectory Carpet (including Quintic S-Curve) onto 960x540 HD image
        is_stop = traffic_light == "red" or hazard_source == "pedestrian"
        if is_stop:
            ribbon_col = (255, 55, 55)
            edge_col = (255, 140, 140)
        elif exp_man in ("change_left", "change_right"):
            ribbon_col = (255, 190, 35)
            edge_col = (255, 235, 110)
        else:
            ribbon_col = (35, 225, 245)
            edge_col = (140, 255, 255)

        left_pts: List[Tuple[int, int]] = []
        right_pts: List[Tuple[int, int]] = []
        if len(pts) > best_i + 2:
            for idx in range(best_i + 1, min(len(pts), best_i + 18)):
                wx, wy, wz, w_yaw = pts[idx]
                rx_v, ry_v = -math.sin(w_yaw) * 1.15, math.cos(w_yaw) * 1.15
                pl = _world_to_pixel(wx - rx_v, wy - ry_v, wz + 0.08)
                pr = _world_to_pixel(wx + rx_v, wy + ry_v, wz + 0.08)
                if pl is not None and pr is not None:
                    left_pts.append(pl)
                    right_pts.append(pr)

        if len(left_pts) >= 2:
            poly = np.array(left_pts + right_pts[::-1], dtype=np.int32)
            overlay = hd.copy()
            cv2.fillPoly(overlay, [poly], ribbon_col)
            cv2.addWeighted(overlay, 0.34, hd, 0.66, 0, hd)
            cv2.polylines(hd, [np.array(left_pts, dtype=np.int32)], False, edge_col, 2, cv2.LINE_AA)
            cv2.polylines(hd, [np.array(right_pts, dtype=np.int32)], False, edge_col, 2, cv2.LINE_AA)

        # 2. Project 3D Target Hazard / Actor Bounding Box & Distance Label
        if self._scenario_actor is not None:
            try:
                a_loc = self._scenario_actor.get_location()
                a_vel = self._scenario_actor.get_velocity()
                a_kmh = 3.6 * math.hypot(a_vel.x, a_vel.y)
                d_act = math.hypot(float(a_loc.x - ego_loc.x), float(a_loc.y - ego_loc.y))
                p_bot = _world_to_pixel(float(a_loc.x), float(a_loc.y), float(a_loc.z))
                p_top = _world_to_pixel(float(a_loc.x), float(a_loc.y), float(a_loc.z + 1.8))
                if p_bot is not None and p_top is not None and d_act < 45.0:
                    bh = max(28, abs(p_bot[1] - p_top[1]))
                    bw = int(bh * (0.65 if "Pedestrian" in self._fsd_scenario_name else 1.35))
                    x1, y1 = p_top[0] - bw // 2, p_top[1]
                    x2, y2 = p_top[0] + bw // 2, p_bot[1]
                    b_col = (255, 50, 50) if is_stop else (255, 195, 30)
                    cv2.rectangle(hd, (x1, y1), (x2, y2), b_col, 2)
                    tag = (
                        f"PEDESTRIAN {d_act:.1f}m [AEB]"
                        if "Pedestrian" in self._fsd_scenario_name
                        else (
                            f"OBSTACLE {d_act:.1f}m [{exp_man.upper()}]"
                            if (
                                "Overtake" in self._fsd_scenario_name
                                or "Bypass" in self._fsd_scenario_name
                            )
                            else f"LEAD {d_act:.1f}m | {a_kmh:.0f}km/h [ACC]"
                        )
                    )
                    cv2.rectangle(hd, (x1, max(0, y1 - 24)), (x1 + len(tag) * 9 + 10, y1), b_col, -1)
                    cv2.putText(
                        hd,
                        tag,
                        (x1 + 5, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (15, 15, 20),
                        2,
                        cv2.LINE_AA,
                    )
            except Exception:
                pass

        # 3. Draw Top-Left Bird's-Eye-View (BEV) FSD Radar Mini-Map & Scenario Banner
        bev_w, bev_h = 150, 195
        cv2.rectangle(hd, (12, 12), (12 + bev_w, 12 + bev_h), (18, 22, 28), -1)
        cv2.rectangle(hd, (12, 12), (12 + bev_w, 12 + bev_h), (80, 185, 255), 1)
        ego_bx, ego_by = 12 + bev_w // 2, 12 + bev_h - 38
        # Draw BEV route waypoints (with Quintic S-curve)
        for idx in range(best_i, min(len(pts), best_i + 18)):
            wx, wy, _, _ = pts[idx]
            dx = float(wx - ego_loc.x)
            dy = float(wy - ego_loc.y)
            fx = dx * cos_y + dy * sin_y
            fy = -dx * sin_y + dy * cos_y
            px = int(ego_bx + fy * 4.2)
            py = int(ego_by - fx * 3.6)
            if 16 <= px <= 10 + bev_w and 16 <= py <= 10 + bev_h:
                cv2.circle(hd, (px, py), 3, ribbon_col, -1)
        # Draw 360-deg Left/Right Blind-Spot Radar Zones next to Ego Car
        bs_left = bool(getattr(self, "_blind_spot_left", False))
        bs_right = bool(getattr(self, "_blind_spot_right", False))
        bs_l_col = (50, 55, 255) if bs_left else (45, 95, 65)
        bs_r_col = (50, 55, 255) if bs_right else (45, 95, 65)
        cv2.rectangle(hd, (ego_bx - 24, ego_by - 10), (ego_bx - 11, ego_by + 14), bs_l_col, 1 if not bs_left else -1)
        cv2.rectangle(hd, (ego_bx + 11, ego_by - 10), (ego_bx + 24, ego_by + 14), bs_r_col, 1 if not bs_right else -1)
        # Draw Ego Car on BEV
        cv2.rectangle(hd, (ego_bx - 7, ego_by - 12), (ego_bx + 7, ego_by + 12), (60, 255, 120), -1)
        cv2.putText(
            hd,
            "FSD 360 BEV RADAR",
            (18, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )
        bs_status = f"BS-L:{'OCC' if bs_left else 'CLR'} BS-R:{'OCC' if bs_right else 'CLR'}"
        cv2.putText(
            hd,
            bs_status,
            (20, 12 + bev_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (165, 230, 185) if not (bs_left or bs_right) else (80, 90, 255),
            1,
            cv2.LINE_AA,
        )

        # Scenario Title & Weather Banner at top-center
        weather_name = str(getattr(self, "_weather_preset", "ClearNoon"))
        banner = f"{self._fsd_scenario_name} | {weather_name} | TL:{traffic_light.upper()}"
        cv2.rectangle(hd, (172, 12), (695, 44), (20, 25, 34), -1)
        cv2.rectangle(hd, (172, 12), (695, 44), ribbon_col, 1)
        cv2.putText(
            hd,
            banner,
            (182, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            ribbon_col,
            2,
            cv2.LINE_AA,
        )
        return hd

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

        try:
            while not self._hd_image_queue.empty():
                self._last_hd_rgb = self._hd_image_queue.get_nowait()
        except Exception:
            pass

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

        ego_tr = v.get_transform()
        ego_yaw = math.radians(float(ego_tr.rotation.yaw))

        # Continuous projection & curvature-adaptive lookahead along `self._route_pts`
        # Shortens lookahead on tight roundabouts/90-deg turns to prevent inner-curb chord cutting!
        pts_preview = getattr(self, "_route_pts", [])
        local_turn_mag = 0.0
        if len(pts_preview) >= 4:
            i_est = min(max(0, self._route_wp_idx), len(pts_preview) - 1)
            j_est = min(len(pts_preview) - 1, i_est + 5)
            local_turn_mag = abs(
                math.atan2(
                    math.sin(pts_preview[j_est][3] - pts_preview[i_est][3]),
                    math.cos(pts_preview[j_est][3] - pts_preview[i_est][3]),
                )
            )
        if local_turn_mag > 0.20:
            lookahead_m = float(np.clip(4.6 + 0.09 * speed_kmh, 4.4, 7.0))
        else:
            lookahead_m = float(np.clip(7.2 + 0.16 * speed_kmh, 7.0, 12.5))

        best_i, proj_x, proj_y, proj_yaw, _, _, ref_yaw = (
            self._sample_continuous_route_pose(
                float(loc.x),
                float(loc.y),
                self._route_wp_idx,
                lookahead_m=lookahead_m,
            )
        )
        self._route_wp_idx = best_i
        if getattr(self, "_route_waypoints", None):
            wp_curr = self._route_waypoints[
                min(best_i, len(self._route_waypoints) - 1)
            ]
        else:
            wp_curr = self._map.get_waypoint(
                loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )

        # Signed lateral offset from continuous projected point (+ = left of path, - = right)
        dx_wp = float(loc.x) - proj_x
        dy_wp = float(loc.y) - proj_y
        right_vec_x = -math.sin(proj_yaw)
        right_vec_y = math.cos(proj_yaw)
        lat_right = dx_wp * right_vec_x + dy_wp * right_vec_y
        lane_offset_m = float(
            np.clip(-lat_right, -self.lane_width * 0.85, self.lane_width * 0.85)
        )

        # Pure heading error relative to smooth path chord `ref_yaw` (+ = nose pointing left of path)
        heading_error = float(
            -math.atan2(math.sin(ego_yaw - ref_yaw), math.cos(ego_yaw - ref_yaw))
        )

        # Multi-horizon curvature directly from Ego's planned A* route `pts_preview`
        # (avoids arbitrary branch[0] selection of wp_curr.next() at roundabouts/junctions!)
        road_curvature = 0.0
        if len(pts_preview) >= 3:
            n_pt = len(pts_preview)
            c_vals: List[float] = []
            for step_off, weight in ((3, 1.15), (5, 1.00), (9, 0.78), (14, 0.56)):
                j_idx = min(n_pt - 1, best_i + step_off)
                if j_idx > best_i:
                    dyaw_j = -math.atan2(
                        math.sin(pts_preview[j_idx][3] - proj_yaw),
                        math.cos(pts_preview[j_idx][3] - proj_yaw),
                    )
                    c_vals.append(float(dyaw_j * weight))
            if c_vals:
                road_curvature = float(np.clip(max(c_vals, key=abs), -1.15, 1.15))
        else:
            next_wps = wp_curr.next(10.0)
            if next_wps:
                next_yaw = math.radians(float(next_wps[0].transform.rotation.yaw))
                road_curvature = float(
                    -math.atan2(math.sin(next_yaw - proj_yaw), math.cos(next_yaw - proj_yaw))
                )

        left_wp = wp_curr.get_left_lane()
        right_wp = wp_curr.get_right_lane()
        left_avail = bool(
            left_wp is not None
            and left_wp.lane_type == carla.LaneType.Driving
            and (left_wp.lane_id * wp_curr.lane_id > 0)
        )
        right_avail = bool(
            right_wp is not None
            and right_wp.lane_type == carla.LaneType.Driving
            and (right_wp.lane_id * wp_curr.lane_id > 0)
        )

        # Scan forward vehicles/pedestrians & 360-degree side/rear blind-spot zones + build radar_targets
        lead_distance_m = 60.0
        lead_speed_kmh = max(40.0, speed_kmh)
        pedestrian_distance_m = 100.0
        hazard_source = "none"
        lead_is_roadblock = False
        lead_is_stalled_in_lane = False
        roadblock_ids = {a.id for a in getattr(self, "_roadblock_actors", []) if a is not None}
        blind_spot_left = False
        blind_spot_right = False
        hx, hy = math.cos(ego_yaw), math.sin(ego_yaw)
        ego_vx_w, ego_vy_w = float(vel.x), float(vel.y)
        radar_targets: List[Dict[str, Any]] = []
        is_overtake_scenario = self._fsd_scenario_name in (
            "FSD-2_Dynamic_Overtake_Left",
            "FSD-5_MultiBlock_Overtake_Right",
            "FSD-7_BlindSpot_Blocked_Overtake",
            "FSD-8_Construction_Zone_Bypass_Right",
        )

        # Precompute cumulative arc-length along planned route for curvilinear Frenet projection
        pts_r = getattr(self, "_route_pts", [])
        route_s: List[float] = [0.0]
        for idx_r in range(1, len(pts_r)):
            route_s.append(
                route_s[-1]
                + math.hypot(
                    pts_r[idx_r][0] - pts_r[idx_r - 1][0],
                    pts_r[idx_r][1] - pts_r[idx_r - 1][1],
                )
            )
        s_ego_r = (
            route_s[min(best_i, len(route_s) - 1)]
            + (float(loc.x) - proj_x) * math.cos(proj_yaw)
            + (float(loc.y) - proj_y) * math.sin(proj_yaw)
            if pts_r
            else 0.0
        )

        for actor in self._world.get_actors():
            if actor.id == v.id:
                continue
            if (
                is_overtake_scenario
                and self._scenario_actor is not None
                and actor.id == self._scenario_actor.id
            ):
                # Handled smoothly by the Quintic S-Curve overtake state machine below
                continue
            type_id = str(getattr(actor, "type_id", ""))
            if not (type_id.startswith("vehicle.") or type_id.startswith("walker.pedestrian.")):
                continue
            a_loc = actor.get_location()
            rx = float(a_loc.x - loc.x)
            ry = float(a_loc.y - loc.y)
            dist = math.hypot(rx, ry)
            if dist > 62.0:
                continue
            fp = rx * hx + ry * hy
            lat_signed = -rx * hy + ry * hx  # + = right of ego, - = left of ego
            lp = abs(lat_signed)
            a_vel = actor.get_velocity()
            dvx = float(a_vel.x) - ego_vx_w
            dvy = float(a_vel.y) - ego_vy_w
            vx_rel = dvx * hx + dvy * hy
            vy_rel = -dvx * hy + dvy * hx
            a_spd_kmh = float(3.6 * math.sqrt(a_vel.x**2 + a_vel.y**2 + a_vel.z**2))
            is_ped = type_id.startswith("walker.pedestrian.")

            # Curvilinear Frenet projection onto Ego's planned route polyline
            frenet_x = float(fp)
            frenet_y = float(lat_signed)
            ref_h_act = ego_yaw
            if len(pts_r) >= 2:
                i_lo = max(0, best_i - 10)
                i_hi = min(len(pts_r), best_i + 35)
                if i_hi > i_lo:
                    ax_f, ay_f = float(a_loc.x), float(a_loc.y)
                    k_best = min(
                        range(i_lo, i_hi),
                        key=lambda k: (pts_r[k][0] - ax_f) ** 2 + (pts_r[k][1] - ay_f) ** 2,
                    )
                    pk_x, pk_y, _, pk_yaw = pts_r[k_best]
                    ref_h_act = float(pk_yaw)
                    dx_k = ax_f - pk_x
                    dy_k = ay_f - pk_y
                    frenet_x = float(
                        (route_s[k_best] + dx_k * math.cos(pk_yaw) + dy_k * math.sin(pk_yaw))
                        - s_ego_r
                    )
                    frenet_y = float(-dx_k * math.sin(pk_yaw) + dy_k * math.cos(pk_yaw))
                    if (
                        not is_ped
                        and a_spd_kmh < 1.5
                        and self._fsd_lane_changed
                        and self._route_wp_idx < self._lane_change_end_idx
                    ):
                        k_tgt = min(len(pts_r) - 1, self._lane_change_end_idx)
                        tx_k, ty_k, _, tyaw_k = pts_r[k_tgt]
                        tgt_lat = float(
                            -(ax_f - tx_k) * math.sin(tyaw_k) + (ay_f - ty_k) * math.cos(tyaw_k)
                        )
                        if abs(tgt_lat) > 2.0:
                            # Stationary obstacle sits in the abandoned lane; target S-curve lane is clear
                            frenet_y = tgt_lat

            a_yaw = math.radians(float(actor.get_transform().rotation.yaw))
            ext_x = 2.2
            ext_y = 0.95
            if not is_ped:
                try:
                    bb_ext = actor.bounding_box.extent
                    ext_x = max(1.5, float(bb_ext.x))
                    ext_y = max(0.8, float(bb_ext.y))
                except Exception:
                    pass
            tail_off = max(0.0, ext_x - 1.0)
            ac_x, ac_y = math.cos(a_yaw), math.sin(a_yaw)
            rear_wx = float(a_loc.x) - tail_off * ac_x
            rear_wy = float(a_loc.y) - tail_off * ac_y

            # Check rear-bumper Frenet clearance too (prevents accelerating into long bus/truck tail swing!)
            rear_frenet_x = frenet_x
            rear_frenet_y = frenet_y
            if not is_ped and tail_off > 1.4 and len(pts_r) >= 2:
                i_lo_r = max(0, best_i - 6)
                i_hi_r = min(len(pts_r), best_i + 30)
                if i_hi_r > i_lo_r:
                    kr = min(
                        range(i_lo_r, i_hi_r),
                        key=lambda k: (pts_r[k][0] - rear_wx) ** 2 + (pts_r[k][1] - rear_wy) ** 2,
                    )
                    pr_x, pr_y, _, pr_yaw = pts_r[kr]
                    rear_frenet_x = float(
                        (route_s[kr] + (rear_wx - pr_x) * math.cos(pr_yaw) + (rear_wy - pr_y) * math.sin(pr_yaw))
                        - s_ego_r
                    )
                    rear_frenet_y = float(-(rear_wx - pr_x) * math.sin(pr_yaw) + (rear_wy - pr_y) * math.cos(pr_yaw))

            heading_cos = math.cos(a_yaw - ref_h_act)
            crossing_conflict = False
            conf_dist = float(fp)
            phys_conf_dist = float(dist)
            r_clear = 2.45 + max(0.0, ext_y - 0.95)
            if not is_ped and heading_cos < 0.55:
                ospd = math.hypot(float(a_vel.x), float(a_vel.y))
                if len(pts_r) > best_i + 2:
                    v_ego_eff = max(speed_kmh / 3.6, 4.0)
                    if ospd <= 0.65:
                        # Stationary or creeping cross/roundabout vehicle (check center + rear/front overhangs)
                        front_wx = float(a_loc.x) + tail_off * ac_x
                        front_wy = float(a_loc.y) + tail_off * ac_y
                        for idx_p in range(best_i + 1, min(len(pts_r), best_i + 22)):
                            s_rel = max(0.5, route_s[idx_p] - s_ego_r)
                            d_obb = min(
                                math.hypot(pts_r[idx_p][0] - float(a_loc.x), pts_r[idx_p][1] - float(a_loc.y)),
                                math.hypot(pts_r[idx_p][0] - rear_wx, pts_r[idx_p][1] - rear_wy),
                                math.hypot(pts_r[idx_p][0] - front_wx, pts_r[idx_p][1] - front_wy),
                            )
                            if d_obb < r_clear + 0.20:
                                crossing_conflict = True
                                conf_dist = max(0.8, s_rel - 3.2)
                                phys_conf_dist = max(0.8, min(float(dist), s_rel - 1.5))
                                break
                    else:
                        act_wp = None
                        if dist < 42.0 and self._map is not None:
                            try:
                                cand_wp = self._map.get_waypoint(
                                    a_loc, project_to_road=True, lane_type=carla.LaneType.Driving
                                )
                                if (
                                    cand_wp is not None
                                    and math.cos(a_yaw - math.radians(float(cand_wp.transform.rotation.yaw))) > 0.70
                                ):
                                    act_wp = cand_wp
                            except Exception:
                                act_wp = None
                        for k_t in range(17):
                            t_obj = k_t * 0.25
                            ox = float(a_loc.x) + float(a_vel.x) * t_obj
                            oy = float(a_loc.y) + float(a_vel.y) * t_obj
                            if act_wp is not None and t_obj > 0.0:
                                try:
                                    nxt_a = act_wp.next(ospd * t_obj)
                                    if nxt_a:
                                        best_awp = min(
                                            nxt_a,
                                            key=lambda w_cand: (
                                                float(w_cand.transform.location.x) - ox
                                            ) ** 2
                                            + (float(w_cand.transform.location.y) - oy) ** 2,
                                        )
                                        ox = float(best_awp.transform.location.x)
                                        oy = float(best_awp.transform.location.y)
                                except Exception:
                                    pass
                            ox_rear = ox - tail_off * ac_x
                            oy_rear = oy - tail_off * ac_y
                            for idx_p in range(best_i + 1, min(len(pts_r), best_i + 24)):
                                s_rel = max(0.5, route_s[idx_p] - s_ego_r)
                                d_obb = min(
                                    math.hypot(pts_r[idx_p][0] - ox, pts_r[idx_p][1] - oy),
                                    math.hypot(pts_r[idx_p][0] - ox_rear, pts_r[idx_p][1] - oy_rear),
                                )
                                if d_obb < r_clear + 0.20:
                                    t_ego = s_rel / v_ego_eff
                                    if (t_obj - t_ego) <= 3.2 and (t_ego - t_obj) <= 2.8:
                                        crossing_conflict = True
                                        conf_dist = max(0.8, s_rel - 3.0)
                                        phys_conf_dist = max(1.5, min(float(dist), float(s_rel)))
                                        break
                            if crossing_conflict:
                                break
                if not crossing_conflict and heading_cos < -0.45:
                    continue
            non_conflicting_cross = bool(
                not is_ped
                and heading_cos < 0.55
                and not crossing_conflict
                and not (0.5 < rear_frenet_x < 24.0 and abs(rear_frenet_y) < 2.25)
            )

            eff_xm = conf_dist if crossing_conflict else float(fp)
            radar_targets.append(
                {
                    "x_m": round(eff_xm, 2),
                    "y_m": round(float(lat_signed), 2),
                    "frenet_x_m": round(frenet_x, 2),
                    "frenet_y_m": round(frenet_y, 2),
                    "vx_rel_mps": round(float(vx_rel), 2),
                    "vy_rel_mps": round(float(vy_rel), 2),
                    "speed_kmh": round(a_spd_kmh, 2),
                    "category": "pedestrian" if is_ped else "vehicle",
                    "crossing_conflict": crossing_conflict,
                    "non_conflicting_cross": non_conflicting_cross,
                    "source": "carla_3d",
                }
            )
            if (
                (-10.0 <= frenet_x <= 7.5 and 1.8 <= abs(frenet_y) <= 5.0)
                or (-18.0 <= frenet_x < -10.0 and 1.8 <= abs(frenet_y) <= 5.0 and vx_rel > 2.5)
            ) and not is_overtake_scenario:
                if frenet_y < 0.0:
                    blind_spot_left = True
                else:
                    blind_spot_right = True
            if is_ped and 0.5 < frenet_x < pedestrian_distance_m and abs(frenet_y) < 3.0:
                pedestrian_distance_m = float(frenet_x)
            if crossing_conflict and 1.5 < phys_conf_dist < lead_distance_m:
                lead_distance_m = float(phys_conf_dist)
                lead_speed_kmh = a_spd_kmh
                hazard_source = "vehicle"
                lead_is_roadblock = False
                lead_is_stalled_in_lane = False
            elif (
                not non_conflicting_cross
                and (
                    (0.5 < frenet_x < lead_distance_m and abs(frenet_y) < (2.6 if is_ped else 2.05))
                    or (not is_ped and 0.5 < rear_frenet_x < lead_distance_m and abs(rear_frenet_y) < 2.25)
                )
            ):
                eff_lead_fx = (
                    min(frenet_x, rear_frenet_x)
                    if (not is_ped and abs(rear_frenet_y) < 2.25 and rear_frenet_x > 0.5)
                    else frenet_x
                )
                lead_distance_m = float(eff_lead_fx)
                lead_speed_kmh = a_spd_kmh
                hazard_source = "pedestrian" if is_ped else "vehicle"
                lead_is_roadblock = bool(not is_ped and actor.id in roadblock_ids)
                lead_is_stalled_in_lane = bool(
                    not is_ped
                    and a_spd_kmh < 0.6
                    and not actor.is_at_traffic_light()
                    and not bool(getattr(wp_curr, "is_junction", False))
                )

        # Merge native CARLA millimeter-wave radar detections & snapshot IMU/GNSS
        with self._sensor_lock:
            native_radar_pts = (
                list(self._latest_radar_front[:8])
                + list(self._latest_radar_left[:6])
                + list(self._latest_radar_right[:6])
            )
            imu_snap = dict(self._latest_imu)
            gnss_snap = dict(self._latest_gnss)
        for pt in native_radar_pts:
            # Keep native CARLA radar detections without duplicating close actor centers
            # (radar sensor is at +2.2m and hits rear bumper ~2.4m closer than actor CoM -> up to 4.8m offset)
            dup = any(
                abs(pt["x_m"] - existing["x_m"]) < 5.2 and abs(pt["y_m"] - existing["y_m"]) < 2.0
                for existing in radar_targets
            )
            if not dup:
                pt_out = dict(pt)
                if len(pts_r) >= 2:
                    wx_p = float(loc.x) + float(pt["x_m"]) * hx - float(pt["y_m"]) * hy
                    wy_p = float(loc.y) + float(pt["x_m"]) * hy + float(pt["y_m"]) * hx
                    i_lo_p = max(0, best_i - 6)
                    i_hi_p = min(len(pts_r), best_i + 30)
                    if i_hi_p > i_lo_p:
                        kp = min(
                            range(i_lo_p, i_hi_p),
                            key=lambda k: (pts_r[k][0] - wx_p) ** 2 + (pts_r[k][1] - wy_p) ** 2,
                        )
                        pkx, pky, _, pkyaw = pts_r[kp]
                        pt_out["frenet_x_m"] = round(
                            float((route_s[kp] + (wx_p - pkx) * math.cos(pkyaw) + (wy_p - pky) * math.sin(pkyaw)) - s_ego_r),
                            2,
                        )
                        fy_p = float(-(wx_p - pkx) * math.sin(pkyaw) + (wy_p - pky) * math.cos(pkyaw))
                        if self._fsd_lane_changed and self._route_wp_idx < self._lane_change_end_idx:
                            kt = min(len(pts_r) - 1, self._lane_change_end_idx)
                            txk, tyk, _, tyawk = pts_r[kt]
                            t_lat_p = float(-(wx_p - txk) * math.sin(tyawk) + (wy_p - tyk) * math.cos(tyawk))
                            if abs(t_lat_p) > 2.0:
                                fy_p = t_lat_p
                        pt_out["frenet_y_m"] = round(fy_p, 2)
                radar_targets.append(pt_out)
            if (-10.0 <= pt["x_m"] <= 7.5 and 1.6 <= abs(pt["y_m"]) <= 5.0) or (
                -18.0 <= pt["x_m"] < -10.0 and 1.6 <= abs(pt["y_m"]) <= 5.0 and pt["vx_rel_mps"] > 2.5
            ):
                if pt["y_m"] < 0.0:
                    blind_spot_left = True
                else:
                    blind_spot_right = True

        self._blind_spot_left = blind_spot_left
        self._blind_spot_right = blind_spot_right

        # Continuous ACC radar lock onto lead vehicle (FSD-4, FSD-6, FSD-10)
        if (
            self._fsd_scenario_name
            in (
                "FSD-4_Urban_ACC_Follow",
                "FSD-6_StopAndGo_Lead_Braking",
                "FSD-10_HeavyRain_Cornering_ACC",
            )
            and self._scenario_actor is not None
        ):
            try:
                a_loc = self._scenario_actor.get_location()
                a_vel = self._scenario_actor.get_velocity()
                d_lead = math.hypot(float(a_loc.x - loc.x), float(a_loc.y - loc.y))
                v_lead_kmh = float(3.6 * math.hypot(a_vel.x, a_vel.y))
                lead_distance_m = min(lead_distance_m, d_lead)
                lead_speed_kmh = max(16.0, v_lead_kmh)
                hazard_source = "vehicle"
            except Exception:
                pass

        # Traffic light state check (with multi-intersection hold reset & cross-group red interlock!)
        traffic_light = "green"
        if v.is_at_traffic_light():
            tl = v.get_traffic_light()
            if tl is not None:
                tl_state = tl.get_state()
                if tl_state == carla.TrafficLightState.Red:
                    if (
                        self._fsd_scenario_name
                        in (
                            "FSD-4_Urban_ACC_Follow",
                            "FSD-6_StopAndGo_Lead_Braking",
                            "FSD-10_HeavyRain_Cornering_ACC",
                        )
                        or self._fsd_stop_hold_steps >= 14
                    ):
                        try:
                            for grp_tl in tl.get_group_traffic_lights():
                                if grp_tl.id != tl.id:
                                    grp_tl.set_state(carla.TrafficLightState.Red)
                            tl.set_state(carla.TrafficLightState.Green)
                        except Exception:
                            pass
                        traffic_light = "green"
                        self._fsd_stop_hold_steps = 0
                    else:
                        traffic_light = "red"
                        hazard_source = "red_light"
                        if not bool(getattr(self, "_external_scenario_control", False)):
                            lead_distance_m = min(lead_distance_m, 8.0)
                            lead_speed_kmh = 0.0
                        self._fsd_stop_hold_steps += 1
                elif tl_state == carla.TrafficLightState.Yellow:
                    traffic_light = "yellow"
                else:
                    self._fsd_stop_hold_steps = 0
        else:
            self._fsd_stop_hold_steps = 0

        nav_command = "go_straight"
        if getattr(self, "_route_options", None):
            upcoming_opts = self._route_options[
                max(0, best_i) : min(len(self._route_options), best_i + 15)
            ]
            for opt_s in upcoming_opts:
                if opt_s == "left":
                    nav_command = "turn_left"
                    break
                if opt_s == "right":
                    nav_command = "turn_right"
                    break
                if opt_s in ("changelaneleft", "changelaneright"):
                    nav_command = "merge"
                    break
            if (
                nav_command == "go_straight"
                and "straight" not in upcoming_opts
                and abs(road_curvature) > 0.52
            ):
                nav_command = "turn_left" if road_curvature > 0 else "turn_right"
        elif abs(road_curvature) > 0.28:
            nav_command = "turn_left" if road_curvature > 0 else "turn_right"

        exp_man = "follow_lead" if lead_distance_m <= 32.0 else "keep_lane"
        ext_active = bool(getattr(self, "_external_scenario_control", False))

        # Release previous lane-change lock once Ego completes the 28m S-curve blend
        if (ext_active or self._navigation_mode == "point_to_point") and self._fsd_lane_changed:
            if self._route_wp_idx >= self._lane_change_end_idx:
                self._fsd_lane_changed = False

        # 1. Off-route recovery: re-plan A* from live pose in Point-to-Point mode if Ego deviated > 4.5m
        if (
            self._navigation_mode == "point_to_point"
            and self._dest_location is not None
            and abs(lat_right) > 4.5
            and (self._step_count - self._last_reroute_step) >= 25
        ):
            replan_res = self.replan_from_current_pose(reason="off_route_recovery")
            if replan_res.get("rerouted"):
                best_i, proj_x, proj_y, proj_yaw, _, _, ref_yaw = self._sample_continuous_route_pose(
                    float(loc.x), float(loc.y), 0, lookahead_m=lookahead_m
                )
                self._route_wp_idx = best_i
                dx_wp = float(loc.x) - proj_x
                dy_wp = float(loc.y) - proj_y
                lat_right = dx_wp * (-math.sin(proj_yaw)) + dy_wp * math.cos(proj_yaw)
                lane_offset_m = float(
                    np.clip(-lat_right, -self.lane_width * 0.85, self.lane_width * 0.85)
                )
                heading_error = float(
                    -math.atan2(math.sin(ego_yaw - ref_yaw), math.cos(ego_yaw - ref_yaw))
                )

        # 2. Blocked-corridor roadblock detection -> Quintic S-curve bypass + A* splice to destination
        if (
            (ext_active or self._navigation_mode == "point_to_point")
            and not self._fsd_lane_changed
            and 0.5 < lead_distance_m < 38.0
            and lead_speed_kmh < 1.5
            and traffic_light == "green"
            and hazard_source == "vehicle"
            and (lead_is_roadblock or (lead_is_stalled_in_lane and abs(road_curvature) < 0.16))
        ):
            self._blocked_hold_steps += 1
            req_hold = 4 if lead_is_roadblock else 30
            if self._blocked_hold_steps >= req_hold:
                b_steps = max(7, min(15, int((lead_distance_m - 2.0) / 2.0)))
                if left_avail and not blind_spot_left and self._switch_route_to_adjacent_lane("left", blend_steps=b_steps):
                    exp_man = "change_left"
                    lead_distance_m = 28.0
                    lead_speed_kmh = 28.0
                    self._blocked_hold_steps = 0
                elif right_avail and not blind_spot_right and self._switch_route_to_adjacent_lane("right", blend_steps=b_steps):
                    exp_man = "change_right"
                    lead_distance_m = 28.0
                    lead_speed_kmh = 28.0
                    self._blocked_hold_steps = 0
        else:
            self._blocked_hold_steps = 0

        if (
            ext_active
            and not self._fsd_lane_changed
            and getattr(self, "_last_gated_maneuver", "keep_lane") in ("change_left", "change_right")
            and abs(road_curvature) < 0.20
        ):
            self._switch_route_to_adjacent_lane(
                "left" if getattr(self, "_last_gated_maneuver", "") == "change_left" else "right"
            )

        # Execute FSD Scenario-Specific State Machine (FSD-1 through FSD-10)
        if self._fsd_scenario_name == "FSD-1_Intersection_Turn" and not self._fsd_event_resolved:
            if 14.0 <= self._progress <= 25.0:
                if speed_kmh < 2.5 or self._fsd_stop_hold_steps >= 10:
                    self._fsd_event_resolved = True
                    self._fsd_metrics["red_light_stops"] += 1
                    traffic_light = "green"
                    hazard_source = "none"
                    lead_distance_m = 60.0
                else:
                    traffic_light = "red"
                    hazard_source = "red_light"
                    lead_distance_m = 8.0
                    lead_speed_kmh = 0.0
                    self._fsd_stop_hold_steps += 1
        elif (
            self._fsd_scenario_name == "FSD-6_StopAndGo_Lead_Braking"
            and not self._fsd_event_resolved
        ):
            if 14.0 <= self._progress <= 24.0:
                if speed_kmh < 3.5 or self._fsd_stop_hold_steps >= 9:
                    self._fsd_event_resolved = True
                    self._fsd_metrics["stop_and_go_recoveries"] += 1
                    hazard_source = "vehicle"
                    lead_distance_m = max(14.5, lead_distance_m)
                    lead_speed_kmh = 24.0
                else:
                    hazard_source = "vehicle"
                    lead_distance_m = min(lead_distance_m, 9.5)
                    lead_speed_kmh = 0.0
                    self._fsd_stop_hold_steps += 1
        elif (
            self._fsd_scenario_name == "FSD-2_Dynamic_Overtake_Left"
            and self._scenario_actor is not None
            and not self._fsd_event_resolved
        ):
            obs_loc = self._scenario_actor.get_location()
            d_obs = math.hypot(float(obs_loc.x - loc.x), float(obs_loc.y - loc.y))
            if d_obs <= 34.0:
                exp_man = "change_left"
                nav_command = "turn_left"
                if (
                    not self._fsd_lane_changed
                    and getattr(self, "_last_gated_maneuver", "keep_lane") == "change_left"
                ):
                    self._switch_route_to_adjacent_lane("left")
                if not self._fsd_lane_changed and d_obs <= 18.0:
                    lead_distance_m = d_obs
                    lead_speed_kmh = 0.0
                else:
                    lead_distance_m = 26.0
                    lead_speed_kmh = 26.0
                if self._fsd_lane_changed and self._route_wp_idx >= self._lane_change_end_idx:
                    self._fsd_event_resolved = True
        elif (
            self._fsd_scenario_name == "FSD-7_BlindSpot_Blocked_Overtake"
            and self._scenario_actor is not None
            and not self._fsd_event_resolved
        ):
            obs_loc = self._scenario_actor.get_location()
            d_obs = math.hypot(float(obs_loc.x - loc.x), float(obs_loc.y - loc.y))
            if d_obs <= 35.5:
                exp_man = "change_left"
                nav_command = "turn_left"
                if self._fsd_stop_hold_steps < 5 and not self._fsd_lane_changed:
                    # Adjacent vehicle occupies left blind-spot zone for 5 frames before clearing
                    blind_spot_left = True
                    self._blind_spot_left = True
                    self._fsd_stop_hold_steps += 1
                    if self._fsd_stop_hold_steps == 1:
                        self._fsd_metrics["blindspot_interlocks"] += 1
                else:
                    blind_spot_left = False
                    self._blind_spot_left = False
                    if (
                        not self._fsd_lane_changed
                        and getattr(self, "_last_gated_maneuver", "keep_lane") == "change_left"
                    ):
                        self._switch_route_to_adjacent_lane("left")
                    if self._fsd_lane_changed and self._route_wp_idx >= self._lane_change_end_idx:
                        self._fsd_event_resolved = True
                if not self._fsd_lane_changed and d_obs <= 18.0:
                    lead_distance_m = d_obs
                    lead_speed_kmh = 0.0
                else:
                    lead_distance_m = 26.0
                    lead_speed_kmh = 26.0
        elif (
            self._fsd_scenario_name
            in (
                "FSD-5_MultiBlock_Overtake_Right",
                "FSD-8_Construction_Zone_Bypass_Right",
            )
            and self._scenario_actor is not None
            and not self._fsd_event_resolved
        ):
            obs_loc = self._scenario_actor.get_location()
            d_obs = math.hypot(float(obs_loc.x - loc.x), float(obs_loc.y - loc.y))
            if d_obs <= 34.0:
                exp_man = "change_right"
                nav_command = "turn_right"
                if self._fsd_scenario_name == "FSD-8_Construction_Zone_Bypass_Right":
                    traffic_light = "yellow"
                if (
                    not self._fsd_lane_changed
                    and getattr(self, "_last_gated_maneuver", "keep_lane") == "change_right"
                ):
                    self._switch_route_to_adjacent_lane("right")
                if not self._fsd_lane_changed and d_obs <= 18.0:
                    lead_distance_m = d_obs
                    lead_speed_kmh = 0.0
                else:
                    lead_distance_m = 26.0
                    lead_speed_kmh = 26.0
                if self._fsd_lane_changed and self._route_wp_idx >= self._lane_change_end_idx:
                    self._fsd_event_resolved = True
        elif (
            self._fsd_scenario_name
            in ("FSD-3_Pedestrian_Crosswalk_AEB", "FSD-9_Night_Headlight_Pedestrian_AEB")
            and self._scenario_actor is not None
            and not self._fsd_event_resolved
        ):
            ped_loc = self._scenario_actor.get_location()
            d_ped = math.hypot(float(ped_loc.x - loc.x), float(ped_loc.y - loc.y))
            if d_ped <= 16.5:
                if speed_kmh < 2.8 or self._fsd_stop_hold_steps >= 10:
                    self._fsd_event_resolved = True
                    self._fsd_metrics["pedestrian_aeb_stops"] += 1
                    try:
                        self._scenario_actor.destroy()
                    except Exception:
                        pass
                    self._scenario_actor = None
                    hazard_source = "none"
                    lead_distance_m = 60.0
                else:
                    hazard_source = "pedestrian"
                    lead_distance_m = min(d_ped, 10.5)
                    lead_speed_kmh = 0.0
                    self._fsd_stop_hold_steps += 1

        # Render 3D AR Trajectory Carpet & 3D Bounding Boxes in CarlaUE4 and onto HD camera frame
        hd_ar_frame = self._render_fsd_ar_overlay(
            ego_tr=ego_tr,
            best_i=self._route_wp_idx,
            exp_man=exp_man,
            traffic_light=traffic_light,
            hazard_source=hazard_source,
            lead_distance_m=lead_distance_m,
            speed_kmh=speed_kmh,
        )

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

        on_driving_road = True
        if abs(lat_right) > self.lane_width * 1.85 and self._map is not None:
            try:
                chk_wp = self._map.get_waypoint(
                    loc, project_to_road=False, lane_type=carla.LaneType.Driving
                )
                on_driving_road = chk_wp is not None
            except Exception:
                on_driving_road = False
        out_of_road = bool(abs(lat_right) > self.lane_width * 1.85 and not on_driving_road)
        if out_of_road:
            self._fsd_metrics["off_road_infractions"] += 1
        if self._collision_detected:
            self._fsd_metrics["vehicle_collisions"] += 1
        crash = bool(self._collision_detected or out_of_road)
        if self._navigation_mode == "point_to_point" and self._dest_location is not None:
            dist_to_dest = math.hypot(
                float(loc.x) - float(self._dest_location[0]),
                float(loc.y) - float(self._dest_location[1]),
            )
            arrive_dest = bool(
                (dist_to_dest <= 8.5 or (len(pts_r) >= 6 and best_i >= len(pts_r) - 2))
                and not crash
            )
        else:
            arrive_dest = bool(self._progress >= self._route_length_m and not crash)

        if self._fsd_lane_changed and self._route_wp_idx < self._lane_change_end_idx:
            # S-curve transition is already embedded in `_route_pts`; prevent double lateral step offset
            left_avail = False
            right_avail = False

        weather_str = str(getattr(self, "_weather_preset", "ClearNoon"))
        is_wet = any(w in weather_str for w in ("Wet", "Rain", "HardRain", "SoftRain", "MidRainy"))
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
            expert_maneuver=exp_man,
            traffic_light=traffic_light,
            hazard_source=hazard_source,
            blind_spot_left=blind_spot_left,
            blind_spot_right=blind_spot_right,
            wet_road=is_wet,
            a_max_mps2=4.5 if is_wet else 6.0,
            uncertainty_margin_m=1.8 if is_wet else 0.0,
        )
        route_comp = float(
            np.clip(self._progress / max(1.0, self._route_length_m), 0.0, 1.0)
        )
        infraction_penalty = (
            (0.50 ** self._fsd_metrics["pedestrian_collisions"])
            * (0.60 ** (1 if crash else 0))
            * (0.70 ** self._fsd_metrics["red_light_infractions"])
        )
        step_li = bool(self._step_lane_invasion)
        self._step_lane_invasion = False
        extra_info = {
            "step": self._step_count,
            "seed": self._current_seed,
            "simulator": "carla",
            "scenario_event": "real_carla",
            "fsd_scenario": self._fsd_scenario_name,
            "weather_preset": str(getattr(self, "_weather_preset", "ClearNoon")),
            "expert_maneuver": exp_man,
            "hd_rgb": hd_ar_frame,
            "lead_distance_m": round(lead_distance_m, 2),
            "lead_speed_kmh": round(lead_speed_kmh, 2),
            "pedestrian_distance_m": round(pedestrian_distance_m, 2),
            "road_curvature": round(road_curvature, 4),
            "traffic_light": traffic_light,
            "hazard_source": hazard_source,
            "blind_spot_left": blind_spot_left,
            "blind_spot_right": blind_spot_right,
            "radar_targets": radar_targets,
            "imu_telemetry": imu_snap,
            "gnss_telemetry": gnss_snap,
            "lane_invasions": int(self._lane_invasion_count),
            "lane_invasion": step_li,
            "wet_road": is_wet,
            "route_completion": route_comp,
            "driving_score": round(route_comp * infraction_penalty * 100.0, 2),
            "fsd_metrics": dict(self._fsd_metrics),
            "nav_state": self._build_real_nav_state(ego_tr, best_i),
            "reroute_count": int(self._reroute_count),
            "arrive_dest": arrive_dest,
            "crash": crash,
            "out_of_road": out_of_road,
        }
        self._last_real_lead_d = float(lead_distance_m)
        return rgb, telemetry, extra_info

    def _sync_physx_tire_friction(self, weather_name: str) -> None:
        """Synchronize CARLA VehiclePhysicsControl wheel tire_friction with weather surface wetness."""
        if self._ego_vehicle is None:
            return
        w_low = str(weather_name or "").lower()
        if "hardrain" in w_low or "heavy" in w_low:
            mu = 1.65
        elif any(k in w_low for k in ("wet", "rain")):
            mu = 2.25
        else:
            mu = 3.50
        try:
            phys = self._ego_vehicle.get_physics_control()
            wheels = []
            for w in phys.wheels:
                w.tire_friction = mu
                wheels.append(w)
            phys.wheels = wheels
            self._ego_vehicle.apply_physics_control(phys)
        except Exception:
            pass

    def set_weather(self, weather: str, precipitation: float = 0.0) -> None:
        """Dynamically change weather preset and PhysX road-tire friction."""
        w_map = {
            "clear": "ClearNoon",
            "wet": "WetNoon",
            "heavy_rain": "HardRainNoon",
            "night_rain": "NightRain",
        }
        preset = w_map.get(str(weather or "clear").strip().lower(), str(weather or "ClearNoon"))
        self._weather_preset = preset
        if self._world is not None and self._carla_mod is not None:
            carla = self._carla_mod
            try:
                wp = getattr(carla.WeatherParameters, preset, None)
                if wp is not None:
                    self._world.set_weather(wp)
                elif "Night" in preset:
                    is_rain = "Rain" in preset or precipitation > 0.2
                    self._world.set_weather(
                        carla.WeatherParameters(
                            cloudiness=85.0 if is_rain else 15.0,
                            precipitation=75.0 if is_rain else float(precipitation) * 100.0,
                            precipitation_deposits=85.0 if is_rain else 0.0,
                            wetness=90.0 if is_rain else 0.0,
                            sun_altitude_angle=-25.0,
                        )
                    )
            except Exception:
                pass
            self._sync_physx_tire_friction(preset)

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
        scenario_idx = self._current_seed % len(FSD_SCENARIO_NAMES)
        ep_offset = self._current_seed - self.start_seed
        weather_shift = (ep_offset // len(FSD_SCENARIO_NAMES)) * 2
        self._fsd_scenario_name = FSD_SCENARIO_NAMES[scenario_idx]
        self._weather_preset = FSD_WEATHER_PRESETS[
            (scenario_idx + weather_shift) % len(FSD_WEATHER_PRESETS)
        ]
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
        self,
        action: Tuple[float, float] | np.ndarray,
        gated_maneuver: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute `[steering, throttle_brake]` in CARLA (via `VehicleControl`) and return next state."""
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))
        if gated_maneuver is not None:
            self._last_gated_maneuver = str(gated_maneuver)
        self._step_count += 1

        if not self.use_mock and self._world is not None and self._ego_vehicle is not None:
            carla = self._carla_mod
            prev_st = float(getattr(self, "_carla_steer_prev", 0.0))
            st_delta = float(np.clip(steering - prev_st, -0.055, 0.055))
            smooth_steering = float(np.clip(prev_st + st_delta, -1.0, 1.0))
            self._carla_steer_prev = smooth_steering
            # Normalize CARLA Tesla Model 3 dual-motor torque (~6.2 m/s^2 at 1.0) to passenger
            # sedan ~3.2 m/s^2 comfort envelope while keeping 100% unattenuated braking authority.
            cur_spd_mps = float(self._ego_vehicle.get_velocity().length())
            if throttle_brake >= 0.0:
                eff_tb = float(min(0.58, throttle_brake * 0.58))
            elif throttle_brake > -0.02 and cur_spd_mps > 1.5:
                eff_tb = 0.0
            else:
                eff_tb = float(throttle_brake)
            ctrl_dict = self.to_carla_vehicle_control(smooth_steering, eff_tb)
            ctrl = carla.VehicleControl(
                steer=ctrl_dict["steer"],
                throttle=ctrl_dict["throttle"],
                brake=ctrl_dict["brake"],
                hand_brake=ctrl_dict["hand_brake"],
                reverse=ctrl_dict["reverse"],
            )
            self._ego_vehicle.apply_control(ctrl)
            self._step_fsd4_lead_vehicle()
            self._world.tick()
            self._update_spectator_chase_cam()
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
        weather_str = str(getattr(self, "_weather_preset", "ClearNoon"))
        is_wet = any(w in weather_str for w in ("Wet", "Rain", "HardRain", "SoftRain", "MidRainy"))
        ped_d = self._lead_distance_m if self._hazard_source == "pedestrian" else 100.0
        return TelemetryState(
            speed_kmh=self._speed_kmh,
            lane_offset_m=self._lane_offset_m,
            heading_error=self._heading_error,
            left_lane_available=bool(self._current_lane_idx > 0),
            right_lane_available=bool(self._current_lane_idx < self._num_lanes - 1),
            nav_command=self._nav_command,
            lead_distance_m=self._lead_distance_m,
            lead_speed_kmh=self._lead_speed_kmh,
            pedestrian_distance_m=ped_d,
            road_curvature=self._road_curvature,
            expert_maneuver=exp_man,
            traffic_light=self._traffic_light,
            hazard_source=self._hazard_source,
            wet_road=is_wet,
            a_max_mps2=4.5 if is_wet else 6.0,
            uncertainty_margin_m=1.8 if is_wet else 0.0,
        )

    def _build_mock_info(
        self, arrive_dest: bool, crash: bool, out_of_road: bool
    ) -> Dict[str, Any]:
        exp_man = (
            self._scenario_event
            if self._scenario_event in ("change_left", "change_right")
            else ("follow_lead" if self._lead_distance_m <= 30.0 else "keep_lane")
        )
        radar_targets: List[Dict[str, Any]] = []
        if self._lead_distance_m < 58.0 and self._hazard_source != "red_light":
            radar_targets.append(
                {
                    "x_m": round(float(self._lead_distance_m), 2),
                    "y_m": 0.0,
                    "vx_rel_mps": round((self._lead_speed_kmh - self._speed_kmh) / 3.6, 2),
                    "vy_rel_mps": 0.0,
                    "speed_kmh": round(float(self._lead_speed_kmh), 2),
                    "category": "pedestrian" if self._hazard_source == "pedestrian" else "vehicle",
                    "source": "mock_radar",
                }
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
            "radar_targets": radar_targets,
            "imu_telemetry": dict(self._latest_imu),
            "gnss_telemetry": dict(self._latest_gnss),
            "lane_invasions": 0,
            "lane_invasion": False,
            "carla_control_preview": self.to_carla_vehicle_control(0.0, 0.5),
            "route_completion": float(
                np.clip(self._progress / max(1.0, self._route_length_m), 0.0, 1.0)
            ),
            "nav_state": {
                "map_name": f"{self.town}-Mock",
                "navigation_mode": self._navigation_mode,
                "ego_xy_yaw": [round(float(self._progress), 1), round(float(self._lane_offset_m), 2), 0.0],
                "dest_xy": [
                    round(float(self._dest_location[0]) if self._dest_location else float(self._route_length_m), 1),
                    round(float(self._dest_location[1]) if self._dest_location else 0.0, 1),
                ],
                "dest_index": self._dest_spawn_idx,
                "active_route_xy": [
                    [round(float(s), 1), 0.0]
                    for s in np.linspace(self._progress, max(self._progress + 10.0, self._route_length_m), 15)
                ],
                "spawn_nodes": self.list_spawn_points(),
                "remaining_route_m": round(max(0.0, self._route_length_m - self._progress), 1),
                "reroute_count": int(self._reroute_count),
            },
            "reroute_count": int(self._reroute_count),
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
            try:
                for s in (
                    self._camera_sensor,
                    self._hd_camera_sensor,
                    self._collision_sensor,
                    self._lane_invasion_sensor,
                    self._radar_front_sensor,
                    self._radar_bsm_left_sensor,
                    self._radar_bsm_right_sensor,
                    self._imu_sensor,
                    self._gnss_sensor,
                ):
                    if s is not None:
                        s.stop()
            except Exception:
                pass
            try:
                for actor in self._world.get_actors():
                    t_id = str(getattr(actor, "type_id", ""))
                    if t_id.startswith("vehicle.") or t_id.startswith("walker.pedestrian."):
                        try:
                            actor.destroy()
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                if self._traffic_manager is not None:
                    self._traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
            try:
                settings = self._world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self._world.apply_settings(settings)
            except Exception:
                pass
            self._world = None
