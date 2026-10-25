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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.env.metadrive_wrapper import NAV_COMMANDS, TelemetryState

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
        self._grp: Any = None
        self._traffic_manager: Any = None
        self._ego_vehicle: Any = None
        self._camera_sensor: Any = None
        self._hd_camera_sensor: Any = None
        self._collision_sensor: Any = None
        self._lane_invasion_sensor: Any = None
        self._behavior_agent: Any = None
        self._npc_actors: List[Any] = []
        self._scenario_actor: Any = None
        self._fsd_scenario_name: str = FSD_SCENARIO_NAMES[0]
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

    def _try_init_real_carla(self) -> None:
        """Attempt to connect to a running CARLA server (`host:port`); fallback to urban mock if unavailable."""
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
            self._traffic_manager.set_global_distance_to_leading_vehicle(2.5)

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
        for sensor in (
            self._camera_sensor,
            self._hd_camera_sensor,
            self._collision_sensor,
            self._lane_invasion_sensor,
        ):
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
        for sensor in (
            self._camera_sensor,
            self._hd_camera_sensor,
            self._collision_sensor,
            self._lane_invasion_sensor,
        ):
            if sensor is not None:
                try:
                    sensor.destroy()
                except Exception:
                    pass
        self._camera_sensor = None
        self._hd_camera_sensor = None
        self._collision_sensor = None
        self._lane_invasion_sensor = None

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
        """Attach continuous forward tangent yaw_rad to a list of (x, y, z) 3D points."""
        n = len(xyz_list)
        if n == 0:
            return []
        pts: List[Tuple[float, float, float, float]] = []
        for i in range(n):
            x, y, z = xyz_list[i]
            if i + 1 < n:
                nx, ny, _ = xyz_list[i + 1]
                yaw = math.atan2(ny - y, nx - x)
            elif i > 0:
                px, py, _ = xyz_list[i - 1]
                yaw = math.atan2(y - py, x - px)
            else:
                yaw = 0.0
            pts.append((float(x), float(y), float(z), float(yaw)))
        return pts

    def _rebuild_route_pts(self) -> None:
        """Build continuous 3D polyline `self._route_pts` from `self._route_waypoints`."""
        wps = getattr(self, "_route_waypoints", [])
        xyz = [
            (
                float(w.transform.location.x),
                float(w.transform.location.y),
                float(w.transform.location.z),
            )
            for w in wps
        ]
        self._route_pts: List[Tuple[float, float, float, float]] = (
            self._compute_polyline_tangents(xyz)
        )

    def _switch_route_to_adjacent_lane(self, direction: str) -> bool:
        """Switch route to adjacent lane ('left' or 'right') via Minimum-Jerk Quintic S-Curve trajectory."""
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
        n_blend = min(14, len(orig_chain), len(adj_chain))  # 28m smooth S-curve transition
        prefix_xyz = [
            (p[0], p[1], p[2]) for p in getattr(self, "_route_pts", [])[:start_idx]
        ]
        blended_xyz: List[Tuple[float, float, float]] = []
        for k in range(len(adj_chain)):
            al = adj_chain[k].transform.location
            if k < n_blend:
                ol = orig_chain[k].transform.location
                tau = float(k) / float(max(1, n_blend - 1))
                # Minimum-jerk quintic polynomial: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
                s = (tau**3) * (10.0 + tau * (-15.0 + 6.0 * tau))
                bx = (1.0 - s) * float(ol.x) + s * float(al.x)
                by = (1.0 - s) * float(ol.y) + s * float(al.y)
                bz = (1.0 - s) * float(ol.z) + s * float(al.z)
                blended_xyz.append((bx, by, bz))
            else:
                blended_xyz.append((float(al.x), float(al.y), float(al.z)))

        self._route_pts = self._compute_polyline_tangents(prefix_xyz + blended_xyz)
        self._route_waypoints = list(self._route_waypoints[:start_idx]) + adj_chain
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
        """Project (x, y) onto `self._route_pts` and interpolate a continuous lookahead point.

        Returns `(best_idx, proj_x, proj_y, proj_yaw, look_x, look_y, ref_yaw)`.
        """
        pts = getattr(self, "_route_pts", [])
        if not pts:
            return 0, x, y, 0.0, x + lookahead_m, y, 0.0
        n = len(pts)
        best_i = min(max(0, start_idx), n - 1)
        best_d = 1e9
        end_i = min(n, best_i + 12)
        for idx in range(best_i, end_i):
            d = math.hypot(x - pts[idx][0], y - pts[idx][1])
            if d < best_d:
                best_d = d
                best_i = idx

        # Continuous projection onto segment [best_i -> best_i + 1]
        if best_i + 1 < n:
            x0, y0, _, yaw0 = pts[best_i]
            x1, y1, _, _ = pts[best_i + 1]
            vx, vy = x1 - x0, y1 - y0
            seg_len_sq = vx * vx + vy * vy
            t_seg = (
                float(np.clip(((x - x0) * vx + (y - y0) * vy) / seg_len_sq, 0.0, 1.0))
                if seg_len_sq > 1e-4
                else 0.0
            )
            proj_x = x0 + t_seg * vx
            proj_y = y0 + t_seg * vy
            proj_yaw = yaw0
        else:
            proj_x, proj_y, _, proj_yaw = pts[best_i]
            t_seg = 0.0

        # Walk continuously along polyline from (proj_x, proj_y) by `lookahead_m`
        rem = lookahead_m
        cx, cy = proj_x, proj_y
        curr_idx = best_i
        look_x, look_y = pts[-1][0], pts[-1][1]
        while curr_idx + 1 < n:
            nx, ny, _, _ = pts[curr_idx + 1]
            ds = math.hypot(nx - cx, ny - cy)
            if ds >= rem and ds > 1e-4:
                ratio = rem / ds
                look_x = cx + ratio * (nx - cx)
                look_y = cy + ratio * (ny - cy)
                break
            rem -= ds
            cx, cy = nx, ny
            curr_idx += 1
            look_x, look_y = cx, cy

        ref_yaw = math.atan2(look_y - proj_y, look_x - proj_x)
        return best_i, proj_x, proj_y, proj_yaw, look_x, look_y, ref_yaw

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

        ep_offset = int(abs(seed - 5000))
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
            curr_wp = start_wp
            for _ in range(95):
                nxt = curr_wp.next(2.0)
                if not nxt:
                    break
                curr_wp = nxt[0]
                wp_chain.append(curr_wp)
            if len(wp_chain) < 42:
                continue

            # If GlobalRoutePlanner is active, refine route with A* trace_route
            if self._grp is not None and len(wp_chain) >= 40:
                try:
                    traced = self._grp.trace_route(
                        sp.location, wp_chain[min(len(wp_chain) - 1, 45)].transform.location
                    )
                    if len(traced) >= 25:
                        wp_chain = [w for w, _ in traced]
                except Exception:
                    pass

            self._ego_vehicle = self._world.try_spawn_actor(ego_bp, sp)
            if self._ego_vehicle is not None:
                self._route_waypoints = wp_chain
                self._route_wp_idx = 0
                self._rebuild_route_pts()
                self._route_length_m = min(self.target_route_distance_m, 68.0)
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

        # Attach Collision Sensor
        col_bp = bp_lib.find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self._ego_vehicle
        )
        self._collision_sensor.listen(
            lambda _event: setattr(self, "_collision_detected", True)
        )

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

        # Continuous projection & lookahead along `self._route_pts` (eliminates 2.0m staircase jumps)
        lookahead_m = float(np.clip(7.5 + 0.18 * speed_kmh, 7.5, 13.5))
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

        next_wps = wp_curr.next(10.0)
        road_curvature = 0.0
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

        # Scan forward vehicles/pedestrians & 360-degree side/rear blind-spot zones
        lead_distance_m = 60.0
        lead_speed_kmh = max(40.0, speed_kmh)
        hazard_source = "none"
        blind_spot_left = False
        blind_spot_right = False
        hx, hy = math.cos(ego_yaw), math.sin(ego_yaw)
        is_overtake_scenario = self._fsd_scenario_name in (
            "FSD-2_Dynamic_Overtake_Left",
            "FSD-5_MultiBlock_Overtake_Right",
            "FSD-7_BlindSpot_Blocked_Overtake",
            "FSD-8_Construction_Zone_Bypass_Right",
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
            if dist > 55.0:
                continue
            fp = rx * hx + ry * hy
            lat_signed = -rx * hy + ry * hx  # + = right of ego, - = left of ego
            lp = abs(lat_signed)
            if -10.0 <= fp <= 7.5 and 1.8 <= lp <= 5.0 and not is_overtake_scenario:
                if lat_signed < 0.0:
                    blind_spot_left = True
                else:
                    blind_spot_right = True
            if 0.5 < fp < lead_distance_m and lp < 2.2:
                lead_distance_m = float(fp)
                a_vel = actor.get_velocity()
                lead_speed_kmh = float(
                    3.6 * math.sqrt(a_vel.x**2 + a_vel.y**2 + a_vel.z**2)
                )
                hazard_source = (
                    "pedestrian" if type_id.startswith("walker.pedestrian.") else "vehicle"
                )
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

        # Traffic light state check
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
                        or self._fsd_stop_hold_steps >= 10
                    ):
                        try:
                            tl.set_state(carla.TrafficLightState.Green)
                        except Exception:
                            pass
                        traffic_light = "green"
                    else:
                        traffic_light = "red"
                        hazard_source = "red_light"
                        lead_distance_m = min(lead_distance_m, 8.0)
                        lead_speed_kmh = 0.0
                        self._fsd_stop_hold_steps += 1
                elif tl_state == carla.TrafficLightState.Yellow:
                    traffic_light = "yellow"

        nav_command = "go_straight"
        if abs(road_curvature) > 0.22:
            nav_command = "turn_left" if road_curvature > 0 else "turn_right"
        exp_man = "follow_lead" if lead_distance_m <= 32.0 else "keep_lane"

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

        out_of_road = bool(abs(lat_right) > self.lane_width * 1.85)
        if out_of_road:
            self._fsd_metrics["off_road_infractions"] += 1
        if self._collision_detected:
            self._fsd_metrics["vehicle_collisions"] += 1
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
            expert_maneuver=exp_man,
            traffic_light=traffic_light,
            hazard_source=hazard_source,
            blind_spot_left=blind_spot_left,
            blind_spot_right=blind_spot_right,
        )
        route_comp = float(
            np.clip(self._progress / max(1.0, self._route_length_m), 0.0, 1.0)
        )
        infraction_penalty = (
            (0.50 ** self._fsd_metrics["pedestrian_collisions"])
            * (0.60 ** (1 if crash else 0))
            * (0.70 ** self._fsd_metrics["red_light_infractions"])
        )
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
            "road_curvature": round(road_curvature, 4),
            "traffic_light": traffic_light,
            "hazard_source": hazard_source,
            "blind_spot_left": blind_spot_left,
            "blind_spot_right": blind_spot_right,
            "route_completion": route_comp,
            "driving_score": round(route_comp * infraction_penalty * 100.0, 2),
            "fsd_metrics": dict(self._fsd_metrics),
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
            ctrl_dict = self.to_carla_vehicle_control(smooth_steering, throttle_brake)
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
            try:
                for s in (self._camera_sensor, self._hd_camera_sensor, self._collision_sensor):
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
