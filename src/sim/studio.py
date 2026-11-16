"""Thread-safe closed-loop simulation studio session backing FastAPI /cockpit and WS /ws/sim."""

from __future__ import annotations

import base64
import importlib.util
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.env import create_driving_env
from src.env.carla_wrapper import AVAILABLE_CARLA_TOWNS
from src.runtime.engine import DrivingRuntimeEngine
from src.schemas import PerceptionModeLiteral
from src.sim.actuator_and_faults import VALID_FAULT_MODES
from src.sim.carla_launcher import get_carla_server_manager
from src.sim.mapless_corridor import LOCALIZATION_MODES
from src.sim.metrics import EpisodeSafetyComfortTracker
from src.sim.scenarios import SCENARIO_CATALOG, RealisticSimEnvWrapper
from src.utils.logging_utils import get_logger

_LOG = get_logger("autonomous_vehicle.studio")


class SimulationStudioManager:
    """Orchestrates live closed-loop 3D simulation + inference + WebSocket telemetry streaming."""

    def __init__(
        self,
        engine: DrivingRuntimeEngine,
        default_simulator: str = "metadrive",
        force_mock: Optional[bool] = None,
    ) -> None:
        self.engine = engine
        self._lock = threading.RLock()
        self.metadrive_3d_available: bool = importlib.util.find_spec("metadrive") is not None
        self.force_mock: Optional[bool] = force_mock

        sim_init = str(default_simulator or "metadrive").strip().lower()
        self.simulator: str = sim_init if sim_init in ("metadrive", "carla") else "metadrive"
        self.scenario: str = "nominal_urban_cruise"
        self.perception_mode: PerceptionModeLiteral = "vision_radar_fusion"
        self.shield_enabled: bool = True
        self.actuator_realism: bool = True
        self.weather: str = "clear"
        self.fault_mode: str = "none"
        self.localization_mode: str = "hd_map"
        self.endless_tour: bool = False
        self.town: str = "Town10HD_Opt"
        self.map_blocks: int = 3
        self.auto_launch_carla: bool = False
        self.manual_override: Dict[str, Any] = {
            "enabled": False,
            "steering": 0.0,
            "throttle_brake": 0.0,
        }

        self.running: bool = False
        self.episode_index: int = 0
        self.step_index: int = 0
        self._seed_base: int = 5000
        self._session_id: str = "cockpit_live"
        self._done: bool = False

        self._env: Optional[RealisticSimEnvWrapper] = None
        self._safety_tracker = EpisodeSafetyComfortTracker(dt=0.05)
        self._last_rgb: Optional[np.ndarray] = None
        self._last_telemetry: Dict[str, Any] = {}
        self._last_info: Dict[str, Any] = {}
        self._last_packet: Optional[Dict[str, Any]] = None

    @property
    def is_real_3d(self) -> bool:
        if self._env is None:
            return False
        return not bool(getattr(self._env.base_env, "use_mock", True))

    def _resolve_use_mock(self, sim_name: str, explicit_mock: Optional[bool] = None) -> bool:
        if explicit_mock is not None:
            return bool(explicit_mock)
        if self.force_mock is not None:
            return bool(self.force_mock)
        if sim_name == "metadrive":
            return not self.metadrive_3d_available
        return False

    def _build_env_locked(self, explicit_mock: Optional[bool] = None) -> None:
        self._close_env_locked()
        cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in self.engine.cfg.items()}
        env_cfg = dict(cfg.get("env", {}))
        env_cfg["map_blocks"] = int(self.map_blocks)
        carla_cfg = dict(env_cfg.get("carla", {}))
        carla_cfg["auto_launch_server"] = bool(self.auto_launch_carla)
        carla_cfg["town"] = str(self.town)
        env_cfg["carla"] = carla_cfg
        cfg["env"] = env_cfg

        use_mock = self._resolve_use_mock(self.simulator, explicit_mock=explicit_mock)
        base_env = create_driving_env(
            config=cfg,
            use_mock=use_mock,
            start_seed=self._seed_base,
            num_scenarios=25,
            simulator=self.simulator,
        )
        self._env = RealisticSimEnvWrapper(
            base_env=base_env,
            scenario=self.scenario,
            actuator_realism=self.actuator_realism,
            fault_mode=self.fault_mode,
            weather=self.weather,
            localization_mode=self.localization_mode,
            endless_tour=self.endless_tour,
            seed=self._seed_base + self.episode_index,
        )
        self._reset_episode_locked()

    def _reset_episode_locked(self) -> None:
        if self._env is None:
            return
        self.episode_index += 1
        self.step_index = 0
        self._done = False
        seed = self._seed_base + ((self.episode_index - 1) % 25)
        self._env.set_scenario(self.scenario)
        self._env.set_weather(self.weather)
        self._env.set_fault_mode(self.fault_mode)
        self._env.set_actuator_realism(self.actuator_realism)
        self._env.set_localization_mode(self.localization_mode)
        self._env.set_endless_tour(self.endless_tour)
        rgb, telemetry, info = self._env.reset(seed=seed)
        self.engine.reset_session(self._session_id)
        self._safety_tracker.reset()
        self._last_rgb = rgb
        self._last_telemetry = dict(telemetry)
        self._last_info = dict(info)

    def _close_env_locked(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception as exc:
                _LOG.warning(f"Error closing studio simulator environment: {exc}")
            finally:
                self._env = None

    @staticmethod
    def _encode_viewport_jpeg_b64(
        rgb: np.ndarray,
        info: Dict[str, Any],
        shield_triggered: bool,
        shield_reason: str,
        is_real_3d: bool,
        simulator: str,
    ) -> str:
        hd_rgb = info.get("hd_rgb")
        src = hd_rgb if (isinstance(hd_rgb, np.ndarray) and hd_rgb.ndim == 3) else rgb
        bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        canvas = cv2.resize(bgr, (640, 320), interpolation=cv2.INTER_LINEAR)

        tag = f"{simulator.upper()} {'3D' if is_real_3d else 'MOCK'}"
        cv2.rectangle(canvas, (8, 8), (185, 32), (16, 20, 28), -1)
        cv2.putText(
            canvas,
            tag,
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (95, 235, 150) if is_real_3d else (245, 195, 70),
            1,
            cv2.LINE_AA,
        )
        if shield_triggered:
            cv2.rectangle(canvas, (195, 8), (632, 32), (35, 35, 165), -1)
            cv2.putText(
                canvas,
                f"SHIELD INTERVENTION: {shield_reason.upper()}",
                (204, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            carla_stat = get_carla_server_manager().status(
                host=str(self.engine.cfg.get("env", {}).get("carla", {}).get("host", "127.0.0.1")),
                port=int(self.engine.cfg.get("env", {}).get("carla", {}).get("port", 2000)),
            )
            return {
                "running": bool(self.running),
                "simulator": self.simulator,
                "is_real_3d": self.is_real_3d,
                "env_initialized": self._env is not None,
                "metadrive_3d_available": self.metadrive_3d_available,
                "carla": carla_stat,
                "town": self.town,
                "map_blocks": int(self.map_blocks),
                "available_carla_towns": list(AVAILABLE_CARLA_TOWNS),
                "scenario": self.scenario,
                "available_scenarios": {
                    k: {"title": v["title"], "description": v["description"]}
                    for k, v in SCENARIO_CATALOG.items()
                },
                "available_faults": list(VALID_FAULT_MODES),
                "localization_mode": self.localization_mode,
                "available_localization_modes": list(LOCALIZATION_MODES),
                "endless_tour": bool(self.endless_tour),
                "perception_mode": self.perception_mode,
                "shield_enabled": bool(self.shield_enabled),
                "actuator_realism": bool(self.actuator_realism),
                "weather": self.weather,
                "fault_mode": self.fault_mode,
                "manual_override": dict(self.manual_override),
                "episode_index": int(self.episode_index),
                "step_index": int(self.step_index),
                "done": bool(self._done),
                "nav_state": self._last_info.get("nav_state") or {},
                "last_packet": self._last_packet,
            }

    def step_once(self) -> Dict[str, Any]:
        """Run a single closed-loop step (inference -> optional override -> 3D env -> telemetry)."""
        with self._lock:
            t0 = time.perf_counter()
            if self._env is None:
                self._build_env_locked()
            elif self._done:
                self._reset_episode_locked()

            assert self._env is not None
            assert self._last_rgb is not None
            sim_dt = float(getattr(self._env, "step_dt", 0.05))

            self.step_index += 1
            tel_payload = dict(self._last_telemetry)
            tel_payload["radar_targets"] = list(self._last_info.get("radar_targets", []) or [])
            tel_payload["imu_telemetry"] = self._last_info.get("imu_telemetry")
            tel_payload["elapsed_steps"] = 1

            resp = self.engine.step(
                session_id=self._session_id,
                image=self._last_rgb,
                telemetry=tel_payload,
                perception_mode=self.perception_mode,
                shield_enabled=self.shield_enabled,
                dt=sim_dt,
                strict_input=True,
            )

            steer = float(resp.control.steering)
            tb = float(resp.control.throttle_brake)
            if bool(self.manual_override.get("enabled", False)):
                steer = float(np.clip(self.manual_override.get("steering", 0.0), -1.0, 1.0))
                tb = float(np.clip(self.manual_override.get("throttle_brake", 0.0), -1.0, 1.0))

            try:
                setattr(self._env, "_last_gated_maneuver", resp.control.gated_maneuver)
                setattr(self._env.base_env, "_last_gated_maneuver", resp.control.gated_maneuver)
                self._env._last_predicted_trajectory = list(resp.decision.predicted_trajectory)
            except Exception:
                pass

            next_rgb, next_tel, _reward, done_step, next_info = self._env.step((steer, tb))
            instant_sc = self._safety_tracker.step(
                telemetry=next_tel,
                control_action=(steer, tb),
                gated_info={
                    "shield_triggered": resp.control.shield_triggered,
                    "shield_reason": resp.control.shield_reason,
                },
                extra_info=next_info,
                dt=sim_dt,
            )
            ep_summary = self._safety_tracker.summarize(
                route_completion=float(next_info.get("route_completion", 0.0)),
                arrived=bool(next_info.get("arrive_dest", False)),
                crashed=bool(next_info.get("crash", False)),
            )

            self._last_rgb = next_rgb
            self._last_telemetry = dict(next_tel)
            self._last_info = dict(next_info)
            self._done = bool(done_step)

            frame_b64 = self._encode_viewport_jpeg_b64(
                rgb=self._last_rgb,
                info=self._last_info,
                shield_triggered=resp.control.shield_triggered,
                shield_reason=resp.control.shield_reason,
                is_real_3d=self.is_real_3d,
                simulator=self.simulator,
            )
            total_ms = (time.perf_counter() - t0) * 1000.0

            packet: Dict[str, Any] = {
                "timestamp": time.time(),
                "running": bool(self.running),
                "done": bool(self._done),
                "simulator": self.simulator,
                "is_real_3d": self.is_real_3d,
                "town": self.town,
                "map_blocks": int(self.map_blocks),
                "scenario": self.scenario,
                "weather": str(self._last_info.get("weather", self.weather)),
                "fault_mode": str(self._last_info.get("fault_mode", self.fault_mode)),
                "localization_mode": str(self._last_info.get("localization_mode", self.localization_mode)),
                "localization_diagnostics": self._last_info.get("localization_diagnostics") or {},
                "endless_tour": bool(self.endless_tour),
                "tour_legs_completed": int(self._last_info.get("tour_legs_completed", 0)),
                "perception_mode": self.perception_mode,
                "shield_enabled": bool(self.shield_enabled),
                "actuator_realism": bool(self.actuator_realism),
                "manual_override": dict(self.manual_override),
                "episode_index": int(self.episode_index),
                "step_index": int(self.step_index),
                "latency_ms": round(float(resp.latency_ms), 2),
                "loop_ms": round(float(total_ms), 2),
                "frame_jpeg_b64": frame_b64,
                "telemetry": dict(self._last_telemetry),
                "radar_targets": list(self._last_info.get("radar_targets", []) or []),
                "imu_telemetry": self._last_info.get("imu_telemetry") or {},
                "nav_state": self._last_info.get("nav_state") or {},
                "route_completion_pct": round(float(self._last_info.get("route_completion", 0.0)) * 100.0, 1),
                "arrive_dest": bool(self._last_info.get("arrive_dest", False)),
                "crash": bool(self._last_info.get("crash", False)),
                "decision": resp.decision.model_dump(mode="python"),
                "aux_perception": resp.aux_perception.model_dump(mode="python"),
                "control": resp.control.model_dump(mode="python"),
                "safety_comfort": instant_sc,
                "episode_summary": ep_summary,
            }
            self._last_packet = packet
            return packet

    def apply_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a control command from HTTP /v1/sim/control or WebSocket /ws/sim."""
        with self._lock:
            action = str(cmd.get("action", "")).strip().lower()
            step_packet: Optional[Dict[str, Any]] = None
            nav_update: Optional[Dict[str, Any]] = None

            if action == "start":
                self.running = True
                if self._env is None:
                    self._build_env_locked()
            elif action == "pause":
                self.running = False
            elif action == "reset":
                if self._env is None:
                    self._build_env_locked()
                else:
                    self._reset_episode_locked()
            elif action == "step":
                self.running = False
                step_packet = self.step_once()
            elif action == "set_simulator":
                sim = str(cmd.get("simulator", self.simulator)).strip().lower()
                if sim in ("metadrive", "carla"):
                    self.simulator = sim
                if "auto_launch_carla" in cmd:
                    self.auto_launch_carla = bool(cmd["auto_launch_carla"])
                explicit_mock = cmd.get("use_mock", None)
                self._build_env_locked(
                    explicit_mock=bool(explicit_mock) if explicit_mock is not None else None
                )
            elif action == "set_town":
                if "map_blocks" in cmd:
                    self.map_blocks = max(1, min(8, int(cmd["map_blocks"])))
                    if self._env is not None and self.simulator == "metadrive":
                        self._env.set_town(str(self.map_blocks))
                        self._reset_episode_locked()
                if "town" in cmd:
                    t_req = str(cmd["town"]).strip()
                    if t_req in AVAILABLE_CARLA_TOWNS or t_req.lower() == "current":
                        self.town = t_req
                        if self._env is not None and self.simulator == "carla":
                            self._env.set_town(self.town)
                            self._reset_episode_locked()
            elif action == "set_destination":
                if self._env is None:
                    self._build_env_locked()
                assert self._env is not None
                d_idx = int(cmd["dest_index"]) if cmd.get("dest_index") is not None else None
                d_xy: Optional[tuple[float, float]] = None
                if isinstance(cmd.get("dest_xy"), (list, tuple)) and len(cmd["dest_xy"]) >= 2:
                    d_xy = (float(cmd["dest_xy"][0]), float(cmd["dest_xy"][1]))
                elif cmd.get("dest_x") is not None and cmd.get("dest_y") is not None:
                    d_xy = (float(cmd["dest_x"]), float(cmd["dest_y"]))
                nav_update = self._env.set_destination(dest_spawn_idx=d_idx, dest_xy=d_xy)
                self._refresh_nav_snapshot_locked()
            elif action == "random_destination":
                if self._env is None:
                    self._build_env_locked()
                assert self._env is not None
                nav_update = self._env.set_destination(min_distance_m=170.0)
                self._refresh_nav_snapshot_locked()
            elif action == "set_endless_tour":
                self.endless_tour = bool(cmd.get("enabled", not self.endless_tour))
                if self._env is not None:
                    self._env.set_endless_tour(self.endless_tour)
            elif action == "set_localization_mode":
                lm = str(cmd.get("localization_mode", self.localization_mode)).strip().lower()
                if lm in LOCALIZATION_MODES:
                    self.localization_mode = lm
                    if self._env is not None:
                        self._env.set_localization_mode(lm)
            elif action == "spawn_roadblock":
                if self._env is None:
                    self._build_env_locked()
                assert self._env is not None
                dist_m = float(cmd.get("distance_m", 24.0))
                nav_update = self._env.spawn_roadblock(distance_ahead_m=dist_m)
            elif action == "set_scenario":
                sc = str(cmd.get("scenario", self.scenario)).strip().lower()
                if sc in SCENARIO_CATALOG:
                    self.scenario = sc
                    sc_meta = SCENARIO_CATALOG[sc]
                    self.weather = str(sc_meta["default_weather"])
                    self.fault_mode = str(sc_meta["default_fault"])
                    if self._env is not None:
                        self._reset_episode_locked()
            elif action == "set_perception_mode":
                pm = str(cmd.get("perception_mode", self.perception_mode)).strip().lower()
                if pm in ("vision_radar_fusion", "vision_estimated", "gt"):
                    self.perception_mode = pm  # type: ignore[assignment]
            elif action == "set_shield":
                self.shield_enabled = bool(cmd.get("enabled", self.shield_enabled))
            elif action == "set_actuator_realism":
                self.actuator_realism = bool(cmd.get("enabled", self.actuator_realism))
                if self._env is not None:
                    self._env.set_actuator_realism(self.actuator_realism)
            elif action == "set_weather":
                w = str(cmd.get("weather", self.weather)).strip().lower()
                if w in ("clear", "wet", "heavy_rain", "night_rain"):
                    self.weather = w
                    if self._env is not None:
                        self._env.set_weather(self.weather)
            elif action == "inject_hazard":
                hz = str(cmd.get("hazard", "cut_in")).strip().lower()
                if self._env is None:
                    self._build_env_locked()
                assert self._env is not None
                self._env.trigger_live_hazard(hz)
            elif action == "inject_fault":
                fm = str(cmd.get("fault_mode", "none")).strip().lower()
                if fm in VALID_FAULT_MODES:
                    self.fault_mode = fm
                    if self._env is not None:
                        self._env.set_fault_mode(fm)
            elif action == "manual_override":
                self.manual_override = {
                    "enabled": bool(cmd.get("enabled", self.manual_override["enabled"])),
                    "steering": float(
                        np.clip(cmd.get("steering", self.manual_override["steering"]), -1.0, 1.0)
                    ),
                    "throttle_brake": float(
                        np.clip(
                            cmd.get("throttle_brake", self.manual_override["throttle_brake"]),
                            -1.0,
                            1.0,
                        )
                    ),
                }

            res = self.get_status()
            if nav_update is not None:
                res["nav_update"] = nav_update
            if step_packet is not None:
                res["frame"] = step_packet
            return res

    def _refresh_nav_snapshot_locked(self) -> None:
        """Refresh `_last_info['nav_state']` immediately after a live destination re-route."""
        if self._env is None:
            return
        base = self._env.base_env
        try:
            if hasattr(base, "_ego_vehicle") and getattr(base, "_ego_vehicle", None) is not None:
                ego_tr = base._ego_vehicle.get_transform()
                best_i = int(getattr(base, "_route_wp_idx", 0))
                self._last_info["nav_state"] = base._build_real_nav_state(ego_tr, best_i)
            elif hasattr(base, "_raw_env") and getattr(base, "_raw_env", None) is not None:
                veh = base._raw_env.agent
                self._last_info["nav_state"] = base._build_real_nav_state(
                    veh, float(self._last_info.get("route_completion", 0.0))
                )
            elif hasattr(base, "list_spawn_points"):
                nav_prev = dict(self._last_info.get("nav_state") or {})
                nav_prev["spawn_nodes"] = base.list_spawn_points()
                nav_prev["reroute_count"] = int(getattr(base, "_reroute_count", 0))
                if getattr(base, "_dest_location", None):
                    nav_prev["dest_xy"] = [
                        round(float(base._dest_location[0]), 1),
                        round(float(base._dest_location[1]), 1),
                    ]
                self._last_info["nav_state"] = nav_prev
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            self.running = False
            self._close_env_locked()


__all__ = ["SimulationStudioManager"]
