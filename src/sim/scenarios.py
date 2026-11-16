"""Adversarial stress scenarios, live hazard injection, and realistic sensor/actuator wrapper over MetaDrive 3D & CARLA."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.sim.actuator_and_faults import (
    ActuatorDynamicsFilter,
    SOTIFSensorFaultInjector,
)
from src.sim.mapless_corridor import (
    LOCALIZATION_MODES,
    MaplessVisionCorridorEstimator,
)

SCENARIO_CATALOG: Dict[str, Dict[str, Any]] = {
    "nominal_urban_cruise": {
        "title": "Nominal Multi-Lane Urban Cruise",
        "description": "Baseline traffic cruise on real 3D MetaDrive/CARLA roads without adversarial faults.",
        "default_weather": "clear",
        "default_fault": "none",
    },
    "aggressive_cut_in": {
        "title": "Aggressive Adjacent-Lane Cut-In",
        "description": "Adjacent slow vehicle sharply cuts into ego lane at 17m while decelerating.",
        "default_weather": "clear",
        "default_fault": "none",
    },
    "sudden_lead_ebrake": {
        "title": "Lead Vehicle Emergency Braking (-6.5 m/s^2)",
        "description": "Lead car slams full emergency brakes to 0 km/h at short headway.",
        "default_weather": "clear",
        "default_fault": "none",
    },
    "blindspot_highspeed_overtaker": {
        "title": "High-Speed Blind-Spot Overtaker During Lane-Change",
        "description": "Slow lead vehicle ahead tempts overtake while a 64 km/h vehicle surges through left blind spot.",
        "default_weather": "clear",
        "default_fault": "none",
    },
    "occluded_jaywalker_dartout": {
        "title": "Occluded Crosswalk Jaywalker Dart-Out",
        "description": "Pedestrian steps into ego corridor laterally at 19m requiring immediate AEB yield.",
        "default_weather": "clear",
        "default_fault": "none",
    },
    "sotif_heavy_rain_glare_dropout": {
        "title": "SOTIF Heavy Rain + Sun Glare + Camera Dropout Burst",
        "description": "Wet low-friction road with heavy rain streaks, glare bloom, and transient camera blackout during lead braking.",
        "default_weather": "heavy_rain",
        "default_fault": "combined_sotif",
    },
}


class AdversarialScenarioDirector:
    """Manages scripted stress scenarios and one-click live hazards on top of real 3D simulator runs."""

    def __init__(self, scenario: str = "nominal_urban_cruise") -> None:
        s = str(scenario or "nominal_urban_cruise").strip().lower()
        self.scenario: str = s if s in SCENARIO_CATALOG else "nominal_urban_cruise"
        self._hazards: List[Dict[str, Any]] = []
        self._pedestrian_aeb_resolved: bool = False
        self._stop_hold_steps: int = 0
        self._script_triggered: bool = False
        self._ego_lat_shift_m: float = 0.0

    def set_scenario(self, scenario: str) -> str:
        s = str(scenario or "nominal_urban_cruise").strip().lower()
        if s in SCENARIO_CATALOG:
            self.scenario = s
        return self.scenario

    def reset(self) -> None:
        self._hazards.clear()
        self._pedestrian_aeb_resolved = False
        self._stop_hold_steps = 0
        self._script_triggered = False
        self._ego_lat_shift_m = 0.0

    def trigger_live_hazard(self, hazard_type: str, ego_speed_kmh: float = 35.0) -> str:
        """Spawn a live dynamic hazard into the current closed-loop episode."""
        ht = str(hazard_type or "clear").strip().lower()
        if ht == "clear":
            self._hazards.clear()
            return "cleared"

        if ht == "cut_in":
            self._hazards.append(
                {
                    "kind": "cut_in",
                    "category": "vehicle",
                    "x_m": 15.0,
                    "y_m": 3.0,
                    "speed_kmh": max(6.0, ego_speed_kmh - 20.0),
                    "vy_mps": -2.05,
                    "accel_mps2": -3.0,
                    "age": 0,
                    "ttl": 75,
                }
            )
        elif ht == "ebrake":
            self._hazards.append(
                {
                    "kind": "ebrake",
                    "category": "vehicle",
                    "x_m": 15.5,
                    "y_m": 0.0,
                    "speed_kmh": max(14.0, min(28.0, ego_speed_kmh - 8.0)),
                    "vy_mps": 0.0,
                    "accel_mps2": -8.0,
                    "age": 0,
                    "ttl": 75,
                }
            )
        elif ht == "jaywalker":
            self._hazards.append(
                {
                    "kind": "jaywalker",
                    "category": "pedestrian",
                    "x_m": 16.5,
                    "y_m": 3.2,
                    "speed_kmh": 0.0,
                    "vy_mps": -1.75,
                    "accel_mps2": 0.0,
                    "age": 0,
                    "ttl": 65,
                }
            )
        elif ht == "blindspot_left":
            self._hazards.append(
                {
                    "kind": "blindspot",
                    "category": "vehicle",
                    "x_m": -9.5,
                    "y_m": -3.2,
                    "speed_kmh": max(58.0, ego_speed_kmh + 26.0),
                    "vy_mps": 0.0,
                    "accel_mps2": 0.0,
                    "age": 0,
                    "ttl": 50,
                }
            )
        elif ht == "blindspot_right":
            self._hazards.append(
                {
                    "kind": "blindspot",
                    "category": "vehicle",
                    "x_m": -9.5,
                    "y_m": 3.2,
                    "speed_kmh": max(58.0, ego_speed_kmh + 26.0),
                    "vy_mps": 0.0,
                    "accel_mps2": 0.0,
                    "age": 0,
                    "ttl": 50,
                }
            )
        return ht

    def _maybe_spawn_scripted(self, step_idx: int, ego_speed_kmh: float) -> None:
        if self._script_triggered or not ((step_idx >= 18 and ego_speed_kmh >= 29.0) or step_idx >= 34):
            return
        sc = self.scenario
        if sc == "aggressive_cut_in":
            self._script_triggered = True
            self.trigger_live_hazard("cut_in", ego_speed_kmh=ego_speed_kmh)
        elif sc == "sudden_lead_ebrake":
            self._script_triggered = True
            self.trigger_live_hazard("ebrake", ego_speed_kmh=ego_speed_kmh)
        elif sc == "blindspot_highspeed_overtaker":
            self._script_triggered = True
            # Slow lead vehicle encourages overtake + fast rear-left blind-spot overtaker
            self._hazards.append(
                {
                    "kind": "slow_lead",
                    "category": "vehicle",
                    "x_m": 20.0,
                    "y_m": 0.0,
                    "speed_kmh": 12.0,
                    "vy_mps": 0.0,
                    "accel_mps2": 0.0,
                    "age": 0,
                    "ttl": 65,
                }
            )
            self.trigger_live_hazard("blindspot_left", ego_speed_kmh=ego_speed_kmh)
        elif sc == "occluded_jaywalker_dartout":
            self._script_triggered = True
            self.trigger_live_hazard("jaywalker", ego_speed_kmh=ego_speed_kmh)
        elif sc == "sotif_heavy_rain_glare_dropout":
            self._script_triggered = True
            self._hazards.append(
                {
                    "kind": "ebrake",
                    "category": "vehicle",
                    "x_m": 16.5,
                    "y_m": 0.0,
                    "speed_kmh": max(12.0, ego_speed_kmh - 10.0),
                    "vy_mps": 0.0,
                    "accel_mps2": -7.5,
                    "age": 0,
                    "ttl": 75,
                }
            )

    def step(
        self,
        rgb: np.ndarray,
        telemetry: Dict[str, Any],
        info: Dict[str, Any],
        step_idx: int,
        dt: float = 0.05,
        gated_maneuver: str = "keep_lane",
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], bool]:
        """Advance scenario hazards, merge sensor targets & check physical hazard collisions."""
        ego_v_kmh = float(telemetry.get("speed_kmh", 0.0))
        ego_v_mps = ego_v_kmh / 3.6
        self._maybe_spawn_scripted(step_idx, ego_v_kmh)

        tel_out = dict(telemetry)
        info_out = copy.deepcopy(info)
        radar_targets: List[Dict[str, Any]] = list(info_out.get("radar_targets", []) or [])
        crash_hazard = False

        if not self._hazards:
            info_out["radar_targets"] = radar_targets
            info_out["scenario_name"] = self.scenario
            return rgb, tel_out, info_out, False

        surviving: List[Dict[str, Any]] = []
        rgb_vis = rgb.copy()
        lead_d = float(tel_out.get("lead_distance_m", 60.0))
        lead_v = float(tel_out.get("lead_speed_kmh", 40.0))
        ped_d = float(tel_out.get("pedestrian_distance_m", 100.0))
        hz_src = str(tel_out.get("hazard_source", "none"))
        bs_left = bool(tel_out.get("blind_spot_left", False))
        bs_right = bool(tel_out.get("blind_spot_right", False))
        left_avail = bool(tel_out.get("left_lane_available", True))
        right_avail = bool(tel_out.get("right_lane_available", True))

        for h in self._hazards:
            h["age"] += 1
            kind = str(h["kind"])
            if h["age"] > int(h["ttl"]):
                continue

            # Target longitudinal dynamics per hazard profile
            v_mps = float(h["speed_kmh"]) / 3.6
            if kind == "cut_in":
                h["y_m"] = float(max(0.0, h["y_m"] + h["vy_mps"] * dt))
                if h["y_m"] <= 0.05:
                    h["vy_mps"] = 0.0
                if h["age"] < 24:
                    v_mps = max(6.0 / 3.6, v_mps + float(h["accel_mps2"]) * dt)
                else:
                    v_mps = min(46.0 / 3.6, v_mps + 3.4 * dt)
            elif kind == "ebrake":
                if h["age"] > 24 or (ego_v_kmh < 4.5 and h["x_m"] < 14.0 and h["age"] > 8):
                    v_mps = min(46.0 / 3.6, v_mps + 4.0 * dt)
                elif 1 <= h["age"] <= 22:
                    decel = float(h.get("accel_mps2", -8.0))
                    v_mps = max(0.0, v_mps + decel * dt)
            elif kind == "jaywalker":
                # Pedestrian crosses laterally; if ego stops to yield, clears corridor safely
                h["y_m"] = float(h["y_m"] + h["vy_mps"] * dt)
                if ego_v_kmh < 2.0 and h["x_m"] < 14.0:
                    self._stop_hold_steps += 1
                if h["y_m"] < -2.2 or (self._stop_hold_steps >= 3 and h["y_m"] < 0.2):
                    self._pedestrian_aeb_resolved = True
                    continue
            elif kind == "slow_lead":
                if h["age"] > 36:
                    v_mps = min(46.0 / 3.6, v_mps + 3.8 * dt)

            if gated_maneuver == "change_left":
                h["y_m"] = float(h["y_m"] + 2.1 * dt)
                self._ego_lat_shift_m -= 2.1 * dt
            elif gated_maneuver == "change_right":
                h["y_m"] = float(h["y_m"] - 2.1 * dt)
                self._ego_lat_shift_m += 2.1 * dt

            h["speed_kmh"] = float(v_mps * 3.6)
            vx_rel = v_mps - ego_v_mps
            h["x_m"] = float(h["x_m"] + vx_rel * dt)

            # Skip targets out of radar field
            if h["x_m"] < -25.0 or h["x_m"] > 72.0:
                continue

            # Physical collision envelope check
            x_env = 2.2 if kind == "blindspot" else 2.0
            if abs(h["x_m"]) < x_env and abs(h["y_m"]) < 1.35:
                crash_hazard = True

            radar_targets.append(
                {
                    "x_m": round(h["x_m"], 2),
                    "y_m": round(h["y_m"], 2),
                    "vx_rel_mps": round(vx_rel, 2),
                    "vy_rel_mps": round(float(h["vy_mps"]), 2),
                    "speed_kmh": round(h["speed_kmh"], 2),
                    "category": str(h["category"]),
                    "source": "scenario_hazard",
                }
            )

            # Update ground-truth telemetry aggregates (do NOT alter left/right lane geometry for blindspot)
            xm, ym = float(h["x_m"]), float(h["y_m"])
            if (-10.5 <= xm <= 7.5 and 1.5 <= abs(ym) <= 5.0) or (
                -20.0 <= xm < -10.5 and 1.5 <= abs(ym) <= 5.0 and vx_rel > 2.5
            ):
                if ym < 0.0:
                    bs_left = True
                else:
                    bs_right = True

            if kind == "slow_lead" and 0.5 < xm < 25.0 and left_avail:
                tel_out["expert_maneuver"] = "change_left"

            in_path_or_cutin = 0.4 < xm < 58.0 and (
                abs(ym) < 2.25
                or (h["category"] == "vehicle" and abs(ym) <= 3.6 and ym * float(h["vy_mps"]) < -0.3)
            )
            if h["category"] == "pedestrian" and 0.4 < xm < ped_d and abs(ym) < 3.4:
                ped_d = xm
                if abs(ym) < 2.4 and xm <= lead_d:
                    lead_d = xm
                    lead_v = 0.0
                    hz_src = "pedestrian"
            elif in_path_or_cutin and xm < lead_d:
                lead_d = xm
                lead_v = float(h["speed_kmh"])
                if hz_src != "pedestrian":
                    hz_src = "vehicle"

            # Project visual bounding cue onto RGB frame for camera perception consistency
            if 2.0 < xm < 50.0 and abs(ym) < 6.0:
                img_h, img_w = rgb_vis.shape[:2]
                horizon_y = int(img_h * 0.24)
                scale = float(np.clip(22.0 / xm, 0.22, 1.45))
                cy = int(horizon_y + (img_h - horizon_y - 10) * np.clip(1.0 - xm / 55.0, 0.10, 0.94))
                cx = int(np.clip(img_w * 0.5 + (ym / 3.5) * (75.0 * scale + 25.0), 10, img_w - 10))
                if h["category"] == "pedestrian":
                    pw = max(8, int(20 * scale))
                    ph = max(16, int(48 * scale))
                    cv2.rectangle(
                        rgb_vis,
                        (max(0, cx - pw // 2), max(horizon_y, cy - ph)),
                        (min(img_w - 1, cx + pw // 2), min(img_h - 1, cy)),
                        (255, 90, 35),
                        -1,
                    )
                else:
                    cw = max(14, int(58 * scale))
                    ch = max(10, int(38 * scale))
                    col = (240, 40, 40) if kind in ("ebrake", "cut_in") else (45, 115, 215)
                    cv2.rectangle(
                        rgb_vis,
                        (max(0, cx - cw // 2), max(horizon_y, cy - ch)),
                        (min(img_w - 1, cx + cw // 2), min(img_h - 1, cy)),
                        col,
                        -1,
                    )

            surviving.append(h)

        self._hazards = surviving
        tel_out["lead_distance_m"] = round(lead_d, 2)
        tel_out["lead_speed_kmh"] = round(lead_v, 2)
        tel_out["pedestrian_distance_m"] = round(ped_d, 2)
        tel_out["hazard_source"] = hz_src
        tel_out["blind_spot_left"] = bs_left
        tel_out["blind_spot_right"] = bs_right
        tel_out["left_lane_available"] = left_avail
        tel_out["right_lane_available"] = right_avail
        info_out["radar_targets"] = radar_targets
        info_out["scenario_name"] = self.scenario
        info_out["active_hazard_count"] = len(surviving)
        if crash_hazard:
            info_out["crash"] = True
        return rgb_vis, tel_out, info_out, crash_hazard


class RealisticSimEnvWrapper:
    """Composes real MetaDrive 3D / CARLA with actuator dynamics, SOTIF faults & stress scenarios."""

    def __init__(
        self,
        base_env: Any,
        scenario: str = "nominal_urban_cruise",
        actuator_realism: bool = True,
        fault_mode: Optional[str] = None,
        weather: Optional[str] = None,
        localization_mode: str = "hd_map",
        endless_tour: bool = False,
        seed: int = 42,
    ) -> None:
        self.base_env = base_env
        try:
            setattr(self.base_env, "_external_scenario_control", True)
        except Exception:
            pass
        self.director = AdversarialScenarioDirector(scenario=scenario)
        self.actuator = ActuatorDynamicsFilter()
        self.actuator_realism: bool = bool(actuator_realism)
        self.fault_injector = SOTIFSensorFaultInjector(seed=seed)
        self.mapless_estimator = MaplessVisionCorridorEstimator(
            lane_width_m=float(getattr(base_env, "lane_width", 3.5)),
            camera_fov_deg=float(getattr(base_env, "camera_fov", 60.0)),
        )
        loc_m = str(localization_mode or "hd_map").strip().lower()
        self.localization_mode: str = loc_m if loc_m in LOCALIZATION_MODES else "hd_map"
        self.endless_tour: bool = bool(endless_tour)
        self.tour_legs_completed: int = 0
        self._last_predicted_trajectory: Optional[List[List[float]]] = None

        sc_meta = SCENARIO_CATALOG[self.director.scenario]
        self.weather: str = str(weather or sc_meta["default_weather"])
        eff_fault = str(fault_mode if fault_mode is not None else sc_meta["default_fault"])
        self.fault_injector.set_mode(eff_fault)

        self._step_count: int = 0
        self._last_speed_kmh: float = 35.0
        self._dt: float = 0.05

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    @property
    def step_dt(self) -> float:
        """Return native control step duration from wrapped simulator."""
        return float(getattr(self.base_env, "step_dt", self._dt))

    def set_scenario(self, scenario: str) -> str:
        s = self.director.set_scenario(scenario)
        sc_meta = SCENARIO_CATALOG[s]
        self.set_weather(str(sc_meta["default_weather"]))
        self.set_fault_mode(str(sc_meta["default_fault"]))
        return s

    def set_weather(self, weather: str, precipitation: float = 0.0) -> None:
        self.weather = str(weather or "clear").strip().lower()
        precip = float(precipitation)
        if precip <= 0.0:
            if "heavy" in self.weather or "rain" in self.weather:
                precip = 0.85 if "heavy" in self.weather else 0.50
            elif self.weather == "wet":
                precip = 0.30
        if hasattr(self.base_env, "set_weather"):
            self.base_env.set_weather(self.weather, precipitation=precip)

    def set_fault_mode(self, fault_mode: str) -> None:
        self.fault_injector.set_mode(fault_mode)

    def set_actuator_realism(self, enabled: bool) -> None:
        self.actuator_realism = bool(enabled)

    def set_localization_mode(self, mode: str) -> str:
        m = str(mode or "hd_map").strip().lower()
        if m in LOCALIZATION_MODES:
            self.localization_mode = m
        return self.localization_mode

    def set_endless_tour(self, enabled: bool) -> bool:
        self.endless_tour = bool(enabled)
        return self.endless_tour

    def set_town(self, town_name: str) -> str:
        if hasattr(self.base_env, "set_town"):
            return str(self.base_env.set_town(town_name))
        if str(town_name).isdigit() and hasattr(self.base_env, "set_map_blocks"):
            blocks = int(self.base_env.set_map_blocks(int(town_name)))
            return f"MetaDrive-{blocks}B"
        return str(town_name)

    def list_spawn_points(self) -> List[Dict[str, Any]]:
        if hasattr(self.base_env, "list_spawn_points"):
            return list(self.base_env.list_spawn_points())
        return []

    def set_destination(
        self,
        dest_spawn_idx: Optional[int] = None,
        dest_xy: Optional[Tuple[float, float]] = None,
        min_distance_m: float = 170.0,
    ) -> Dict[str, Any]:
        if hasattr(self.base_env, "set_destination"):
            try:
                return dict(
                    self.base_env.set_destination(
                        dest_spawn_idx=dest_spawn_idx,
                        dest_xy=dest_xy,
                        min_distance_m=min_distance_m,
                    )
                )
            except TypeError:
                return dict(
                    self.base_env.set_destination(
                        dest_spawn_idx=dest_spawn_idx,
                        dest_xy=dest_xy,
                    )
                )
        return {"rerouted": False}

    def spawn_roadblock(self, distance_ahead_m: float = 24.0) -> Dict[str, Any]:
        if hasattr(self.base_env, "spawn_roadblock"):
            return dict(self.base_env.spawn_roadblock(distance_ahead_m=distance_ahead_m))
        return {"spawned": False, "hazard": self.trigger_live_hazard("ebrake")}

    def trigger_live_hazard(self, hazard_type: str) -> str:
        return self.director.trigger_live_hazard(
            hazard_type=hazard_type, ego_speed_kmh=self._last_speed_kmh
        )

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        self._step_count = 0
        self.tour_legs_completed = 0
        self._last_predicted_trajectory = None
        self.director.reset()
        self.actuator.reset()
        self.fault_injector.reset(seed=seed)
        self.mapless_estimator.reset()
        try:
            setattr(self, "_last_gated_maneuver", "keep_lane")
            setattr(self.base_env, "_last_gated_maneuver", "keep_lane")
            setattr(self.base_env, "_external_scenario_control", True)
        except Exception:
            pass
        self.set_weather(self.weather)
        rgb, telemetry, info = self.base_env.reset(seed=seed)
        self._last_speed_kmh = float(telemetry.get("speed_kmh", 35.0))
        info = dict(info)
        info.setdefault("radar_targets", [])
        tel_resolved, loc_diag = self.mapless_estimator.resolve_telemetry(
            localization_mode=self.localization_mode,
            hd_telemetry=telemetry,
            rgb=rgb,
            imu_telemetry=info.get("imu_telemetry"),
            radar_targets=info.get("radar_targets"),
            predicted_trajectory=None,
            hd_map_dropped=self.fault_injector.is_hd_map_dropped(0),
            dt=self.step_dt,
        )
        info["scenario_name"] = self.director.scenario
        info["fault_mode"] = self.fault_injector.fault_mode
        info["weather"] = self.weather
        info["localization_mode"] = self.localization_mode
        info["localization_diagnostics"] = loc_diag
        info["endless_tour"] = self.endless_tour
        info["tour_legs_completed"] = self.tour_legs_completed
        return rgb, tel_resolved, info

    def step(
        self, action: Tuple[float, float] | np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any], float, bool, Dict[str, Any]]:
        self._step_count += 1
        dt_eff = self.step_dt
        steer = float(action[0])
        tb = float(action[1])
        act_diag: Dict[str, float] = {}
        if self.actuator_realism:
            (steer, tb), act_diag = self.actuator.apply(
                steer, tb, speed_kmh=self._last_speed_kmh, dt=dt_eff
            )

        rgb, telemetry, reward, done, info = self.base_env.step((steer, tb))
        self._last_speed_kmh = float(telemetry.get("speed_kmh", 0.0))

        gated_maneuver = str(
            getattr(
                self,
                "_last_gated_maneuver",
                getattr(self.base_env, "_last_gated_maneuver", "keep_lane"),
            )
        )
        rgb_sc, tel_sc, info_sc, crash_hz = self.director.step(
            rgb,
            telemetry,
            info,
            step_idx=self._step_count,
            dt=dt_eff,
            gated_maneuver=gated_maneuver,
        )
        rgb_fault = self.fault_injector.inject_camera(rgb_sc, self._step_count)
        info_sc["radar_targets"] = self.fault_injector.inject_radar(
            list(info_sc.get("radar_targets", []) or []), self._step_count
        )
        hd_dropped = self.fault_injector.is_hd_map_dropped(self._step_count)
        tel_sc, loc_diag = self.mapless_estimator.resolve_telemetry(
            localization_mode=self.localization_mode,
            hd_telemetry=tel_sc,
            rgb=rgb_fault,
            imu_telemetry=info_sc.get("imu_telemetry"),
            radar_targets=info_sc.get("radar_targets"),
            predicted_trajectory=self._last_predicted_trajectory,
            hd_map_dropped=hd_dropped,
            dt=dt_eff,
        )

        info_sc["fault_mode"] = self.fault_injector.fault_mode
        info_sc["weather"] = self.weather
        info_sc["localization_mode"] = self.localization_mode
        info_sc["localization_diagnostics"] = loc_diag
        if act_diag:
            info_sc["actuator_diagnostics"] = act_diag

        if "heavy" in self.weather or self.weather in ("wet", "night_rain", "rain"):
            tel_sc["wet_road"] = True
            tel_sc["a_max_mps2"] = 4.5
            tel_sc["uncertainty_margin_m"] = max(float(tel_sc.get("uncertainty_margin_m", 0.0)), 1.8)

        if crash_hz:
            done = True
            reward = float(reward) - 10.0
        elif self.endless_tour and bool(info_sc.get("arrive_dest", False)) and not bool(info_sc.get("crash", False)):
            reroute_info = self.set_destination(min_distance_m=170.0)
            if reroute_info.get("rerouted"):
                self.tour_legs_completed += 1
                info_sc["arrive_dest"] = False
                info_sc["tour_leg_completed"] = True
                done = False
        info_sc["endless_tour"] = self.endless_tour
        info_sc["tour_legs_completed"] = self.tour_legs_completed
        return rgb_fault, tel_sc, float(reward), bool(done), info_sc

    def close(self) -> None:
        self.base_env.close()


__all__ = [
    "AdversarialScenarioDirector",
    "RealisticSimEnvWrapper",
    "SCENARIO_CATALOG",
]
