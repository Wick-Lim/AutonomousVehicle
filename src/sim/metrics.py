"""Standardized ISO/CARLA-Leaderboard Safety & Comfort Telemetry Metrics (TTC, THW, Jerk, Driving Score)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class EpisodeSafetyComfortTracker:
    """Computes per-step and episode-level TTC, THW, acceleration/jerk, comfort rate, and Driving Score."""

    def __init__(self, dt: float = 0.05) -> None:
        self.dt = max(1e-3, float(dt))
        self.reset()

    def reset(self) -> None:
        self._step_idx: int = 0
        self._prev_speed_mps: Optional[float] = None
        self._prev_steer: float = 0.0
        self._ema_long_accel: float = 0.0
        self._ema_lat_accel: float = 0.0
        self._ema_long_jerk: float = 0.0
        self._ema_lat_jerk: float = 0.0

        self._ttcs: List[float] = []
        self._thws: List[float] = []
        self._long_accels: List[float] = []
        self._lat_accels: List[float] = []
        self._long_jerks: List[float] = []
        self._lat_jerks: List[float] = []
        self._comfort_flags: List[bool] = []
        self._cruise_comfort_flags: List[bool] = []

        self.vehicle_collisions: int = 0
        self.pedestrian_collisions: int = 0
        self.red_light_infractions: int = 0
        self.off_road_events: int = 0
        self.lane_invasions: int = 0
        self.critical_ttc_violations: int = 0
        self._prev_lane_inv_count: int = 0

    def step(
        self,
        telemetry: Dict[str, Any],
        control_action: Tuple[float, float],
        gated_info: Optional[Dict[str, Any]] = None,
        extra_info: Optional[Dict[str, Any]] = None,
        dt: Optional[float] = None,
    ) -> Dict[str, float]:
        """Record one closed-loop step and return instantaneous safety/comfort metrics."""
        self._step_idx += 1
        dt_eff = max(1e-3, float(dt if dt is not None else self.dt))
        info = extra_info or {}
        _ = gated_info

        speed_mps = max(0.0, float(telemetry.get("speed_kmh", 0.0)) / 3.6)

        # 1. Time-to-Collision (TTC) & Time Headway (THW) across vehicles and pedestrians
        ttc_candidates: List[float] = []
        lead_d = float(telemetry.get("lead_distance_m", 60.0))
        lead_v_mps = max(0.0, float(telemetry.get("lead_speed_kmh", 40.0)) / 3.6)
        v_close_lead = speed_mps - lead_v_mps
        if lead_d < 58.0 and v_close_lead > 0.15 and speed_mps > 0.5:
            ttc_candidates.append(max(0.05, (lead_d - 0.5) / v_close_lead))

        ped_d = float(telemetry.get("pedestrian_distance_m", 100.0))
        if ped_d < 50.0 and speed_mps > 0.3:
            ttc_candidates.append(max(0.05, (ped_d - 0.5) / speed_mps))

        ttc_s = min(ttc_candidates) if ttc_candidates else 99.0
        thw_s = (lead_d / speed_mps) if (lead_d < 58.0 and speed_mps > 1.0) else 99.0
        if ttc_s < 0.75:
            self.critical_ttc_violations += 1

        # 2. Longitudinal & lateral acceleration + smoothed jerk (m/s^3)
        imu = info.get("imu_telemetry") if isinstance(info, dict) else None
        raw_long_acc = 0.0
        if self._prev_speed_mps is not None:
            raw_long_acc = (speed_mps - self._prev_speed_mps) / dt_eff
        elif isinstance(imu, dict) and "long_accel_mps2" in imu:
            raw_long_acc = float(imu["long_accel_mps2"])
        raw_long_acc = float(np.clip(raw_long_acc, -12.0, 12.0))

        steer = float(control_action[0])
        if isinstance(imu, dict) and "lat_accel_mps2" in imu and abs(float(imu["lat_accel_mps2"])) > 1e-4:
            raw_lat_acc = float(imu["lat_accel_mps2"])
        else:
            yaw_rate_est = (
                float(imu["yaw_rate_rps"])
                if isinstance(imu, dict) and abs(float(imu.get("yaw_rate_rps", 0.0))) > 1e-4
                else steer * 0.14 * min(15.0, speed_mps)
            )
            raw_lat_acc = speed_mps * yaw_rate_est
        raw_lat_acc = float(np.clip(raw_lat_acc, -12.0, 12.0))

        prev_long_acc = self._ema_long_accel
        prev_lat_acc = self._ema_lat_accel
        is_crash_step = bool(info.get("crash", False))
        if self._step_idx == 1:
            self._ema_long_accel = raw_long_acc
            self._ema_lat_accel = raw_lat_acc
            self._ema_long_jerk = 0.0
            self._ema_lat_jerk = 0.0
            long_jerk = 0.0
            lat_jerk = 0.0
        elif is_crash_step:
            # Exclude rigid-body collision contact impulse from kinematic ride-comfort series
            long_jerk = self._ema_long_jerk
            lat_jerk = self._ema_lat_jerk
        else:
            alpha_acc = dt_eff / (0.25 + dt_eff)
            self._ema_long_accel = (1.0 - alpha_acc) * prev_long_acc + alpha_acc * raw_long_acc
            self._ema_lat_accel = (1.0 - alpha_acc) * prev_lat_acc + alpha_acc * raw_lat_acc
            raw_long_jerk = (self._ema_long_accel - prev_long_acc) / dt_eff
            raw_lat_jerk = (self._ema_lat_accel - prev_lat_acc) / dt_eff
            alpha_jerk = dt_eff / (0.25 + dt_eff)
            self._ema_long_jerk = (1.0 - alpha_jerk) * self._ema_long_jerk + alpha_jerk * raw_long_jerk
            self._ema_lat_jerk = (1.0 - alpha_jerk) * self._ema_lat_jerk + alpha_jerk * raw_lat_jerk
            long_jerk = self._ema_long_jerk
            lat_jerk = self._ema_lat_jerk

        self._prev_speed_mps = speed_mps
        self._prev_steer = steer

        if self._step_idx > 1:
            self._ttcs.append(float(ttc_s))
            self._thws.append(float(thw_s))
            if not is_crash_step:
                self._long_accels.append(float(self._ema_long_accel))
                self._lat_accels.append(float(self._ema_lat_accel))
                self._long_jerks.append(float(long_jerk))
                self._lat_jerks.append(float(lat_jerk))

                comfortable = bool(
                    abs(self._ema_long_accel) <= 3.5
                    and abs(self._ema_lat_accel) <= 3.0
                    and abs(long_jerk) <= 5.0
                )
                self._comfort_flags.append(comfortable)
                if ttc_s >= 2.2:
                    self._cruise_comfort_flags.append(comfortable)

        # 3. Track infractions
        if bool(info.get("out_of_road", False)):
            self.off_road_events += 1
        elif bool(info.get("crash", False)):
            if str(telemetry.get("hazard_source", "")).lower() == "pedestrian":
                self.pedestrian_collisions += 1
            else:
                self.vehicle_collisions += 1

        if (
            str(telemetry.get("traffic_light", "green")).lower() == "red"
            and speed_mps > 3.5
            and lead_d < 4.0
        ):
            self.red_light_infractions += 1

        if "lane_invasions" in info:
            cur_inv = int(info.get("lane_invasions", 0))
            if cur_inv > self._prev_lane_inv_count:
                self.lane_invasions += cur_inv - self._prev_lane_inv_count
                self._prev_lane_inv_count = cur_inv
        elif bool(info.get("lane_invasion", False)):
            self.lane_invasions += 1

        return {
            "ttc_s": round(float(ttc_s), 2),
            "thw_s": round(float(thw_s), 2),
            "long_accel_mps2": round(float(self._ema_long_accel), 3),
            "lat_accel_mps2": round(float(self._ema_lat_accel), 3),
            "long_jerk_mps3": round(float(long_jerk), 3),
            "lat_jerk_mps3": round(float(lat_jerk), 3),
        }

    def summarize(
        self,
        route_completion: float = 0.0,
        arrived: bool = False,
        crashed: bool = False,
    ) -> Dict[str, Any]:
        """Compute final episode safety, ISO comfort & CARLA-Leaderboard Driving Score."""
        progress_pct = 100.0 if arrived else float(np.clip(route_completion * 100.0, 0.0, 100.0))
        if crashed and self.vehicle_collisions == 0 and self.pedestrian_collisions == 0 and self.off_road_events == 0:
            veh_cols = 1
        else:
            veh_cols = self.vehicle_collisions

        penalty = (
            (0.50 ** min(3, self.pedestrian_collisions))
            * (0.60 ** min(3, veh_cols))
            * (0.70 ** min(3, self.red_light_infractions))
            * (0.85 ** min(3, self.off_road_events))
            * (0.92 ** min(5, self.lane_invasions))
            * (0.95 ** min(3, self.critical_ttc_violations))
        )
        driving_score = float(np.clip(progress_pct * penalty, 0.0, 100.0))

        closing_ttcs = [t for t in self._ttcs if t < 90.0]
        finite_thws = [h for h in self._thws if h < 90.0]
        abs_jerks = [abs(j) for j in self._long_jerks]

        return {
            "steps": self._step_idx,
            "route_progress_pct": round(progress_pct, 2),
            "driving_score_pct": round(driving_score, 2),
            "min_ttc_s": round(float(min(closing_ttcs)) if closing_ttcs else 99.0, 2),
            "mean_thw_s": round(float(np.mean(finite_thws)) if finite_thws else 99.0, 2),
            "rms_long_jerk_mps3": round(
                float(np.sqrt(np.mean(np.square(self._long_jerks)))) if self._long_jerks else 0.0,
                3,
            ),
            "rms_lat_jerk_mps3": round(
                float(np.sqrt(np.mean(np.square(self._lat_jerks)))) if self._lat_jerks else 0.0,
                3,
            ),
            "p95_jerk_mps3": round(
                float(np.percentile(abs_jerks, 95)) if abs_jerks else 0.0, 3
            ),
            "max_jerk_mps3": round(float(max(abs_jerks)) if abs_jerks else 0.0, 3),
            "iso_comfort_rate_pct": round(
                float(np.mean(self._comfort_flags) * 100.0) if self._comfort_flags else 100.0,
                2,
            ),
            "cruise_comfort_rate_pct": round(
                float(np.mean(self._cruise_comfort_flags) * 100.0)
                if self._cruise_comfort_flags
                else 100.0,
                2,
            ),
            "infraction_counts": {
                "vehicle_collisions": int(veh_cols),
                "pedestrian_collisions": int(self.pedestrian_collisions),
                "red_light_infractions": int(self.red_light_infractions),
                "off_road_events": int(self.off_road_events),
                "lane_invasions": int(self.lane_invasions),
                "critical_ttc_violations": int(self.critical_ttc_violations),
            },
        }


def aggregate_leaderboard(episode_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-episode safety & comfort summaries across scenarios."""
    if not episode_summaries:
        return {
            "episodes": 0,
            "mean_driving_score_pct": 0.0,
            "min_ttc_s": 99.0,
            "mean_thw_s": 99.0,
            "mean_rms_long_jerk_mps3": 0.0,
            "mean_iso_comfort_rate_pct": 100.0,
        }

    closing_ttcs = [float(e["min_ttc_s"]) for e in episode_summaries if float(e.get("min_ttc_s", 99.0)) < 90.0]
    finite_thws = [float(e["mean_thw_s"]) for e in episode_summaries if float(e.get("mean_thw_s", 99.0)) < 90.0]
    return {
        "episodes": len(episode_summaries),
        "mean_driving_score_pct": round(
            float(np.mean([float(e["driving_score_pct"]) for e in episode_summaries])), 2
        ),
        "min_ttc_s": round(float(min(closing_ttcs)) if closing_ttcs else 99.0, 2),
        "mean_thw_s": round(float(np.mean(finite_thws)) if finite_thws else 99.0, 2),
        "mean_rms_long_jerk_mps3": round(
            float(np.mean([float(e["rms_long_jerk_mps3"]) for e in episode_summaries])), 3
        ),
        "mean_rms_lat_jerk_mps3": round(
            float(np.mean([float(e["rms_lat_jerk_mps3"]) for e in episode_summaries])), 3
        ),
        "mean_iso_comfort_rate_pct": round(
            float(np.mean([float(e["iso_comfort_rate_pct"]) for e in episode_summaries])), 2
        ),
    }


__all__ = ["EpisodeSafetyComfortTracker", "aggregate_leaderboard"]
