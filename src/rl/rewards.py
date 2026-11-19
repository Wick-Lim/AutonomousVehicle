"""Physics-Based Closed-Loop Driving Reward Function for Track 3 RL Fine-Tuning.

Implements:
    r_t = 1.5 * delta_s_t
        + 15.0 * I_arrive
        - 25.0 * I_crash
        - 15.0 * I_oor
        - 2.0 * max(0, (2.0 - TTC_t) / 2.0)^2
        - 0.35 * |e_{y,t}|
        - 0.15 * min(1, |j_t| / 5.0)^2
        - 0.8 * I_lane_inv
with a legal-stop exemption (zero stall penalty at red lights, pedestrian yields, or stopped lead traffic).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


class ClosedLoopDrivingReward:
    """Stateful per-episode physical reward calculator for closed-loop RL rollouts."""

    def __init__(self, dt: float = 0.10) -> None:
        self.dt = max(1e-3, float(dt))
        self.reset()

    def reset(self) -> None:
        self._prev_route_comp: Optional[float] = None
        self._prev_ego_xy: Optional[Tuple[float, float]] = None
        self._prev_speed_mps: Optional[float] = None
        self._prev_accel_mps2: float = 0.0
        self._prev_lane_inv_count: int = 0

    def compute(
        self,
        telemetry: Dict[str, Any],
        info: Dict[str, Any],
        dt: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute scalar step reward and component breakdown."""
        eff_dt = max(1e-3, float(dt if dt is not None else self.dt))
        speed_kmh = max(0.0, float(telemetry.get("speed_kmh", 0.0)))
        speed_mps = speed_kmh / 3.6
        lane_off = abs(float(telemetry.get("lane_offset_m", 0.0)))

        # 1. Forward progress delta_s_t (meters)
        delta_s = 0.0
        nav_state = info.get("nav_state") if isinstance(info, dict) else None
        cur_xy: Optional[Tuple[float, float]] = None
        if (
            isinstance(nav_state, dict)
            and isinstance(nav_state.get("ego_xy_yaw"), (list, tuple))
            and len(nav_state["ego_xy_yaw"]) >= 2
        ):
            cur_xy = (
                float(nav_state["ego_xy_yaw"][0]),
                float(nav_state["ego_xy_yaw"][1]),
            )

        route_comp = info.get("route_completion")
        if cur_xy is not None and self._prev_ego_xy is not None:
            step_dist = math.hypot(
                cur_xy[0] - self._prev_ego_xy[0],
                cur_xy[1] - self._prev_ego_xy[1],
            )
            delta_s = float(np.clip(step_dist, 0.0, 4.5))
            if route_comp is not None and self._prev_route_comp is not None:
                if float(route_comp) < float(self._prev_route_comp) - 1e-4:
                    delta_s = 0.0  # Reject backward/off-route motion
        elif route_comp is not None and self._prev_route_comp is not None:
            delta_s = float(
                np.clip((float(route_comp) - float(self._prev_route_comp)) * 180.0, -0.5, 4.5)
            )
        else:
            delta_s = float(np.clip(speed_mps * eff_dt, 0.0, 4.5))

        self._prev_ego_xy = cur_xy
        if route_comp is not None:
            self._prev_route_comp = float(route_comp)

        # 2. Continuous TTC safety penalty across vehicles and pedestrians
        lead_d = float(telemetry.get("lead_distance_m", 60.0))
        lead_v_mps = max(0.0, float(telemetry.get("lead_speed_kmh", 40.0)) / 3.6)
        ped_d = float(telemetry.get("pedestrian_distance_m", 100.0))

        ttc_cands = []
        v_close = speed_mps - lead_v_mps
        if lead_d < 50.0 and v_close > 0.15 and speed_mps > 0.5:
            ttc_cands.append(max(0.05, (lead_d - 0.5) / v_close))
        if ped_d < 45.0 and speed_mps > 0.3:
            ttc_cands.append(max(0.05, (ped_d - 0.5) / speed_mps))
        ttc_s = min(ttc_cands) if ttc_cands else 99.0
        ttc_pen = -2.0 * (max(0.0, (2.0 - ttc_s) / 2.0) ** 2)

        # 3. Longitudinal jerk penalty (ISO 2631-1 smoothness)
        if self._prev_speed_mps is not None:
            raw_acc = float(np.clip((speed_mps - self._prev_speed_mps) / eff_dt, -10.0, 10.0))
            jerk = abs((raw_acc - self._prev_accel_mps2) / eff_dt)
            self._prev_accel_mps2 = 0.7 * self._prev_accel_mps2 + 0.3 * raw_acc
        else:
            jerk = 0.0
            self._prev_accel_mps2 = 0.0
        self._prev_speed_mps = speed_mps
        jerk_pen = -0.15 * (min(1.0, jerk / 5.0) ** 2)

        # 4. Terminal & infraction terms
        oor = bool(info.get("out_of_road", False))
        crash = bool(info.get("crash", False)) and not oor
        arrive = bool(info.get("arrive_dest", False)) and not crash and not oor

        lane_inv = False
        if "lane_invasions" in info:
            cur_inv = int(info.get("lane_invasions", 0))
            if cur_inv > self._prev_lane_inv_count:
                lane_inv = True
                self._prev_lane_inv_count = cur_inv
        elif bool(info.get("lane_invasion", False)):
            lane_inv = True

        # 5. Anti-stall term with strict legal-stop exemption
        legal_stop = bool(
            str(telemetry.get("traffic_light", "green")).lower() == "red"
            or ped_d < 22.0
            or lead_d < 14.0
        )
        stall_pen = -0.08 if (speed_kmh < 2.0 and not legal_stop and not arrive) else 0.0

        r_prog = 1.5 * delta_s
        r_arrive = 15.0 if arrive else 0.0
        r_crash = -25.0 if crash else 0.0
        r_oor = -20.0 if oor else 0.0
        # Dense lane-centering + heading alignment feedback for crisp curve-tracking credit assignment
        excess_lat = max(0.0, min(lane_off, 4.0) - 0.5)
        heading_err = abs(float(telemetry.get("heading_error", 0.0)))
        r_lat = -0.65 * min(lane_off, 4.0) - 0.45 * (excess_lat ** 2) - 0.50 * min(heading_err, 1.0)
        r_inv = -0.8 if lane_inv else 0.0

        total = (
            r_prog
            + r_arrive
            + r_crash
            + r_oor
            + ttc_pen
            + r_lat
            + jerk_pen
            + r_inv
            + stall_pen
        )
        return float(total), {
            "progress": round(r_prog, 4),
            "arrive": round(r_arrive, 4),
            "crash": round(r_crash, 4),
            "oor": round(r_oor, 4),
            "ttc_pen": round(ttc_pen, 4),
            "lat_pen": round(r_lat, 4),
            "jerk_pen": round(jerk_pen, 4),
            "lane_inv": round(r_inv, 4),
            "stall_pen": round(stall_pen, 4),
            "ttc_s": round(ttc_s, 2),
        }


__all__ = ["ClosedLoopDrivingReward"]
