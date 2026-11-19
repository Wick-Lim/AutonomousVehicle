"""Pure End-to-End Neural Trajectory Tracker (Zero HD-Map PID, Zero Safety Shield).

Tracks exclusively the 6 ego-frame waypoints `[(x_k, y_k, v_k)]_{k=1}^6` predicted by
the neural decoder (`LayaPilot` in `obstacle_aware_e2e` mode) using geometric
Pure-Pursuit steering and simple PD longitudinal speed tracking.

Coordinate & control sign convention (unified across MetaDrive and CARLA via
`CarlaWrapper.to_carla_vehicle_control`):
  - Ego frame: `+x` forward (m), `+y` LEFT (m), `-y` RIGHT (m), `v` speed (km/h).
  - Control output: `steering in [-1, 1]` (`+` = turn LEFT, `-` = turn RIGHT),
    `throttle_brake in [-1, 1]` (`+` = throttle, `-` = brake).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class PureE2ETrajectoryTracker:
    """Geometric Pure-Pursuit + PD tracker driven strictly by neural waypoints."""

    def __init__(
        self,
        wheelbase_m: float = 2.85,
        max_steer_rad: float = 0.42,
        min_lookahead_m: float = 5.0,
        max_lookahead_m: float = 14.0,
        lookahead_speed_gain: float = 0.30,
        preview_blend: float = 0.30,
        max_steer_rate_per_sec: float = 3.2,
        steer_smoothing_alpha: float = 0.72,
        kp_speed: float = 0.085,
        kd_speed: float = 0.015,
        aeb_threshold: float = 0.70,
    ) -> None:
        self.wheelbase_m = float(wheelbase_m)
        self.max_steer_rad = float(max(0.15, max_steer_rad))
        self.min_lookahead_m = float(min_lookahead_m)
        self.max_lookahead_m = float(max_lookahead_m)
        self.lookahead_speed_gain = float(lookahead_speed_gain)
        self.preview_blend = float(np.clip(preview_blend, 0.0, 0.6))
        self.max_steer_rate_per_sec = float(max_steer_rate_per_sec)
        self.steer_smoothing_alpha = float(np.clip(steer_smoothing_alpha, 0.05, 1.0))
        self.kp_speed = float(kp_speed)
        self.kd_speed = float(kd_speed)
        self.aeb_threshold = float(aeb_threshold)

        self._prev_steer: float = 0.0
        self._prev_speed_error: float = 0.0
        self._has_prev_error: bool = False

    def reset(self) -> None:
        """Reset steering smoother and longitudinal derivative state at episode start."""
        self._prev_steer = 0.0
        self._prev_speed_error = 0.0
        self._has_prev_error = False

    @staticmethod
    def to_carla_control(steering: float, throttle_brake: float) -> Dict[str, Any]:
        """Convert `[steering (+left), throttle_brake]` to CARLA `VehicleControl` (`+steer = right`)."""
        steer = float(np.clip(-steering, -1.0, 1.0))
        tb = float(np.clip(throttle_brake, -1.0, 1.0))
        return {
            "steer": round(steer, 4),
            "throttle": round(tb if tb >= 0.0 else 0.0, 4),
            "brake": round(-tb if tb < 0.0 else 0.0, 4),
            "hand_brake": False,
            "reverse": False,
        }

    def _interpolate_lookahead_point(
        self, waypoints: np.ndarray, lookahead_m: float
    ) -> Tuple[float, float]:
        """Interpolate target point `(x_d, y_d)` along polyline `[(0, 0), wp_1, ..., wp_6]`."""
        pts = np.vstack([np.array([[0.0, 0.0]], dtype=np.float64), waypoints[:, :2]])
        cum_s = 0.0
        for idx in range(1, pts.shape[0]):
            dx = float(pts[idx, 0] - pts[idx - 1, 0])
            dy = float(pts[idx, 1] - pts[idx - 1, 1])
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-6:
                continue
            if cum_s + seg_len >= lookahead_m:
                alpha = (lookahead_m - cum_s) / seg_len
                x_d = float(pts[idx - 1, 0] + alpha * dx)
                y_d = float(pts[idx - 1, 1] + alpha * dy)
                return x_d, y_d
            cum_s += seg_len
        return float(pts[-1, 0]), float(pts[-1, 1])

    def compute_control(
        self,
        decision: Dict[str, Any],
        telemetry: Optional[Dict[str, Any]] = None,
        speed_kmh: Optional[float] = None,
        dt: float = 0.05,
    ) -> Tuple[Tuple[float, float], Dict[str, Any]]:
        """Compute `(steering, throttle_brake)` strictly from neural waypoints + ego speed.

        Note: If `telemetry` is supplied, ONLY `speed_kmh` is read. All HD-map fields
        (`lane_offset_m`, `heading_error`, `road_curvature`) and obstacle/radar fields
        (`lead_distance_m`, `pedestrian_distance_m`, `traffic_light`, etc.) are strictly ignored.
        """
        eff_dt = max(1e-3, float(dt))
        if speed_kmh is not None:
            ego_v_kmh = max(0.0, float(speed_kmh))
        elif telemetry is not None:
            ego_v_kmh = max(0.0, float(telemetry.get("speed_kmh", 0.0)))
        else:
            ego_v_kmh = 0.0
        ego_v_mps = ego_v_kmh / 3.6

        raw_traj = decision.get("predicted_trajectory")
        if raw_traj is None or len(raw_traj) == 0:
            # Safe neutral fallback if no trajectory is present
            self._prev_steer = 0.0
            return (0.0, -0.5), {
                "maneuver": str(decision.get("maneuver", "keep_lane")),
                "speed_level": int(decision.get("speed_level", 0)),
                "target_speed_kmh": 0.0,
                "shield_triggered": False,
                "shield_reason": "pure_e2e_missing_trajectory",
                "mpc_feedforward": 0.0,
                "steer_ramp_substeps": [0.0] * 5,
                "carla_control": self.to_carla_control(0.0, -0.5),
            }

        wp = np.asarray(raw_traj, dtype=np.float64)
        if wp.ndim != 2 or wp.shape[1] < 2:
            raise ValueError(f"Expected (K, 3) predicted_trajectory, got {wp.shape}")

        # 1. Pure-Pursuit Geometric Steering strictly on neural polyline (near + preview chord)
        lookahead_m = float(
            np.clip(
                self.min_lookahead_m + self.lookahead_speed_gain * ego_v_mps,
                self.min_lookahead_m,
                self.max_lookahead_m,
            )
        )
        x_d, y_d = self._interpolate_lookahead_point(wp, lookahead_m)
        dist_sq = x_d * x_d + y_d * y_d
        if dist_sq < 0.25 or (x_d <= 0.1 and ego_v_kmh < 1.5):
            raw_steer = 0.0
            curvature = 0.0
        else:
            # +y_d (LEFT) -> +curvature -> +steering (TURN LEFT)
            curv_near = (2.0 * y_d) / max(dist_sq, 1.0)
            prev_ld = float(np.clip(lookahead_m * 1.65, lookahead_m + 3.0, 22.0))
            xp, yp = self._interpolate_lookahead_point(wp, prev_ld)
            p_dist_sq = xp * xp + yp * yp
            curv_prev = (2.0 * yp) / max(p_dist_sq, 1.0) if p_dist_sq >= 1.0 else curv_near
            curvature = (1.0 - self.preview_blend) * curv_near + self.preview_blend * curv_prev
            steer_rad = math.atan(self.wheelbase_m * curvature)
            raw_steer = float(np.clip(steer_rad / self.max_steer_rad, -1.0, 1.0))

        # Rate-limit + exponential smoothing for smooth actuator commands
        max_delta = self.max_steer_rate_per_sec * eff_dt
        limited_steer = float(
            np.clip(
                raw_steer,
                self._prev_steer - max_delta,
                self._prev_steer + max_delta,
            )
        )
        steering = float(
            np.clip(
                self.steer_smoothing_alpha * limited_steer
                + (1.0 - self.steer_smoothing_alpha) * self._prev_steer,
                -1.0,
                1.0,
            )
        )
        prev_s = self._prev_steer
        self._prev_steer = steering

        # 2. Longitudinal PD speed tracking strictly from neural waypoint speeds & neural AEB prob
        aeb_prob = float(decision.get("emergency_brake", 0.0))
        if wp.shape[1] >= 3:
            v1 = float(max(0.0, wp[0, 2]))
            v2 = float(max(0.0, wp[min(1, wp.shape[0] - 1), 2]))
            target_speed_kmh = 0.6 * v1 + 0.4 * v2
        else:
            target_speed_kmh = float(decision.get("target_speed_kmh", 30.0))

        if aeb_prob >= self.aeb_threshold:
            target_speed_kmh = 0.0
            throttle_brake = float(
                -np.clip(0.55 + 0.015 * ego_v_kmh, 0.55, 1.0)
            )
            self._prev_speed_error = -ego_v_kmh
            self._has_prev_error = True
        else:
            speed_err = target_speed_kmh - ego_v_kmh
            d_err = (
                (speed_err - self._prev_speed_error) / eff_dt
                if self._has_prev_error
                else 0.0
            )
            self._prev_speed_error = speed_err
            self._has_prev_error = True

            if target_speed_kmh < 1.5 and ego_v_kmh < 2.0:
                throttle_brake = -0.25
            else:
                u = self.kp_speed * speed_err + self.kd_speed * d_err
                throttle_brake = float(np.clip(u, -1.0, 1.0))

        # 5-substep linear steering ramp for actuator continuity
        substeps: List[float] = [
            round(prev_s + (steering - prev_s) * (k / 5.0), 4) for k in range(1, 6)
        ]

        gated_info: Dict[str, Any] = {
            "maneuver": str(decision.get("maneuver", "keep_lane")),
            "speed_level": int(decision.get("speed_level", 2)),
            "target_speed_kmh": round(float(target_speed_kmh), 2),
            "shield_triggered": False,
            "shield_reason": "pure_e2e_no_shield",
            "mpc_feedforward": round(float(curvature), 5),
            "lookahead_point": [round(x_d, 3), round(y_d, 3)],
            "steer_ramp_substeps": substeps,
            "carla_control": self.to_carla_control(steering, throttle_brake),
        }
        return (steering, throttle_brake), gated_info


__all__ = ["PureE2ETrajectoryTracker"]
