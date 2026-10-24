"""Safety Shield & Lateral/Longitudinal PID Controllers (Calibrated Safety Gating)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.env.expert_labeler import SPEED_LEVEL_KMH


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    integral_limit: float = 5.0


class ScalarPID:
    """Discrete-time PID controller with anti-windup clamping."""

    def __init__(self, gains: PIDGains) -> None:
        self.gains = gains
        self.integral: float = 0.0
        self.prev_error: Optional[float] = None

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def step(self, error: float, dt: float = 0.05) -> float:
        # Suppress derivative spike when crossing discrete lane boundary
        if self.prev_error is not None and abs(error - self.prev_error) > 1.2:
            derivative = 0.0
            self.integral = 0.0
        else:
            self.integral = float(
                np.clip(
                    self.integral + error * dt,
                    -self.gains.integral_limit,
                    self.gains.integral_limit,
                )
            )
            derivative = (
                0.0
                if self.prev_error is None
                else (error - self.prev_error) / max(1e-4, dt)
            )
        self.prev_error = error
        return (
            self.gains.kp * error
            + self.gains.ki * self.integral
            + self.gains.kd * derivative
        )


class SafetyShieldPID:
    """Calibrated Safety Gating Shield + Lateral/Longitudinal PID Controller.

    Enforces Section 3.3 Safety Gating Policy:
      - Rule 1 (AEB Priority): If `emergency_brake >= 0.70`, override all maneuvers and
        immediately execute maximum braking (`throttle_brake = -1.0`).
      - Rule 2 (Confidence Fallback): If `maneuver_confidence < 0.80`, reject any lane-change
        command (`change_left` / `change_right`), force `keep_lane`, and step down
        `speed_level` by 1 level (`max(0, speed_level - 1)`).
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, use_mock: bool = False
    ) -> None:
        cfg = config or {}
        ctrl_cfg = cfg.get("control", {})
        shield_cfg = ctrl_cfg.get("safety_shield", {})
        lat_cfg = ctrl_cfg.get("lateral_pid", {})
        lon_cfg = ctrl_cfg.get("longitudinal_pid", {})
        env_cfg = cfg.get("env", {})
        self.use_mock: bool = bool(
            use_mock or cfg.get("project", {}).get("use_mock", False)
        )

        self.aeb_threshold: float = float(shield_cfg.get("aeb_threshold", 0.70))
        self.confidence_threshold: float = float(
            shield_cfg.get("confidence_threshold", 0.80)
        )
        self.lane_width: float = float(env_cfg.get("lane_width", 3.5))
        self.heading_weight: float = float(lat_cfg.get("heading_weight", 1.25))

        self.lateral_pid = ScalarPID(
            PIDGains(
                kp=float(lat_cfg.get("kp", 0.45)),
                ki=float(lat_cfg.get("ki", 0.01)),
                kd=float(lat_cfg.get("kd", 0.18)),
            )
        )
        self.longitudinal_pid = ScalarPID(
            PIDGains(
                kp=float(lon_cfg.get("kp", 0.08)),
                ki=float(lon_cfg.get("ki", 0.005)),
                kd=float(lon_cfg.get("kd", 0.02)),
            )
        )

    def reset(self) -> None:
        """Reset PID controller internal states at episode start."""
        self.lateral_pid.reset()
        self.longitudinal_pid.reset()

    def apply_safety_shield(
        self,
        decision: Dict[str, Any],
        telemetry: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply calibrated safety gating rules to Laya's raw decision output."""
        raw_maneuver = str(decision.get("maneuver", "keep_lane"))
        raw_speed_level = int(decision.get("speed_level", 2))
        maneuver_conf = float(decision.get("maneuver_confidence", 1.0))
        aeb_prob = float(decision.get("emergency_brake", 0.0))
        lead_distance_m = float(telemetry.get("lead_distance_m", 60.0))
        lead_speed_kmh = float(telemetry.get("lead_speed_kmh", 60.0))
        current_speed_kmh = float(telemetry.get("speed_kmh", 0.0))
        traffic_light = str(telemetry.get("traffic_light", "green"))
        hazard_source = str(telemetry.get("hazard_source", "none"))

        gated_maneuver = raw_maneuver
        gated_speed_level = raw_speed_level
        shield_triggered = False
        shield_reason = "none"

        # Rule 1: Emergency Brake Priority (emergency_brake >= 0.70, CARLA red light/pedestrian, or physical TTC hazard)
        if (
            aeb_prob >= self.aeb_threshold
            or traffic_light == "red"
            or (hazard_source == "pedestrian" and lead_distance_m <= 16.0)
            or (
                lead_distance_m <= 11.5
                and current_speed_kmh > max(6.0, lead_speed_kmh + 6.0)
            )
        ):
            gated_maneuver = "keep_lane"
            gated_speed_level = 0
            shield_triggered = True
            shield_reason = "rule1_emergency_brake"
        else:
            # Rule 2: Low Confidence Fallback (maneuver_confidence < 0.80)
            if maneuver_conf < self.confidence_threshold:
                if raw_maneuver in ("change_left", "change_right"):
                    gated_maneuver = "keep_lane"
                gated_speed_level = (
                    max(1, raw_speed_level - 1) if raw_speed_level > 0 else 0
                )
                shield_triggered = True
                shield_reason = "rule2_low_confidence_fallback"

            # Physical boundary guard: block lane change if target lane does not exist
            if gated_maneuver == "change_left" and not bool(
                telemetry.get("left_lane_available", True)
            ):
                gated_maneuver = "keep_lane"
                shield_triggered = True
                shield_reason = "boundary_left_unavailable"
            elif gated_maneuver == "change_right" and not bool(
                telemetry.get("right_lane_available", True)
            ):
                gated_maneuver = "keep_lane"
                shield_triggered = True
                shield_reason = "boundary_right_unavailable"

        target_speed_kmh = SPEED_LEVEL_KMH.get(gated_speed_level, 20.0)
        if gated_maneuver == "follow_lead" and gated_speed_level >= 2:
            target_speed_kmh = min(target_speed_kmh, 30.0)
        if lead_distance_m < 52.0 and gated_speed_level > 0:
            # Smoothly harmonize with MetaDrive traffic flow (30 km/h) as lead vehicle approaches
            smooth_cap = 30.0 + max(0.0, lead_distance_m - 30.0) * 1.4
            target_speed_kmh = min(target_speed_kmh, smooth_cap)
        if lead_distance_m < 32.0 and gated_speed_level > 0:
            lead_speed_kmh = float(telemetry.get("lead_speed_kmh", target_speed_kmh))
            v_ego = current_speed_kmh / 3.6
            v_lead = lead_speed_kmh / 3.6
            dv = max(0.0, v_ego - v_lead)
            d_star = 10.0 + v_ego * 1.5 + (v_ego * dv) / (2.0 * 2.236)
            if lead_distance_m < d_star * 1.15:
                safe_follow_speed = max(
                    0.0,
                    min(lead_speed_kmh, (lead_distance_m - 7.5) * 1.8),
                )
                target_speed_kmh = min(target_speed_kmh, safe_follow_speed)

        return {
            "raw_maneuver": raw_maneuver,
            "raw_speed_level": raw_speed_level,
            "maneuver": gated_maneuver,
            "speed_level": gated_speed_level,
            "target_speed_kmh": target_speed_kmh,
            "maneuver_confidence": maneuver_conf,
            "emergency_brake": aeb_prob,
            "shield_triggered": shield_triggered,
            "shield_reason": shield_reason,
        }

    def compute_control(
        self,
        decision: Dict[str, Any],
        telemetry: Dict[str, Any],
        dt: float = 0.05,
    ) -> Tuple[Tuple[float, float], Dict[str, Any]]:
        """Compute physical control [steering, throttle_brake] in [-1.0, 1.0]."""
        gated = self.apply_safety_shield(decision, telemetry)

        current_speed_kmh = float(telemetry.get("speed_kmh", 0.0))
        lane_offset_m = float(telemetry.get("lane_offset_m", 0.0))
        heading_error = float(telemetry.get("heading_error", 0.0))

        # 1. Lateral Target Offset based on gated maneuver
        maneuver = gated["maneuver"]
        if self.use_mock and maneuver == "change_left":
            target_offset_m = self.lane_width * 0.85
        elif self.use_mock and maneuver == "change_right":
            target_offset_m = -self.lane_width * 0.85
        elif (
            not self.use_mock
            and not bool(telemetry.get("left_lane_available", True))
            and bool(telemetry.get("right_lane_available", False))
        ):
            # Keep a 0.18m safety margin away from the left continuous yellow line in lane 0
            target_offset_m = -0.18
        else:
            target_offset_m = 0.0

        lateral_error = (target_offset_m - lane_offset_m) - (
            self.heading_weight * heading_error
        )
        steering = float(np.clip(self.lateral_pid.step(lateral_error, dt=dt), -1.0, 1.0))

        # 2. Longitudinal Control: Rule 1 forces immediate full brake (-1.0)
        if gated["shield_reason"] == "rule1_emergency_brake":
            throttle_brake = -1.0
        else:
            speed_error = float(gated["target_speed_kmh"] - current_speed_kmh)
            throttle_brake = float(
                np.clip(self.longitudinal_pid.step(speed_error, dt=dt), -1.0, 1.0)
            )

        gated["steering"] = steering
        gated["throttle_brake"] = throttle_brake
        gated["carla_control"] = self.to_carla_control(steering, throttle_brake)
        return (steering, throttle_brake), gated

    @staticmethod
    def to_carla_control(steering: float, throttle_brake: float) -> Dict[str, Any]:
        """Convert `[steering, throttle_brake]` in `[-1, 1]` to CARLA `VehicleControl` dict."""
        steer = float(np.clip(steering, -1.0, 1.0))
        tb = float(np.clip(throttle_brake, -1.0, 1.0))
        return {
            "steer": round(steer, 4),
            "throttle": round(tb if tb >= 0.0 else 0.0, 4),
            "brake": round(-tb if tb < 0.0 else 0.0, 4),
            "hand_brake": False,
            "reverse": False,
        }
