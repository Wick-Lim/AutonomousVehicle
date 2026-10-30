"""Safety Shield & Lateral/Longitudinal PID Controllers backed by native Rust (`PyO3`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.control.rust_backend import rust_core_module


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    integral_limit: float = 5.0


class ScalarPID:
    """Discrete-time PID controller with anti-windup clamping backed by `RustScalarPID`."""

    def __init__(self, gains: PIDGains) -> None:
        if rust_core_module is None:
            raise RuntimeError(
                "Rust extension `autonomous_vehicle_rust` is required. Build `rust_core` with `cargo build --release`."
            )
        self.gains = gains
        self._rust_pid = rust_core_module.RustScalarPID(
            float(gains.kp),
            float(gains.ki),
            float(gains.kd),
            float(gains.integral_limit),
        )

    @property
    def integral(self) -> float:
        return float(self._rust_pid.integral)

    @integral.setter
    def integral(self, val: float) -> None:
        self._rust_pid.integral = float(val)

    @property
    def prev_error(self) -> Optional[float]:
        return self._rust_pid.prev_error

    @prev_error.setter
    def prev_error(self, val: Optional[float]) -> None:
        self._rust_pid.prev_error = None if val is None else float(val)

    def reset(self) -> None:
        self._rust_pid.reset()

    def step(self, error: float, dt: float = 0.05) -> float:
        return float(self._rust_pid.step(float(error), float(dt)))


class SafetyShieldPID:
    """Calibrated Safety Gating Shield + Lateral/Longitudinal PID Controller backed by `RustSafetyShieldPID`.

    Enforces Section 3.3 Safety Gating Policy in native Rust (`autonomous_vehicle_rust`):
      - Rule 1 (AEB Priority): If `emergency_brake >= 0.70`, red light/pedestrian, or kinematic closing-rate AEB,
        override all maneuvers and immediately execute maximum braking (`throttle_brake = -1.0`).
      - Rule 2 (Confidence Fallback): If `maneuver_confidence < 0.80`, reject any lane-change
        command (`change_left` / `change_right`), force `keep_lane`, and step down
        `speed_level` by 1 level (`max(1, speed_level - 1)`).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_mock: bool = False,
    ) -> None:
        if rust_core_module is None:
            raise RuntimeError(
                "Rust extension `autonomous_vehicle_rust` is required. Build `rust_core` with `cargo build --release`."
            )
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
        self.max_steer_rate_per_sec: float = float(
            lat_cfg.get("max_steer_rate_per_sec", 3.2)
        )
        self.steer_smoothing_alpha: float = float(
            lat_cfg.get("steer_smoothing_alpha", 0.72)
        )
        self.current_target_offset_m: float = 0.0

        lat_kp = float(lat_cfg.get("kp", 0.45))
        lat_ki = float(lat_cfg.get("ki", 0.01))
        lat_kd = float(lat_cfg.get("kd", 0.18))
        lon_kp = float(lon_cfg.get("kp", 0.08))
        lon_ki = float(lon_cfg.get("ki", 0.005))
        lon_kd = float(lon_cfg.get("kd", 0.02))

        self.lateral_pid = ScalarPID(PIDGains(kp=lat_kp, ki=lat_ki, kd=lat_kd))
        self.longitudinal_pid = ScalarPID(PIDGains(kp=lon_kp, ki=lon_ki, kd=lon_kd))

        self._rust_shield = rust_core_module.RustSafetyShieldPID(
            self.use_mock,
            self.aeb_threshold,
            self.confidence_threshold,
            self.lane_width,
            self.heading_weight,
            self.max_steer_rate_per_sec,
            self.steer_smoothing_alpha,
            lat_kp,
            lat_ki,
            lat_kd,
            lon_kp,
            lon_ki,
            lon_kd,
        )

    @property
    def stop_hold_active(self) -> bool:
        return bool(self._rust_shield.stop_hold_active)

    @stop_hold_active.setter
    def stop_hold_active(self, val: bool) -> None:
        self._rust_shield.stop_hold_active = bool(val)

    @property
    def current_steering(self) -> float:
        return float(self._rust_shield.current_steering)

    @current_steering.setter
    def current_steering(self, val: float) -> None:
        self._rust_shield.current_steering = float(val)

    @property
    def current_throttle_brake(self) -> float:
        return float(self._rust_shield.current_throttle_brake)

    @current_throttle_brake.setter
    def current_throttle_brake(self, val: float) -> None:
        self._rust_shield.current_throttle_brake = float(val)

    def reset(self) -> None:
        """Reset PID controller internal states at episode start."""
        self.lateral_pid.reset()
        self.longitudinal_pid.reset()
        self.current_target_offset_m = 0.0
        self._rust_shield.reset()

    @staticmethod
    def compute_kinematic_aeb_distance(
        current_speed_kmh: float,
        lead_speed_kmh: float,
        a_max_mps2: float = 6.0,
        uncertainty_margin_m: float = 0.0,
    ) -> float:
        """Velocity-, friction-, and closing-rate-dependent AEB boundary d_AEB(v_ego, v_lead) via Rust core."""
        assert rust_core_module is not None
        return float(
            rust_core_module.compute_kinematic_aeb_distance(
                float(current_speed_kmh),
                float(lead_speed_kmh),
                float(a_max_mps2),
                float(uncertainty_margin_m),
            )
        )

    def apply_safety_shield(
        self,
        decision: Dict[str, Any],
        telemetry: Dict[str, Any],
        shield_enabled: bool = True,
    ) -> Dict[str, Any]:
        """Apply calibrated safety gating rules to Laya's raw decision output via Rust core."""
        return dict(
            self._rust_shield.apply_safety_shield(
                decision, telemetry, bool(shield_enabled)
            )
        )

    def compute_mpc_feedforward(
        self,
        decision: Dict[str, Any],
        telemetry: Dict[str, Any],
    ) -> float:
        """Compute Kinematic Bicycle MPC feedforward steering command (wheelbase L = 2.85m) via Rust core."""
        assert rust_core_module is not None
        road_curv = float(telemetry.get("road_curvature", 0.0))
        lane_offset_m = float(telemetry.get("lane_offset_m", 0.0))
        pred_traj = decision.get("predicted_trajectory")
        if isinstance(pred_traj, list) and len(pred_traj) >= 3:
            return float(
                rust_core_module.compute_mpc_feedforward(
                    road_curv,
                    lane_offset_m,
                    float(pred_traj[2][0]),
                    float(pred_traj[2][1]),
                    True,
                )
            )
        return float(
            rust_core_module.compute_mpc_feedforward(
                road_curv, lane_offset_m, 0.0, 0.0, False
            )
        )

    def compute_control(
        self,
        decision: Dict[str, Any],
        telemetry: Dict[str, Any],
        dt: float = 0.05,
        shield_enabled: bool = True,
    ) -> Tuple[Tuple[float, float], Dict[str, Any]]:
        """Compute physical control [steering, throttle_brake] in [-1.0, 1.0] via Rust core."""
        (steering, throttle_brake), gated_dict = self._rust_shield.compute_control(
            decision, telemetry, float(dt), bool(shield_enabled)
        )
        gated = dict(gated_dict)
        if "carla_control" in gated and not isinstance(gated["carla_control"], dict):
            gated["carla_control"] = dict(gated["carla_control"])
        return (float(steering), float(throttle_brake)), gated

    @staticmethod
    def to_carla_control(steering: float, throttle_brake: float) -> Dict[str, Any]:
        """Convert `[steering, throttle_brake]` in `[-1, 1]` to CARLA `VehicleControl` dict."""
        steer = max(-1.0, min(1.0, float(steering)))
        tb = max(-1.0, min(1.0, float(throttle_brake)))
        return {
            "steer": round(steer, 4),
            "throttle": round(tb if tb >= 0.0 else 0.0, 4),
            "brake": round(-tb if tb < 0.0 else 0.0, 4),
            "hand_brake": False,
            "reverse": False,
        }
