"""Vehicle actuator dynamics (1st-order servo lag, rate limits, deadzone) & SOTIF sensor fault injection."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

VALID_FAULT_MODES: Tuple[str, ...] = (
    "none",
    "rain_streaks",
    "sun_glare",
    "lens_occlusion",
    "camera_dropout",
    "radar_noise",
    "hd_map_dropout",
    "combined_sotif",
)


class ActuatorDynamicsFilter:
    """Realistic steer-by-wire / drive-by-wire 1st-order servo lag, rate limiter & pedal deadzone."""

    def __init__(
        self,
        steer_tau_s: float = 0.08,
        max_steer_rate_rps: float = 2.4,
        steer_deadband: float = 0.006,
        throttle_tau_s: float = 0.10,
        brake_tau_s: float = 0.08,
        throttle_deadzone: float = 0.015,
        road_grade_rad: float = 0.0,
        max_throttle_rate_per_s: float = 2.4,
        max_brake_rate_per_s: float = 2.5,
        max_brake_cmd: float = 0.54,
    ) -> None:
        self.steer_tau_s = float(steer_tau_s)
        self.max_steer_rate_rps = float(max_steer_rate_rps)
        self.steer_deadband = float(steer_deadband)
        self.throttle_tau_s = float(throttle_tau_s)
        self.brake_tau_s = float(brake_tau_s)
        self.throttle_deadzone = float(throttle_deadzone)
        self.road_grade_rad = float(road_grade_rad)
        self.max_throttle_rate_per_s = float(max_throttle_rate_per_s)
        self.max_brake_rate_per_s = float(max_brake_rate_per_s)
        self.max_brake_cmd = float(max_brake_cmd)

        self._prev_steer: float = 0.0
        self._prev_tb: float = 0.0

    def reset(self) -> None:
        self._prev_steer = 0.0
        self._prev_tb = 0.0

    def apply(
        self,
        steering: float,
        throttle_brake: float,
        speed_kmh: float = 0.0,
        dt: float = 0.05,
    ) -> Tuple[Tuple[float, float], Dict[str, float]]:
        """Filter raw [steering, throttle_brake] through physical actuator dynamics."""
        dt_eff = max(1e-3, float(dt))
        cmd_steer = float(np.clip(steering, -1.0, 1.0))
        cmd_tb = float(np.clip(throttle_brake, -1.0, 1.0))

        # 1. Steering deadband + 1st-order servo lag + speed-dependent rate limit
        if abs(cmd_steer) < self.steer_deadband:
            cmd_steer = 0.0
        alpha_s = dt_eff / (self.steer_tau_s + dt_eff)
        lag_steer = self._prev_steer + alpha_s * (cmd_steer - self._prev_steer)

        speed_factor = 1.0 / (1.0 + 0.008 * max(0.0, float(speed_kmh)))
        max_delta = self.max_steer_rate_rps * speed_factor * dt_eff
        delta_s = float(np.clip(lag_steer - self._prev_steer, -max_delta, max_delta))
        eff_steer = float(np.clip(self._prev_steer + delta_s, -1.0, 1.0))

        # 2. Throttle deadzone + speed-feathered tire-limited brake saturation + servo lag & rate limit
        if 0.0 < cmd_tb < self.throttle_deadzone:
            cmd_tb = 0.0
        if cmd_tb < 0.0:
            feather = (
                1.0
                if float(speed_kmh) >= 6.0
                else (0.10 + 0.90 * (max(0.0, float(speed_kmh)) / 6.0))
            )
            cmd_tb = max(cmd_tb, -self.max_brake_cmd * feather)
        tau_tb = self.brake_tau_s if cmd_tb < self._prev_tb or cmd_tb < 0.0 else self.throttle_tau_s
        alpha_tb = dt_eff / (tau_tb + dt_eff)
        lag_tb = self._prev_tb + alpha_tb * (cmd_tb - self._prev_tb)
        delta_tb = float(
            np.clip(
                lag_tb - self._prev_tb,
                -self.max_brake_rate_per_s * dt_eff,
                self.max_throttle_rate_per_s * dt_eff,
            )
        )
        eff_tb = float(np.clip(self._prev_tb + delta_tb, -1.0, 1.0))

        if abs(self.road_grade_rad) > 1e-4:
            # Uphill positive grade opposes forward acceleration
            grade_offset = -0.25 * math.sin(self.road_grade_rad)
            eff_tb = float(np.clip(eff_tb + grade_offset, -1.0, 1.0))

        self._prev_steer = eff_steer
        self._prev_tb = eff_tb
        diag = {
            "raw_steer": round(cmd_steer, 4),
            "effective_steer": round(eff_steer, 4),
            "raw_throttle_brake": round(cmd_tb, 4),
            "effective_throttle_brake": round(eff_tb, 4),
            "steer_lag_error": round(abs(cmd_steer - eff_steer), 4),
        }
        return (eff_steer, eff_tb), diag


class SOTIFSensorFaultInjector:
    """Configurable camera & radar SOTIF degradation injector (rain, glare, dirt, blackout, radar noise)."""

    def __init__(self, fault_mode: str = "none", seed: int = 42) -> None:
        self.fault_mode: str = str(fault_mode or "none").strip().lower()
        self.seed: int = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def set_mode(self, fault_mode: str) -> None:
        m = str(fault_mode or "none").strip().lower()
        self.fault_mode = m if m in VALID_FAULT_MODES else "none"

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def is_hd_map_dropped(self, step_idx: int) -> bool:
        """Return True when HD-map/GPS centerline localization is unavailable."""
        _ = step_idx
        return self.fault_mode == "hd_map_dropout"

    def inject_camera(self, rgb: np.ndarray, step_idx: int) -> np.ndarray:
        """Apply visual SOTIF fault (non-mutating; always returns HxWx3 uint8 RGB)."""
        if self.fault_mode in ("none", "radar_noise", "hd_map_dropout"):
            return rgb

        out = rgb.copy()
        h, w = out.shape[:2]
        mode = self.fault_mode

        if mode in ("camera_dropout", "combined_sotif"):
            # In combined_sotif, trigger blackout at step 24 and across the wet-road emergency braking window (27..42)
            if (
                mode == "camera_dropout"
                or step_idx == 24
                or (27 <= step_idx <= 42)
                or (step_idx > 42 and (step_idx % 22) in range(6, 13))
            ):
                out[:] = (out.astype(np.float32) * 0.02).astype(np.uint8)
                return out

        if mode in ("rain_streaks", "combined_sotif"):
            overlay = out.copy()
            n_drops = 38 if mode == "rain_streaks" else 48
            for _ in range(n_drops):
                x0 = int(self._rng.integers(0, max(2, w)))
                y0 = int(self._rng.integers(0, max(2, h - 18)))
                length = int(self._rng.integers(12, 28))
                dx = int(self._rng.integers(-4, 5))
                cv2.line(overlay, (x0, y0), (x0 + dx, min(h - 1, y0 + length)), (215, 225, 238), 1)
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
            out = cv2.GaussianBlur(out, (3, 3), 0.7)

        if mode in ("sun_glare", "combined_sotif"):
            yy, xx = np.ogrid[:h, :w]
            cx = int(w * 0.52)
            cy = int(h * 0.24)
            dist_sq = ((xx - cx) ** 2) / max(1.0, (w * 0.26) ** 2) + ((yy - cy) ** 2) / max(1.0, (h * 0.24) ** 2)
            bloom = np.exp(-1.1 * dist_sq)[..., None]
            glare_layer = out.astype(np.float32) + bloom * 235.0
            out = np.clip(glare_layer, 0, 255).astype(np.uint8)

        if mode == "lens_occlusion":
            cx, cy = int(w * 0.50), int(h * 0.56)
            cv2.circle(out, (cx, cy), max(18, int(min(h, w) * 0.22)), (42, 38, 34), -1)
            cv2.circle(out, (cx + 28, cy - 16), max(12, int(min(h, w) * 0.12)), (55, 50, 44), -1)

        return out

    def inject_radar(
        self, radar_targets: List[Dict[str, Any]], step_idx: int
    ) -> List[Dict[str, Any]]:
        """Add realistic Gaussian radar range/Doppler jitter & sparse clutter dropouts."""
        _ = step_idx
        if self.fault_mode not in ("radar_noise", "combined_sotif") or not radar_targets:
            return list(radar_targets)

        p_drop = 0.08 if self.fault_mode == "radar_noise" else 0.05
        sigma_r = 0.22
        sigma_v = 0.16
        noisy: List[Dict[str, Any]] = []
        for tgt in radar_targets:
            if len(radar_targets) > 1 and float(self._rng.random()) < p_drop:
                continue
            item = dict(tgt)
            item["x_m"] = round(float(tgt["x_m"]) + float(self._rng.normal(0.0, sigma_r)), 2)
            item["y_m"] = round(float(tgt["y_m"]) + float(self._rng.normal(0.0, sigma_r * 0.7)), 2)
            item["vx_rel_mps"] = round(float(tgt.get("vx_rel_mps", 0.0)) + float(self._rng.normal(0.0, sigma_v)), 2)
            item["vy_rel_mps"] = round(float(tgt.get("vy_rel_mps", 0.0)) + float(self._rng.normal(0.0, sigma_v * 0.5)), 2)
            noisy.append(item)
        return noisy if noisy else list(radar_targets[:1])


__all__ = [
    "ActuatorDynamicsFilter",
    "SOTIFSensorFaultInjector",
    "VALID_FAULT_MODES",
]
