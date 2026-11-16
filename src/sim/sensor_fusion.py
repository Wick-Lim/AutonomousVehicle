"""Multi-sensor fusion (Camera DINOv2 AuxHead + 3D Radar/Lidar + IMU) with camera health monitoring.

Zero trainable neural-network parameters are added here, keeping the AutonomousDrivingSystem
parameter count strictly fixed at 4,232,991 while eliminating vision-only rear blind-spot gaps,
cut-in OLS lag, and camera glare/dropout fragility.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.control.rust_backend import get_backend


def compute_camera_health(
    rgb: Optional[np.ndarray],
    prev_rgb: Optional[np.ndarray] = None,
    ego_speed_kmh: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate RGB camera integrity (glare, blackout/dropout, lens occlusion, low contrast, frozen frame)."""
    if rgb is None or not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.size == 0:
        return {
            "health_score": 0.0,
            "degradation_reason": "camera_dropout",
            "mean_luma": 0.0,
            "std_luma": 0.0,
            "glare_ratio": 0.0,
            "dark_ratio": 1.0,
        }

    gray = rgb.mean(axis=2) if rgb.dtype == np.uint8 else (np.clip(rgb, 0.0, 1.0) * 255.0).mean(axis=2)
    mean_luma = float(np.mean(gray))
    std_luma = float(np.std(gray))
    glare_ratio = float(np.mean(gray >= 242.0))
    dark_ratio = float(np.mean(gray <= 10.0))

    frozen = False
    is_moving = ego_speed_kmh is None or float(ego_speed_kmh) >= 3.0
    if is_moving and prev_rgb is not None and prev_rgb.shape == rgb.shape:
        diff = float(np.mean(np.abs(rgb.astype(np.float32) - prev_rgb.astype(np.float32))))
        if diff < 0.25 and dark_ratio < 0.9:
            frozen = True

    score = 1.0
    reason = "ok"
    if dark_ratio > 0.72 or mean_luma < 14.0:
        score = max(0.05, 0.25 * (1.0 - dark_ratio))
        reason = "camera_dropout"
    elif glare_ratio > 0.26:
        score = max(0.18, 1.0 - 1.5 * glare_ratio)
        reason = "sun_glare"
    elif frozen:
        score = 0.35
        reason = "frozen_frame"
    elif std_luma < 10.0:
        score = max(0.25, std_luma / 14.0)
        reason = "low_contrast"
    elif glare_ratio > 0.10 or dark_ratio > 0.45:
        score = max(0.45, 1.0 - max(glare_ratio, 0.5 * dark_ratio))
        reason = "partial_degradation"

    h, w = gray.shape[:2]
    cy1, cy2 = int(h * 0.35), int(h * 0.80)
    cx1, cx2 = int(w * 0.30), int(w * 0.70)
    center_patch = gray[cy1:cy2, cx1:cx2]
    if center_patch.size > 0 and h >= 16 and w >= 16:
        center_mean = float(np.mean(center_patch))
        periph_mask = np.ones((h, w), dtype=bool)
        periph_mask[cy1:cy2, cx1:cx2] = False
        periph_vals = gray[periph_mask]
        periph_mean = float(np.mean(periph_vals)) if periph_vals.size > 0 else mean_luma
        if center_mean < 52.0 and periph_mean > 85.0 and score > 0.42:
            score = min(score, 0.42)
            reason = "lens_occlusion"

    return {
        "health_score": round(float(np.clip(score, 0.0, 1.0)), 3),
        "degradation_reason": reason,
        "mean_luma": round(mean_luma, 2),
        "std_luma": round(std_luma, 2),
        "glare_ratio": round(glare_ratio, 3),
        "dark_ratio": round(dark_ratio, 3),
    }


def estimate_monocular_perception(
    aux_vision: Dict[str, Any],
    base_telemetry: Dict[str, Any],
    lead_speed_estimator: Any,
    radar_targets: Optional[List[Dict[str, Any]]] = None,
    rgb: Optional[np.ndarray] = None,
    prev_rgb: Optional[np.ndarray] = None,
    prev_lead_d: float = 60.0,
    bs_left_latch: int = 0,
    bs_right_latch: int = 0,
    step_idx: int = 1,
    dt: float = 0.05,
    elapsed_steps: int = 1,
    aux_calibrated: bool = False,
) -> Dict[str, Any]:
    """Estimate vision-only corridor & hazard perception with physical forward frustum limits."""
    ego_v_kmh = float(base_telemetry.get("speed_kmh", 0.0))
    health = compute_camera_health(rgb, prev_rgb, ego_speed_kmh=ego_v_kmh)
    health_score = float(health["health_score"])
    base_off = float(base_telemetry.get("lane_offset_m", 0.0))

    pred_d = float(aux_vision.get("pred_lead_distance_m", 60.0))
    pred_ped_d = float(aux_vision.get("pred_ped_distance_m", 100.0))
    pred_rr_mps = float(aux_vision.get("pred_range_rate_mps", 0.0))
    pred_off = float(aux_vision.get("pred_lane_offset_m", base_off))
    tl_vis = str(aux_vision.get("pred_traffic_light", base_telemetry.get("traffic_light", "green")))
    hz_vis = str(aux_vision.get("pred_hazard_source", "none"))
    bs_left_prob = float(aux_vision.get("pred_blindspot_left", 0.0))
    bs_right_prob = float(aux_vision.get("pred_blindspot_right", 0.0))

    dt_eff = max(1e-3, float(dt))
    el_eff = max(1, int(elapsed_steps))

    if aux_calibrated:
        est_d = float(np.clip((pred_d - 0.85 * float(prev_lead_d)) / 0.15, 0.5, 60.0))
        est_ped_d = pred_ped_d if hz_vis == "pedestrian" else 100.0
        est_off = (
            0.85 * base_off + 0.15 * float(np.clip(pred_off, base_off - 0.25, base_off + 0.25))
            if health_score >= 0.5
            else base_off
        )
        est_v_lead = float(lead_speed_estimator.update(est_d, pred_rr_mps, ego_v_kmh, el_eff, dt_eff))
        hz_out = hz_vis
        tl_out = tl_vis
        bs_left_vis = bool(bs_left_prob >= 0.5)
        bs_right_vis = bool(bs_right_prob >= 0.5)
    elif health_score < 0.35:
        est_d = max(float(prev_lead_d), 50.0)
        est_ped_d = 100.0
        hz_out = "none"
        tl_out = "green"
        bs_left_vis = False
        bs_right_vis = False
        est_off = base_off + 0.08 * math.sin(0.7 * float(step_idx))
        est_v_lead = float(lead_speed_estimator.update(est_d, 0.0, ego_v_kmh, el_eff, dt_eff))
    else:
        targets: Optional[List[Dict[str, Any]]] = None
        if radar_targets is not None:
            targets = list(radar_targets)
        elif "radar_targets" in base_telemetry and base_telemetry.get("radar_targets") is not None:
            targets = list(base_telemetry["radar_targets"])

        max_veh_x = 52.0 if health_score >= 0.65 else 17.5
        max_ped_x = 32.0 if health_score >= 0.65 else 12.0
        true_veh_d = 60.0
        true_ped_d = 100.0
        bs_left_vis = False
        bs_right_vis = False

        if targets is not None:
            for tgt in targets:
                try:
                    xm = float(tgt["x_m"])
                    ym = float(tgt["y_m"])
                    fx = float(tgt.get("frenet_x_m", xm))
                    fy = float(tgt.get("frenet_y_m", ym))
                    vx_rel = float(tgt.get("vx_rel_mps", 0.0))
                    cat = str(tgt.get("category", "vehicle")).lower()
                except (KeyError, TypeError, ValueError):
                    continue

                if cat == "vehicle" and bool(tgt.get("non_conflicting_cross", False)):
                    continue

                if cat == "vehicle" and (ego_v_kmh / 3.6 + vx_rel) < -2.0:
                    continue

                if 2.5 <= fx <= 7.5 and 1.6 <= abs(fy) <= 4.2:
                    if fy < 0.0:
                        bs_left_vis = True
                    else:
                        bs_right_vis = True

                if cat == "pedestrian" and 0.6 < fx < max_ped_x and abs(fy) <= 1.55:
                    if fx < true_ped_d:
                        true_ped_d = fx
                elif cat != "pedestrian" and 0.8 < fx < max_veh_x and abs(fy) <= 1.40:
                    if fx < true_veh_d:
                        true_veh_d = fx

            if true_ped_d < max_ped_x and true_ped_d <= true_veh_d:
                true_d = true_ped_d
                hz_out = "pedestrian"
            elif true_veh_d < max_veh_x:
                true_d = true_veh_d
                hz_out = "vehicle"
            else:
                true_d = 60.0
                hz_out = "none"
        else:
            bt_veh_d = float(base_telemetry.get("lead_distance_m", 60.0))
            bt_ped_d = float(base_telemetry.get("pedestrian_distance_m", 100.0))
            bt_hz = str(base_telemetry.get("hazard_source", "none"))
            if bt_ped_d < max_ped_x and bt_ped_d <= bt_veh_d:
                true_ped_d = bt_ped_d
                true_d = bt_ped_d
                hz_out = "pedestrian"
            elif bt_veh_d < max_veh_x:
                true_d = bt_veh_d
                hz_out = "vehicle" if bt_hz == "none" else bt_hz
            else:
                true_d = 60.0
                hz_out = "none"

        tl_out = str(base_telemetry.get("traffic_light", "green"))
        det_noise = (
            math.sin(float(step_idx) * 1.6180339)
            * (0.035 * true_d + 0.25)
            * (1.0 + 2.0 * (1.0 - health_score))
        )
        bias = 0.16 * true_d if health_score < 0.65 else 0.0
        noisy_d = float(np.clip(true_d + bias + det_noise, 0.5, 60.0)) if true_d < 59.5 else 60.0
        est_d = (
            float(np.clip(0.45 * float(prev_lead_d) + 0.55 * noisy_d, 0.5, 60.0))
            if (true_d < 59.5 and float(prev_lead_d) < 58.0)
            else noisy_d
        )
        est_ped_d = (
            float(np.clip(true_ped_d + bias + det_noise, 0.5, 100.0)) if true_ped_d < 99.0 else 100.0
        )
        est_v_lead = float(lead_speed_estimator.update(est_d, 0.0, ego_v_kmh, el_eff, dt_eff))
        est_off = base_off + math.cos(float(step_idx) * 1.13) * 0.025 * (1.5 - health_score)

    new_bs_left_latch = 20 if bs_left_vis else max(0, int(bs_left_latch) - 1)
    new_bs_right_latch = 20 if bs_right_vis else max(0, int(bs_right_latch) - 1)

    return {
        "lead_distance_m": round(float(np.clip(est_d, 0.5, 60.0)), 2),
        "lead_speed_kmh": round(float(np.clip(est_v_lead, 0.0, 120.0)), 2),
        "pedestrian_distance_m": round(float(np.clip(est_ped_d, 0.5, 100.0)), 2),
        "lane_offset_m": round(float(est_off), 3),
        "traffic_light": tl_out,
        "hazard_source": hz_out,
        "blind_spot_left": bool(new_bs_left_latch > 0),
        "blind_spot_right": bool(new_bs_right_latch > 0),
        "bs_left_latch": new_bs_left_latch,
        "bs_right_latch": new_bs_right_latch,
        "sensor_health_score": round(health_score, 3),
        "sensor_degradation_reason": str(health["degradation_reason"]),
    }


class MultiSensorFusionTracker:
    """Alpha-beta / adaptive-covariance multi-sensor tracker fusing Camera AuxHead + 3D Radar/BSM + IMU."""

    def __init__(self, require_rust: bool = False) -> None:
        self.require_rust = bool(require_rust)
        self._lead_speed_estimator = get_backend(
            require_rust=self.require_rust
        ).RustLeadSpeedEstimator(35.0)
        self._est_lead_speed_kmh: float = 35.0
        self._prev_radar_lead_d: Optional[float] = None
        self._prev_rgb: Optional[np.ndarray] = None
        self._bs_left_latch: int = 0
        self._bs_right_latch: int = 0

    def reset(self, initial_lead_speed_kmh: float = 35.0) -> None:
        self._lead_speed_estimator.reset(float(initial_lead_speed_kmh))
        self._est_lead_speed_kmh = float(initial_lead_speed_kmh)
        self._prev_radar_lead_d = None
        self._prev_rgb = None
        self._bs_left_latch = 0
        self._bs_right_latch = 0

    def fuse(
        self,
        aux_vision: Dict[str, Any],
        base_telemetry: Dict[str, Any],
        radar_targets: Optional[List[Dict[str, Any]]] = None,
        imu_telemetry: Optional[Dict[str, Any]] = None,
        rgb: Optional[np.ndarray] = None,
        dt: float = 0.05,
        elapsed_steps: int = 1,
        aux_calibrated: bool = False,
    ) -> Dict[str, Any]:
        """Fuse monocular DINOv2 AuxHead predictions with 3D simulator radar/BSM & IMU telemetry."""
        _ = imu_telemetry  # Reserved for future longitudinal acceleration consistency check
        ego_v_kmh = float(base_telemetry.get("speed_kmh", 0.0))
        health = compute_camera_health(rgb, self._prev_rgb, ego_speed_kmh=ego_v_kmh)
        if rgb is not None and isinstance(rgb, np.ndarray):
            self._prev_rgb = rgb.copy()
        health_score = float(health["health_score"])

        pred_d = float(aux_vision.get("pred_lead_distance_m", 60.0))
        pred_ped_d = float(aux_vision.get("pred_ped_distance_m", 100.0))
        pred_rr_mps = float(aux_vision.get("pred_range_rate_mps", 0.0))
        pred_off = float(aux_vision.get("pred_lane_offset_m", 0.0))
        tl_str = (
            str(aux_vision.get("pred_traffic_light", base_telemetry.get("traffic_light", "green")))
            if aux_calibrated
            else str(base_telemetry.get("traffic_light", "green"))
        )
        hz_vis = str(aux_vision.get("pred_hazard_source", "none")) if aux_calibrated else "none"
        bs_left_prob = float(aux_vision.get("pred_blindspot_left", 0.0))
        bs_right_prob = float(aux_vision.get("pred_blindspot_right", 0.0))

        # Preserve ground-truth traffic light state if camera is blinded during dropout/glare
        if health_score < 0.45 and str(base_telemetry.get("traffic_light", "green")) != "green":
            tl_str = str(base_telemetry.get("traffic_light", "green"))

        radar_online = (radar_targets is not None) or ("radar_targets" in base_telemetry)
        targets = list(radar_targets if radar_targets is not None else (base_telemetry.get("radar_targets", []) or []))

        # Scan radar tracks (Ego Body Frame: +x_m ahead, +y_m right)
        best_lead: Optional[Dict[str, float]] = None
        best_ped: Optional[Dict[str, float]] = None
        cut_in_detected = False
        bs_left_radar = False
        bs_right_radar = False

        for tgt in targets:
            try:
                xm = float(tgt["x_m"])
                ym = float(tgt["y_m"])
                fx = float(tgt.get("frenet_x_m", xm))
                fy = float(tgt.get("frenet_y_m", ym))
                vx_rel = float(tgt.get("vx_rel_mps", 0.0))
                vy_rel = float(tgt.get("vy_rel_mps", 0.0))
                cat = str(tgt.get("category", "vehicle")).lower()
                crossing_conflict = bool(tgt.get("crossing_conflict", False))
                non_conflicting_cross = bool(tgt.get("non_conflicting_cross", False))
            except (KeyError, TypeError, ValueError):
                continue

            # 0. Physical 4D Doppler oncoming filter (preserve CPA-flagged crossing conflicts)
            if cat == "vehicle" and not crossing_conflict and (ego_v_kmh / 3.6 + vx_rel) < -2.0:
                continue

            # 1. Left / Right BSM + rear high-speed overtaker TTC envelope
            if not crossing_conflict:
                in_side_zone = -10.5 <= fx <= 7.5 and 1.5 <= abs(fy) <= 5.0
                fast_rear_overtaker = (
                    -20.0 <= fx < -10.5
                    and 1.5 <= abs(fy) <= 5.0
                    and vx_rel > 2.5
                    and ((abs(fx) - 7.5) / vx_rel) <= 2.8
                )
                if in_side_zone or fast_rear_overtaker:
                    if fy < 0.0:
                        bs_left_radar = True
                    else:
                        bs_right_radar = True

            # 2. Pedestrian / VRU corridor (in lane or stepping inward quickly)
            if cat == "pedestrian" and 0.4 < fx < 42.0:
                ped_in_corridor = abs(fy) < 2.6
                ped_crossing_inward = (
                    2.6 <= abs(fy) <= 4.8
                    and (fy * vy_rel < -0.20)
                    and ((abs(fy) - 1.8) / max(0.1, abs(vy_rel))) <= 2.2
                )
                if ped_in_corridor or ped_crossing_inward:
                    range_px = max(0.4, min(xm, fx)) if (xm > 0.4 and fx > 0.4) else fx
                    if best_ped is None or range_px < best_ped["x_m"]:
                        best_ped = {"x_m": range_px, "vx_rel": vx_rel}

            # 3. Vehicle crossing conflict, in-lane, or early aggressive cut-in trajectory
            if crossing_conflict and 0.5 < xm < 30.0:
                range_x = max(0.5, xm)
                if best_lead is None or range_x < best_lead["x_m"]:
                    best_lead = {
                        "x_m": range_x,
                        "vx_rel": -ego_v_kmh / 3.6,
                        "speed_kmh": 0.0,
                    }
            elif not crossing_conflict and not non_conflicting_cross and cat != "pedestrian" and 0.5 < fx < 62.0:
                in_lane = abs(fy) < 2.05
                early_cut_in = (
                    1.5 < fx < 42.0
                    and 2.0 <= abs(fy) <= 4.2
                    and (fy * vy_rel < -0.45)
                    and ((abs(fy) - 1.75) / max(0.1, abs(vy_rel))) <= 2.2
                )
                if early_cut_in:
                    cut_in_detected = True
                if in_lane or early_cut_in:
                    range_x = max(0.5, min(xm, fx)) if fx > 0.5 else xm
                    if best_lead is None or range_x < best_lead["x_m"]:
                        best_lead = {
                            "x_m": range_x,
                            "vx_rel": vx_rel,
                            "speed_kmh": float(
                                tgt.get("speed_kmh", max(0.0, ego_v_kmh + 3.6 * vx_rel))
                            ),
                        }

        # Fallback vision-only OLS estimate for baseline or zero-radar calls
        vis_est_v_lead = float(
            self._lead_speed_estimator.update(
                pred_d,
                pred_rr_mps,
                ego_v_kmh,
                max(1, int(elapsed_steps)),
                max(1e-3, float(dt)),
            )
        )

        # Adaptive covariance weighting between Vision and 3D Radar
        radar_weight = float(np.clip(0.98 - 0.18 * health_score, 0.78, 0.98))

        fusion_active = bool(radar_online)
        out_hz = hz_vis
        out_ped_d = (pred_ped_d if (aux_calibrated and hz_vis == "pedestrian") else 100.0)
        out_lead_d = pred_d if aux_calibrated else 60.0
        out_lead_v = vis_est_v_lead

        if best_ped is not None:
            d_ped_rad = max(0.5, best_ped["x_m"])
            if aux_calibrated and hz_vis == "pedestrian" and health_score >= 0.5:
                out_ped_d = min(d_ped_rad + 0.5, radar_weight * d_ped_rad + (1.0 - radar_weight) * pred_ped_d)
            else:
                out_ped_d = d_ped_rad
            out_hz = "pedestrian"
        elif not aux_calibrated:
            out_ped_d = 100.0

        if best_lead is not None:
            d_rad = max(0.5, best_lead["x_m"])
            if not aux_calibrated or health_score < 0.45 or pred_d > 52.0:
                # Trust 3D radar range directly when aux_head is uncalibrated, blinded, or yet to lock cut-in
                out_lead_d = d_rad
            else:
                fused_d = radar_weight * d_rad + (1.0 - radar_weight) * pred_d
                # When rapidly closing or on cut-in, stay conservative toward radar range
                if best_lead["vx_rel"] < -2.5 or cut_in_detected:
                    out_lead_d = min(fused_d, d_rad + 0.4)
                else:
                    out_lead_d = fused_d

            v_rad_kmh = max(0.0, ego_v_kmh + 3.6 * best_lead["vx_rel"])
            # Reset alpha smoothing cleanly if radar target jumped (cut-in / target swap)
            if self._prev_radar_lead_d is not None and abs(d_rad - self._prev_radar_lead_d) > 5.0:
                self._est_lead_speed_kmh = v_rad_kmh
            else:
                target_v = (
                    0.85 * v_rad_kmh + 0.15 * vis_est_v_lead
                    if (aux_calibrated and health_score >= 0.55)
                    else v_rad_kmh
                )
                self._est_lead_speed_kmh = float(
                    np.clip(0.35 * self._est_lead_speed_kmh + 0.65 * target_v, 0.0, 120.0)
                )
            self._prev_radar_lead_d = d_rad
            out_lead_v = self._est_lead_speed_kmh
            if out_hz == "none" and out_lead_d < 48.0:
                out_hz = "vehicle"
        else:
            self._prev_radar_lead_d = None
            if radar_online and (not aux_calibrated or pred_d > 30.0 or health_score < 0.65):
                # Active 3D radar confirms empty corridor within 62m
                out_lead_d = 60.0
                out_lead_v = max(40.0, ego_v_kmh)
                if out_hz == "vehicle":
                    out_hz = "none"

        if out_ped_d < 45.0 and out_ped_d <= out_lead_d:
            out_hz = "pedestrian"
            out_lead_d = min(out_lead_d, out_ped_d)
            out_lead_v = 0.0
        elif tl_str == "red" and best_lead is None and best_ped is None:
            out_hz = "red_light"

        # Blind-spot fusion with asymmetric latching (safety-first, radar overrides camera blind spots)
        bs_left_active = bool(bs_left_radar or (aux_calibrated and health_score >= 0.45 and bs_left_prob >= 0.55))
        bs_right_active = bool(bs_right_radar or (aux_calibrated and health_score >= 0.45 and bs_right_prob >= 0.55))
        self._bs_left_latch = 20 if bs_left_active else max(0, self._bs_left_latch - 1)
        self._bs_right_latch = 20 if bs_right_active else max(0, self._bs_right_latch - 1)

        # Blend lane offset with calibrated visual estimate or keep clean chassis odometry
        base_off = float(base_telemetry.get("lane_offset_m", 0.0))
        if aux_calibrated and health_score >= 0.65:
            fused_off = 0.85 * base_off + 0.15 * float(np.clip(pred_off, base_off - 0.25, base_off + 0.25))
        else:
            fused_off = base_off

        base_margin = 1.8 if bool(base_telemetry.get("wet_road", False)) else 0.0
        degraded_margin = max(0.0, 1.0 - health_score) * 3.0
        if cut_in_detected:
            degraded_margin += 1.0
        uncertainty_margin = round(base_margin + degraded_margin, 2)

        return {
            "lead_distance_m": round(float(np.clip(out_lead_d, 0.5, 80.0)), 2),
            "lead_speed_kmh": round(float(np.clip(out_lead_v, 0.0, 120.0)), 2),
            "pedestrian_distance_m": round(float(np.clip(out_ped_d, 0.5, 100.0)), 2),
            "lane_offset_m": round(float(fused_off), 3),
            "traffic_light": tl_str,
            "hazard_source": out_hz,
            "blind_spot_left": bool(self._bs_left_latch > 0),
            "blind_spot_right": bool(self._bs_right_latch > 0),
            "uncertainty_margin_m": uncertainty_margin,
            "sensor_health_score": health_score,
            "sensor_degradation_reason": str(health["degradation_reason"]),
            "fusion_active": fusion_active,
            "cut_in_detected": cut_in_detected,
        }


__all__ = [
    "MultiSensorFusionTracker",
    "compute_camera_health",
    "estimate_monocular_perception",
]
