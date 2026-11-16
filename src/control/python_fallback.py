"""Exact pure-Python fallback mirror of the `autonomous_vehicle_rust` PyO3 extension."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _round_n(val: float, decimals: int) -> float:
    """Match Rust `(val * factor).round() / factor` (half away from zero)."""
    factor = 10.0 ** decimals
    scaled = val * factor
    if scaled == 0.0:
        return 0.0
    rounded = math.copysign(math.floor(abs(scaled) + 0.5), scaled)
    return rounded / factor


def _clamp(val: float, lo: float, hi: float) -> float:
    return lo if val < lo else (hi if val > hi else val)


def _speed_level_to_kmh(level: int) -> float:
    return {0: 0.0, 1: 20.0, 2: 40.0, 3: 65.0}.get(int(level), 20.0)


def _extract_f64(mapping: Mapping[str, Any], key: str, default: float) -> float:
    val = mapping.get(key, default)
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _extract_f64_2keys(
    mapping: Mapping[str, Any], key1: str, key2: str, default: float
) -> float:
    for key in (key1, key2):
        if key in mapping and mapping[key] is not None and not isinstance(mapping[key], bool):
            try:
                f = float(mapping[key])
                if math.isfinite(f):
                    return f
            except (TypeError, ValueError):
                continue
    return default


def _extract_i64(mapping: Mapping[str, Any], key: str, default: int) -> int:
    val = mapping.get(key, default)
    if val is None or isinstance(val, bool):
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _extract_bool(mapping: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in mapping or mapping[key] is None:
        return default
    return bool(mapping[key])


def _extract_str(mapping: Mapping[str, Any], key: str, default: str) -> str:
    val = mapping.get(key, default)
    if val is None:
        return default
    return str(val)


def _extract_wp2(decision: Mapping[str, Any]) -> Tuple[float, float, bool]:
    traj = decision.get("predicted_trajectory")
    if isinstance(traj, (list, tuple)) and len(traj) >= 3:
        wp2 = traj[2]
        if isinstance(wp2, (list, tuple)) and len(wp2) >= 2:
            try:
                x = float(wp2[0])
                y = float(wp2[1])
                if math.isfinite(x) and math.isfinite(y):
                    return x, y, True
            except (TypeError, ValueError):
                pass
    return 0.0, 0.0, False


def compute_kinematic_aeb_distance(
    current_speed_kmh: float,
    lead_speed_kmh: float,
    a_max_mps2: float = 6.0,
    uncertainty_margin_m: float = 0.0,
) -> float:
    cur_v = float(current_speed_kmh) if math.isfinite(current_speed_kmh) else 0.0
    lead_v = float(lead_speed_kmh) if math.isfinite(lead_speed_kmh) else 0.0
    amax = float(a_max_mps2) if math.isfinite(a_max_mps2) else 6.0
    unc = float(uncertainty_margin_m) if math.isfinite(uncertainty_margin_m) else 0.0

    v_ego_mps = max(0.0, cur_v / 3.6)
    v_lead_mps = max(0.0, lead_v / 3.6)
    closing_mps = max(0.0, v_ego_mps - v_lead_mps)
    eff_a_max = max(2.5, amax)
    return 5.5 + (v_ego_mps * v_ego_mps) / (2.0 * eff_a_max) + 0.45 * closing_mps + unc


def compute_mpc_feedforward(
    road_curv: float,
    lane_offset_m: float,
    wp2_x: float = 0.0,
    wp2_y: float = 0.0,
    has_wp2: bool = False,
) -> float:
    wheelbase_m = 2.85
    if has_wp2:
        lx = max(3.5, float(wp2_x))
        ly = float(wp2_y) - float(lane_offset_m)
        traj_curv = (2.0 * ly) / (lx * lx + ly * ly)
    else:
        traj_curv = 0.0
    eff_curv = 0.70 * (-float(road_curv) * 0.045) + 0.30 * traj_curv
    delta_rad = math.atan(wheelbase_m * eff_curv)
    return _clamp(delta_rad / 0.70, -0.35, 0.35)


def ols_5step_slope_mps(window: Sequence[float], dt: float = 0.05) -> float:
    n_w = len(window)
    if n_w < 2:
        return 0.0
    n_f = float(n_w)
    t_mean = (n_f - 1.0) * 0.5 * float(dt)
    d_mean = sum(float(d) for d in window) / n_f
    num = 0.0
    den = 0.0
    for i, d in enumerate(window):
        dt_i = float(i) * float(dt) - t_mean
        num += dt_i * (float(d) - d_mean)
        den += dt_i * dt_i
    return num / max(den, 1e-6)


def timestamp_ols_slope_mps(
    timestamps: Sequence[float], distances: Sequence[float]
) -> float:
    n = min(len(timestamps), len(distances))
    if n < 2:
        return 0.0
    n_f = float(n)
    t_mean = sum(float(timestamps[i]) for i in range(n)) / n_f
    d_mean = sum(float(distances[i]) for i in range(n)) / n_f
    num = 0.0
    den = 0.0
    for i in range(n):
        dt_i = float(timestamps[i]) - t_mean
        num += dt_i * (float(distances[i]) - d_mean)
        den += dt_i * dt_i
    return num / max(den, 1e-6)


def count_sustained_aeb_events(
    throttle_brake_trace: Sequence[float],
    threshold: float = -0.50,
    min_contiguous_ticks: int = 3,
    max_gap_bridge: int = 2,
) -> Tuple[int, int]:
    active = [1 if float(u) <= float(threshold) else 0 for u in throttle_brake_trace]
    total_ticks = sum(active)
    bridged = list(active)
    n = len(bridged)
    i = 0
    while i < n:
        if bridged[i] == 1:
            j = i + 1
            while j < n and bridged[j] == 0 and (j - i - 1) <= max_gap_bridge:
                j += 1
            if j < n and bridged[j] == 1 and (j - i - 1) <= max_gap_bridge:
                for k in range(i + 1, j):
                    bridged[k] = 1
            i = j
        else:
            i += 1

    events = 0
    run_len = 0
    for val in bridged:
        if val == 1:
            run_len += 1
        else:
            if run_len >= min_contiguous_ticks:
                events += 1
            run_len = 0
    if run_len >= min_contiguous_ticks:
        events += 1
    return events, total_ticks


def compute_polyline_tangents(
    xyz_list: Sequence[Tuple[float, float, float]],
) -> List[Tuple[float, float, float, float]]:
    n = len(xyz_list)
    if n == 0:
        return []
    pts: List[Tuple[float, float, float, float]] = []
    for i in range(n):
        x, y, z = float(xyz_list[i][0]), float(xyz_list[i][1]), float(xyz_list[i][2])
        if i + 1 < n:
            nx, ny = float(xyz_list[i + 1][0]), float(xyz_list[i + 1][1])
            yaw = math.atan2(ny - y, nx - x)
        elif i > 0:
            px, py = float(xyz_list[i - 1][0]), float(xyz_list[i - 1][1])
            yaw = math.atan2(y - py, x - px)
        else:
            yaw = 0.0
        pts.append((x, y, z, yaw))
    return pts


def quintic_s_curve_blend(
    orig_xyz: Sequence[Tuple[float, float, float]],
    adj_xyz: Sequence[Tuple[float, float, float]],
    n_blend: int = 14,
) -> List[Tuple[float, float, float]]:
    eff_blend = min(int(n_blend), len(orig_xyz), len(adj_xyz))
    denom = float(max(1, max(0, eff_blend - 1)))
    out: List[Tuple[float, float, float]] = []
    for k, (ax, ay, az) in enumerate(adj_xyz):
        ax_f, ay_f, az_f = float(ax), float(ay), float(az)
        if k < eff_blend:
            ox, oy, oz = (
                float(orig_xyz[k][0]),
                float(orig_xyz[k][1]),
                float(orig_xyz[k][2]),
            )
            tau = float(k) / denom
            s = (tau * tau * tau) * (10.0 + tau * (-15.0 + 6.0 * tau))
            out.append(
                (
                    (1.0 - s) * ox + s * ax_f,
                    (1.0 - s) * oy + s * ay_f,
                    (1.0 - s) * oz + s * az_f,
                )
            )
        else:
            out.append((ax_f, ay_f, az_f))
    return out


def sample_continuous_route_pose(
    pts: Sequence[Tuple[float, float, float, float]],
    x: float,
    y: float,
    start_idx: int = 0,
    lookahead_m: float = 8.0,
) -> Tuple[int, float, float, float, float, float, float]:
    if not pts:
        return (0, float(x), float(y), 0.0, float(x) + float(lookahead_m), float(y), 0.0)
    n = len(pts)
    best_i = min(max(0, int(start_idx)), n - 1)
    best_d = 1e9
    end_i = min(n, best_i + 12)
    for idx in range(best_i, end_i):
        d = math.hypot(float(x) - pts[idx][0], float(y) - pts[idx][1])
        if d < best_d:
            best_d = d
            best_i = idx

    if best_i + 1 < n:
        x0, y0, _, yaw0 = pts[best_i]
        x1, y1, _, _ = pts[best_i + 1]
        vx = x1 - x0
        vy = y1 - y0
        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq > 1e-4:
            t_seg = _clamp(((float(x) - x0) * vx + (float(y) - y0) * vy) / seg_len_sq, 0.0, 1.0)
        else:
            t_seg = 0.0
        proj_x = x0 + t_seg * vx
        proj_y = y0 + t_seg * vy
        proj_yaw = yaw0
    else:
        proj_x, proj_y, _, proj_yaw = pts[best_i]

    rem = float(lookahead_m)
    cx = proj_x
    cy = proj_y
    curr_idx = best_i
    look_x = pts[n - 1][0]
    look_y = pts[n - 1][1]
    while curr_idx + 1 < n:
        nx, ny, _, _ = pts[curr_idx + 1]
        ds = math.hypot(nx - cx, ny - cy)
        if ds >= rem and ds > 1e-4:
            ratio = rem / ds
            look_x = cx + ratio * (nx - cx)
            look_y = cy + ratio * (ny - cy)
            break
        rem -= ds
        cx = nx
        cy = ny
        curr_idx += 1
        look_x = cx
        look_y = cy

    ref_yaw = math.atan2(look_y - proj_y, look_x - proj_x)
    return (best_i, proj_x, proj_y, proj_yaw, look_x, look_y, ref_yaw)


class RustScalarPID:
    """Pure-Python equivalent of `RustScalarPID`."""

    def __init__(
        self, kp: float, ki: float, kd: float, integral_limit: float = 5.0
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = float(integral_limit)
        self.integral: float = 0.0
        self.prev_error: Optional[float] = None

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def step(self, error: float, dt: float = 0.05) -> float:
        err = float(error)
        step_dt = float(dt)
        if not math.isfinite(err) or not math.isfinite(step_dt):
            return 0.0

        if self.prev_error is not None:
            if abs(err - self.prev_error) > 1.2:
                self.integral = 0.0
                derivative = 0.0
            else:
                self.integral = _clamp(
                    self.integral + err * step_dt,
                    -self.integral_limit,
                    self.integral_limit,
                )
                derivative = (err - self.prev_error) / max(step_dt, 1e-4)
        else:
            self.integral = _clamp(
                self.integral + err * step_dt,
                -self.integral_limit,
                self.integral_limit,
            )
            derivative = 0.0

        self.prev_error = err
        return self.kp * err + self.ki * self.integral + self.kd * derivative


class RustLeadSpeedEstimator:
    """Pure-Python equivalent of `RustLeadSpeedEstimator`."""

    def __init__(self, initial_v_lead_kmh: float = 35.0) -> None:
        self.window: List[Tuple[float, float]] = []
        self.obs_time_s: float = 0.0
        self.prev_est_v_lead: float = float(initial_v_lead_kmh)

    def reset(self, initial_v_lead_kmh: float = 35.0) -> None:
        self.window.clear()
        self.obs_time_s = 0.0
        self.prev_est_v_lead = float(initial_v_lead_kmh)

    def update(
        self,
        pred_d: float,
        pred_rr_mps: float,
        ego_v_kmh: float,
        elapsed_steps: int = 1,
        dt: float = 0.05,
    ) -> float:
        steps = max(1, int(elapsed_steps))
        self.obs_time_s += float(steps) * float(dt)

        d_val = float(pred_d) if math.isfinite(pred_d) else 60.0
        rr_val = float(pred_rr_mps) if math.isfinite(pred_rr_mps) else 0.0
        v_ego = float(ego_v_kmh) if math.isfinite(ego_v_kmh) else 0.0

        if self.window and abs(d_val - self.window[-1][1]) > 6.0:
            self.window.clear()
        self.window.append((self.obs_time_s, d_val))
        if len(self.window) > 5:
            self.window.pop(0)

        if len(self.window) < 2:
            dd_dt_reg = rr_val
        else:
            n_f = float(len(self.window))
            t_mean = sum(pt[0] for pt in self.window) / n_f
            d_mean = sum(pt[1] for pt in self.window) / n_f
            num = 0.0
            den = 0.0
            for t_i, d_i in self.window:
                dt_i = t_i - t_mean
                num += dt_i * (d_i - d_mean)
                den += dt_i * dt_i
            dd_dt_reg = num / max(den, 1e-6)

        dd_dt_fused = 0.70 * dd_dt_reg + 0.30 * rr_val
        raw_v_lead = v_ego + 3.6 * dd_dt_fused
        est_v_lead = _clamp(0.65 * self.prev_est_v_lead + 0.35 * raw_v_lead, 0.0, 60.0)
        self.prev_est_v_lead = est_v_lead
        return est_v_lead


class RustSafetyShieldPID:
    """Pure-Python equivalent of `RustSafetyShieldPID`."""

    def __init__(
        self,
        use_mock: bool = False,
        aeb_threshold: float = 0.70,
        confidence_threshold: float = 0.80,
        lane_width: float = 3.5,
        heading_weight: float = 1.25,
        max_steer_rate_per_sec: float = 3.2,
        steer_smoothing_alpha: float = 0.72,
        lat_kp: float = 0.45,
        lat_ki: float = 0.01,
        lat_kd: float = 0.18,
        lon_kp: float = 0.08,
        lon_ki: float = 0.005,
        lon_kd: float = 0.02,
    ) -> None:
        self.use_mock = bool(use_mock)
        self.aeb_threshold = float(aeb_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.lane_width = float(lane_width)
        self.heading_weight = float(heading_weight)
        self.max_steer_rate_per_sec = float(max_steer_rate_per_sec)
        self.steer_smoothing_alpha = float(steer_smoothing_alpha)
        self.current_steering = 0.0
        self.current_throttle_brake = 0.0
        self.current_target_offset_m = 0.0
        self.stop_hold_active = False
        self.lateral_pid = RustScalarPID(lat_kp, lat_ki, lat_kd, 5.0)
        self.longitudinal_pid = RustScalarPID(lon_kp, lon_ki, lon_kd, 5.0)

    def reset(self) -> None:
        self.lateral_pid.reset()
        self.longitudinal_pid.reset()
        self.current_steering = 0.0
        self.current_throttle_brake = 0.0
        self.current_target_offset_m = 0.0
        self.stop_hold_active = False

    def _apply_safety_shield_core(
        self,
        raw_maneuver: str,
        raw_speed_level: int,
        maneuver_conf: float,
        aeb_prob: float,
        lead_distance_m: float,
        lead_speed_kmh: float,
        has_lead_speed: bool,
        ped_distance_m: float,
        current_speed_kmh: float,
        traffic_light: str,
        hazard_source: str,
        a_max_mps2: float,
        unc_margin_m: float,
        left_lane_available: bool,
        right_lane_available: bool,
        blind_spot_left: bool,
        blind_spot_right: bool,
        road_curvature: float,
        shield_enabled: bool,
    ) -> Dict[str, Any]:
        d_aeb_m = compute_kinematic_aeb_distance(
            current_speed_kmh, lead_speed_kmh, a_max_mps2, unc_margin_m
        )
        gated_maneuver = raw_maneuver
        gated_speed_level = raw_speed_level
        shield_triggered = False
        shield_reason = "none"

        if not shield_enabled:
            if aeb_prob >= self.aeb_threshold:
                gated_maneuver = "keep_lane"
                gated_speed_level = 0
                shield_triggered = True
                shield_reason = "model_aeb_only"
            target_speed_kmh = _speed_level_to_kmh(gated_speed_level)
            if lead_distance_m < 38.0 and gated_speed_level > 0:
                if self.stop_hold_active:
                    if lead_speed_kmh >= 3.5 or lead_distance_m >= 9.5:
                        self.stop_hold_active = False
                elif lead_speed_kmh < 2.0 and lead_distance_m <= 8.5:
                    self.stop_hold_active = True

                if self.stop_hold_active:
                    target_speed_kmh = 0.0
                else:
                    desired_gap_m = 8.0 + 1.2 * (current_speed_kmh / 3.6)
                    gap_error_m = lead_distance_m - desired_gap_m
                    acc_upper_kmh = min(35.0, lead_speed_kmh + 6.0)
                    target_speed_kmh = _clamp(
                        lead_speed_kmh + 0.85 * gap_error_m, 0.0, acc_upper_kmh
                    )
            return {
                "raw_maneuver": raw_maneuver,
                "raw_speed_level": raw_speed_level,
                "maneuver": gated_maneuver,
                "speed_level": gated_speed_level,
                "target_speed_kmh": target_speed_kmh,
                "mpc_cornering_cap_kmh": 65.0,
                "maneuver_confidence": maneuver_conf,
                "emergency_brake": aeb_prob,
                "shield_triggered": shield_triggered,
                "shield_reason": shield_reason,
                "blind_spot_left": blind_spot_left,
                "blind_spot_right": blind_spot_right,
            }

        if (
            aeb_prob >= self.aeb_threshold
            or traffic_light == "red"
            or ped_distance_m <= 16.0
            or (hazard_source == "pedestrian" and lead_distance_m <= 16.0)
            or (
                lead_distance_m <= d_aeb_m
                and current_speed_kmh > max(6.0, lead_speed_kmh + 6.0)
            )
        ):
            gated_maneuver = "keep_lane"
            gated_speed_level = 0
            shield_triggered = True
            shield_reason = "rule1_emergency_brake"
        else:
            if maneuver_conf < self.confidence_threshold:
                if raw_maneuver in ("change_left", "change_right"):
                    gated_maneuver = "keep_lane"
                gated_speed_level = max(1, raw_speed_level - 1) if raw_speed_level > 0 else 0
                shield_triggered = True
                shield_reason = "rule2_low_confidence_fallback"

            if gated_maneuver == "change_left" and not left_lane_available:
                gated_maneuver = "keep_lane"
                shield_triggered = True
                shield_reason = "boundary_left_unavailable"
            elif gated_maneuver == "change_right" and not right_lane_available:
                gated_maneuver = "keep_lane"
                shield_triggered = True
                shield_reason = "boundary_right_unavailable"
            elif gated_maneuver == "change_left" and blind_spot_left:
                gated_maneuver = "keep_lane"
                shield_triggered = True
                shield_reason = "blindspot_left_occupied"
            elif gated_maneuver == "change_right" and blind_spot_right:
                gated_maneuver = "keep_lane"
                shield_triggered = True
                shield_reason = "blindspot_right_occupied"

        target_speed_kmh = _speed_level_to_kmh(gated_speed_level)
        road_curv_abs = abs(road_curvature)
        effective_kappa = road_curv_abs * 0.065
        mpc_cornering_cap_kmh = min(65.0, 3.6 * math.sqrt(2.85 / max(effective_kappa, 1e-3)))
        if road_curv_abs > 0.22 and gated_speed_level > 0:
            target_speed_kmh = min(target_speed_kmh, max(20.0, mpc_cornering_cap_kmh))

        if gated_maneuver == "follow_lead" and gated_speed_level >= 2:
            target_speed_kmh = min(target_speed_kmh, 30.0)
        if lead_distance_m < 52.0 and gated_speed_level > 0:
            smooth_cap = 30.0 + max(0.0, lead_distance_m - 30.0) * 1.4
            target_speed_kmh = min(target_speed_kmh, smooth_cap)
        if lead_distance_m < 38.0 and gated_speed_level > 0:
            eff_lead_speed_kmh = lead_speed_kmh if has_lead_speed else target_speed_kmh
            if self.stop_hold_active:
                if eff_lead_speed_kmh >= 3.5 or lead_distance_m >= 9.5:
                    self.stop_hold_active = False
            elif eff_lead_speed_kmh < 2.0 and lead_distance_m <= 8.5:
                self.stop_hold_active = True

            if self.stop_hold_active:
                target_speed_kmh = 0.0
            else:
                desired_gap_m = 8.0 + 1.2 * (current_speed_kmh / 3.6)
                gap_error_m = lead_distance_m - desired_gap_m
                acc_upper_kmh = min(35.0, eff_lead_speed_kmh + 6.0)
                target_speed_kmh = _clamp(
                    eff_lead_speed_kmh + 0.85 * gap_error_m, 0.0, acc_upper_kmh
                )

        return {
            "raw_maneuver": raw_maneuver,
            "raw_speed_level": raw_speed_level,
            "maneuver": gated_maneuver,
            "speed_level": gated_speed_level,
            "target_speed_kmh": target_speed_kmh,
            "mpc_cornering_cap_kmh": _round_n(mpc_cornering_cap_kmh, 1),
            "maneuver_confidence": maneuver_conf,
            "emergency_brake": aeb_prob,
            "shield_triggered": shield_triggered,
            "shield_reason": shield_reason,
            "blind_spot_left": blind_spot_left,
            "blind_spot_right": blind_spot_right,
        }

    def apply_safety_shield(
        self,
        decision: Mapping[str, Any],
        telemetry: Mapping[str, Any],
        shield_enabled: bool = True,
    ) -> Dict[str, Any]:
        wet_road = _extract_bool(telemetry, "wet_road", False)
        return self._apply_safety_shield_core(
            raw_maneuver=_extract_str(decision, "maneuver", "keep_lane"),
            raw_speed_level=_extract_i64(decision, "speed_level", 2),
            maneuver_conf=_extract_f64(decision, "maneuver_confidence", 1.0),
            aeb_prob=_extract_f64(decision, "emergency_brake", 0.0),
            lead_distance_m=_extract_f64_2keys(
                telemetry, "lead_distance_m", "lead_distance", 60.0
            ),
            lead_speed_kmh=_extract_f64(telemetry, "lead_speed_kmh", 60.0),
            has_lead_speed=telemetry.get("lead_speed_kmh") is not None,
            ped_distance_m=_extract_f64_2keys(
                telemetry, "pedestrian_distance_m", "pedestrian_distance", 100.0
            ),
            current_speed_kmh=_extract_f64(telemetry, "speed_kmh", 0.0),
            traffic_light=_extract_str(telemetry, "traffic_light", "green"),
            hazard_source=_extract_str(telemetry, "hazard_source", "none"),
            a_max_mps2=_extract_f64(telemetry, "a_max_mps2", 4.5 if wet_road else 6.0),
            unc_margin_m=_extract_f64(
                telemetry, "uncertainty_margin_m", 1.8 if wet_road else 0.0
            ),
            left_lane_available=_extract_bool(telemetry, "left_lane_available", True),
            right_lane_available=_extract_bool(telemetry, "right_lane_available", True),
            blind_spot_left=_extract_bool(telemetry, "blind_spot_left", False),
            blind_spot_right=_extract_bool(telemetry, "blind_spot_right", False),
            road_curvature=_extract_f64(telemetry, "road_curvature", 0.0),
            shield_enabled=bool(shield_enabled),
        )

    def compute_control(
        self,
        decision: Mapping[str, Any],
        telemetry: Mapping[str, Any],
        dt: float = 0.05,
        shield_enabled: bool = True,
    ) -> Tuple[Tuple[float, float], Dict[str, Any]]:
        step_dt = float(dt) if math.isfinite(dt) and dt > 0.0 else 0.05
        gated = self.apply_safety_shield(decision, telemetry, shield_enabled=shield_enabled)

        wp2_x, wp2_y, has_wp2 = _extract_wp2(decision)
        current_speed_kmh = _extract_f64(telemetry, "speed_kmh", 0.0)
        lane_offset_m = _extract_f64(telemetry, "lane_offset_m", 0.0)
        heading_error = _extract_f64(telemetry, "heading_error", 0.0)
        road_curvature = _extract_f64(telemetry, "road_curvature", 0.0)
        left_lane_available = _extract_bool(telemetry, "left_lane_available", True)
        right_lane_available = _extract_bool(telemetry, "right_lane_available", True)

        if self.use_mock and gated["maneuver"] == "change_left":
            target_offset_m = self.lane_width * 0.85
        elif self.use_mock and gated["maneuver"] == "change_right":
            target_offset_m = -self.lane_width * 0.85
        elif (
            not self.use_mock
            and abs(road_curvature) > 0.25
            and not left_lane_available
            and right_lane_available
        ):
            target_offset_m = -0.18
        else:
            target_offset_m = 0.0

        lateral_error = (target_offset_m - lane_offset_m) - (
            self.heading_weight * heading_error
        )
        pid_steering = _clamp(self.lateral_pid.step(lateral_error, step_dt), -1.0, 1.0)
        mpc_ff = (
            0.0
            if self.use_mock
            else compute_mpc_feedforward(
                road_curvature, lane_offset_m, wp2_x, wp2_y, has_wp2
            )
        )
        raw_target_steering = _clamp(pid_steering + 0.12 * mpc_ff, -1.0, 1.0)

        steer_from = self.current_steering
        road_curv_abs = abs(road_curvature)
        if self.use_mock:
            max_delta, alpha = 1.0, 1.0
        elif road_curv_abs >= 0.35:
            speed_damping = 1.0 + 0.012 * max(0.0, current_speed_kmh)
            max_delta, alpha = min(0.22, 3.5 * (1.1 * step_dt) / speed_damping), 0.82
        else:
            speed_damping = 1.0 + 0.012 * max(0.0, current_speed_kmh)
            max_delta, alpha = (1.1 * step_dt) / speed_damping, 0.55

        steer_step = _clamp((raw_target_steering - steer_from) * alpha, -max_delta, max_delta)
        steering = _clamp(steer_from + steer_step, -1.0, 1.0)
        self.current_steering = steering

        accel_from = self.current_throttle_brake
        if gated["shield_reason"] in ("rule1_emergency_brake", "model_aeb_only"):
            throttle_brake = -1.0
        else:
            speed_error = gated["target_speed_kmh"] - current_speed_kmh
            raw_tb = _clamp(self.longitudinal_pid.step(speed_error, step_dt), -1.0, 1.0)
            if self.use_mock:
                throttle_brake = raw_tb
            else:
                tb_step = _clamp(raw_tb - accel_from, -0.25, 0.20)
                throttle_brake = _clamp(accel_from + tb_step, -1.0, 1.0)
        self.current_throttle_brake = throttle_brake

        steer_ramp_substeps = [
            _round_n(steer_from + (float(idx + 1) / 5.0) * (steering - steer_from), 4)
            for idx in range(5)
        ]

        steer_c = _clamp(steering, -1.0, 1.0)
        tb_c = _clamp(throttle_brake, -1.0, 1.0)
        gated.update(
            {
                "steer_from": _round_n(steer_from, 4),
                "steer_to": _round_n(steering, 4),
                "steer_target": _round_n(raw_target_steering, 4),
                "steer_delta": _round_n(steering - steer_from, 4),
                "mpc_feedforward": _round_n(mpc_ff, 4),
                "steer_ramp_substeps": steer_ramp_substeps,
                "steering": steering,
                "throttle_brake": throttle_brake,
                "carla_control": {
                    "steer": _round_n(steer_c, 4),
                    "throttle": _round_n(tb_c if tb_c >= 0.0 else 0.0, 4),
                    "brake": _round_n(-tb_c if tb_c < 0.0 else 0.0, 4),
                    "hand_brake": False,
                    "reverse": False,
                },
            }
        )
        return (steering, throttle_brake), gated


class RustExpertLabeler:
    """Pure-Python equivalent of `RustExpertLabeler`."""

    def __init__(
        self,
        lane_change_steer_thresh: float = 0.22,
        aeb_decel_thresh: float = -0.75,
        aeb_distance_thresh_m: float = 13.5,
        follow_distance_thresh_m: float = 30.0,
        curve_curvature_thresh: float = 0.40,
    ) -> None:
        self.lane_change_steer_thresh = float(lane_change_steer_thresh)
        self.aeb_decel_thresh = float(aeb_decel_thresh)
        self.aeb_distance_thresh_m = float(aeb_distance_thresh_m)
        self.follow_distance_thresh_m = float(follow_distance_thresh_m)
        self.curve_curvature_thresh = float(curve_curvature_thresh)

    def label_step(
        self,
        telemetry: Mapping[str, Any],
        expert_action: Tuple[float, float],
        info: Mapping[str, Any],
    ) -> Tuple[str, int, int, int]:
        steering = float(expert_action[0]) if math.isfinite(expert_action[0]) else 0.0
        throttle_brake = (
            float(expert_action[1]) if math.isfinite(expert_action[1]) else 0.0
        )
        speed_kmh = _extract_f64(telemetry, "speed_kmh", 0.0)
        left_avail = _extract_bool(telemetry, "left_lane_available", False)
        right_avail = _extract_bool(telemetry, "right_lane_available", False)
        nav_command = _extract_str(telemetry, "nav_command", "go_straight")

        lead_distance_m = _extract_f64(
            info, "lead_distance_m", _extract_f64(telemetry, "lead_distance_m", 60.0)
        )
        road_curvature = abs(
            _extract_f64(
                info, "road_curvature", _extract_f64(telemetry, "road_curvature", 0.0)
            )
        )
        scenario_event = _extract_str(info, "scenario_event", "")
        traffic_light = _extract_str(
            telemetry, "traffic_light", _extract_str(info, "traffic_light", "green")
        )
        hazard_source = _extract_str(
            telemetry, "hazard_source", _extract_str(info, "hazard_source", "none")
        )

        is_real_sim = scenario_event in ("real_metadrive", "real_carla")
        is_aeb = (
            scenario_event == "emergency_brake"
            or traffic_light == "red"
            or (hazard_source == "pedestrian" and lead_distance_m <= 16.0)
            or (
                not is_real_sim
                and throttle_brake <= self.aeb_decel_thresh
                and lead_distance_m <= self.aeb_distance_thresh_m
            )
            or (lead_distance_m < 9.5 and speed_kmh > 15.0)
        )
        emergency_brake = 1 if is_aeb else 0

        expert_maneuver_info = _extract_str(info, "expert_maneuver", "")
        expert_maneuver = (
            expert_maneuver_info
            if expert_maneuver_info
            else _extract_str(telemetry, "expert_maneuver", "")
        )

        if scenario_event in ("keep_lane", "change_left", "change_right", "follow_lead"):
            maneuver = scenario_event
        elif expert_maneuver in ("change_left", "change_right"):
            maneuver = expert_maneuver
        elif is_aeb:
            maneuver = "follow_lead"
        elif (
            not is_real_sim
            and steering >= self.lane_change_steer_thresh
            and left_avail
            and road_curvature < self.curve_curvature_thresh
        ):
            maneuver = "change_left"
        elif (
            not is_real_sim
            and steering <= -self.lane_change_steer_thresh
            and right_avail
            and road_curvature < self.curve_curvature_thresh
        ):
            maneuver = "change_right"
        elif lead_distance_m <= self.follow_distance_thresh_m:
            maneuver = "follow_lead"
        else:
            maneuver = "keep_lane"

        maneuver_idx = {
            "keep_lane": 0,
            "change_left": 1,
            "change_right": 2,
            "follow_lead": 3,
        }.get(maneuver, 0)

        if emergency_brake == 1:
            speed_level = 0
        elif scenario_event == "sharp_curve" or road_curvature >= self.curve_curvature_thresh:
            speed_level = 1
        elif (
            maneuver in ("change_left", "change_right", "follow_lead")
            or nav_command in ("merge", "turn_left", "turn_right")
            or road_curvature >= 0.22
        ):
            speed_level = 2 if lead_distance_m > 16.0 else 1
        elif lead_distance_m >= 45.0 and road_curvature < 0.22:
            speed_level = 3
        else:
            speed_level = 2

        return maneuver, maneuver_idx, speed_level, emergency_brake


def benchmark_rust_control_loop_us(iterations: int = 10000) -> float:
    n = max(1, int(iterations))
    ctrl = RustSafetyShieldPID()
    ols = RustLeadSpeedEstimator(24.0)
    route_xyz = [(float(i) * 2.0, 0.0, 0.0) for i in range(50)]
    pts = compute_polyline_tangents(route_xyz)
    t0 = time.perf_counter()
    for step in range(n):
        d_obs = 22.0 - float(step % 5) * 0.2
        v_lead = ols.update(d_obs, -4.0, 28.0, 1, 0.05)
        _, _, _, _, lx, ly, _ = sample_continuous_route_pose(
            pts, float(step % 40), 0.05, 0, 8.0
        )
        _ = ctrl.compute_control(
            {
                "maneuver": "follow_lead",
                "maneuver_confidence": 0.95,
                "speed_level": 2,
                "emergency_brake": 0.02,
                "predicted_trajectory": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [lx, ly, v_lead]],
            },
            {
                "speed_kmh": 28.0,
                "lead_distance_m": d_obs,
                "lead_speed_kmh": v_lead,
                "pedestrian_distance_m": 100.0,
                "lane_offset_m": 0.05,
                "heading_error": 0.01,
                "traffic_light": "green",
                "hazard_source": "vehicle",
                "wet_road": False,
                "a_max_mps2": 6.0,
                "uncertainty_margin_m": 0.0,
                "left_lane_available": True,
                "right_lane_available": True,
                "blind_spot_left": False,
                "blind_spot_right": False,
                "road_curvature": 0.12,
            },
            dt=0.05,
            shield_enabled=True,
        )
    return ((time.perf_counter() - t0) * 1e6) / float(n)


__all__ = [
    "RustExpertLabeler",
    "RustLeadSpeedEstimator",
    "RustSafetyShieldPID",
    "RustScalarPID",
    "benchmark_rust_control_loop_us",
    "compute_kinematic_aeb_distance",
    "compute_mpc_feedforward",
    "compute_polyline_tangents",
    "count_sustained_aeb_events",
    "ols_5step_slope_mps",
    "quintic_s_curve_blend",
    "sample_continuous_route_pose",
    "timestamp_ols_slope_mps",
]
