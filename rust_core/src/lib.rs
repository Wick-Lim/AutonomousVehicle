#![allow(non_local_definitions)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::time::Instant;

#[inline]
fn round_n(val: f64, decimals: i32) -> f64 {
    let factor = 10_f64.powi(decimals);
    (val * factor).round() / factor
}

#[inline]
fn speed_level_to_kmh(level: i64) -> f64 {
    match level {
        0 => 0.0,
        1 => 20.0,
        2 => 40.0,
        3 => 65.0,
        _ => 20.0,
    }
}

#[derive(Clone, Debug)]
pub struct ScalarPidCore {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub integral_limit: f64,
    pub integral: f64,
    pub prev_error: Option<f64>,
}

impl ScalarPidCore {
    pub fn new(kp: f64, ki: f64, kd: f64, integral_limit: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral_limit,
            integral: 0.0,
            prev_error: None,
        }
    }

    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = None;
    }

    #[inline]
    pub fn step(&mut self, error: f64, dt: f64) -> f64 {
        let derivative = match self.prev_error {
            Some(prev) if (error - prev).abs() > 1.2 => {
                self.integral = 0.0;
                0.0
            }
            Some(prev) => {
                self.integral = (self.integral + error * dt)
                    .clamp(-self.integral_limit, self.integral_limit);
                (error - prev) / dt.max(1e-4)
            }
            None => {
                self.integral = (self.integral + error * dt)
                    .clamp(-self.integral_limit, self.integral_limit);
                0.0
            }
        };
        self.prev_error = Some(error);
        self.kp * error + self.ki * self.integral + self.kd * derivative
    }
}

#[inline]
pub fn compute_kinematic_aeb_distance_core(
    current_speed_kmh: f64,
    lead_speed_kmh: f64,
    a_max_mps2: f64,
    uncertainty_margin_m: f64,
) -> f64 {
    let v_ego_mps = (current_speed_kmh / 3.6).max(0.0);
    let v_lead_mps = (lead_speed_kmh / 3.6).max(0.0);
    let closing_mps = (v_ego_mps - v_lead_mps).max(0.0);
    let eff_a_max = a_max_mps2.max(2.5);
    5.5 + (v_ego_mps * v_ego_mps) / (2.0 * eff_a_max) + 0.45 * closing_mps + uncertainty_margin_m
}

#[inline]
pub fn compute_mpc_feedforward_core(
    road_curv: f64,
    lane_offset_m: f64,
    wp2_x: f64,
    wp2_y: f64,
    has_wp2: bool,
) -> f64 {
    let wheelbase_m = 2.85_f64;
    let traj_curv = if has_wp2 {
        let lx = wp2_x.max(3.5);
        let ly = wp2_y - lane_offset_m;
        (2.0 * ly) / (lx * lx + ly * ly)
    } else {
        0.0
    };
    let eff_curv = 0.70 * (-road_curv * 0.045) + 0.30 * traj_curv;
    let delta_rad = (wheelbase_m * eff_curv).atan();
    (delta_rad / 0.70).clamp(-0.35, 0.35)
}

#[derive(Clone, Debug)]
pub struct ShieldGatedOutput {
    pub raw_maneuver: String,
    pub raw_speed_level: i64,
    pub maneuver: String,
    pub speed_level: i64,
    pub target_speed_kmh: f64,
    pub mpc_cornering_cap_kmh: f64,
    pub maneuver_confidence: f64,
    pub emergency_brake: f64,
    pub shield_triggered: bool,
    pub shield_reason: String,
    pub blind_spot_left: bool,
    pub blind_spot_right: bool,
}

#[derive(Clone, Debug)]
pub struct ControlStepOutput {
    pub steering: f64,
    pub throttle_brake: f64,
    pub steer_from: f64,
    pub steer_to: f64,
    pub steer_target: f64,
    pub steer_delta: f64,
    pub mpc_feedforward: f64,
    pub steer_ramp_substeps: [f64; 5],
    pub gated: ShieldGatedOutput,
}

#[derive(Clone, Debug)]
pub struct SafetyShieldPidCore {
    pub use_mock: bool,
    pub aeb_threshold: f64,
    pub confidence_threshold: f64,
    pub lane_width: f64,
    pub heading_weight: f64,
    pub max_steer_rate_per_sec: f64,
    pub steer_smoothing_alpha: f64,
    pub current_steering: f64,
    pub current_throttle_brake: f64,
    pub current_target_offset_m: f64,
    pub stop_hold_active: bool,
    pub lateral_pid: ScalarPidCore,
    pub longitudinal_pid: ScalarPidCore,
}

impl SafetyShieldPidCore {
    pub fn new(
        use_mock: bool,
        aeb_threshold: f64,
        confidence_threshold: f64,
        lane_width: f64,
        heading_weight: f64,
        max_steer_rate_per_sec: f64,
        steer_smoothing_alpha: f64,
        lat_kp: f64,
        lat_ki: f64,
        lat_kd: f64,
        lon_kp: f64,
        lon_ki: f64,
        lon_kd: f64,
    ) -> Self {
        Self {
            use_mock,
            aeb_threshold,
            confidence_threshold,
            lane_width,
            heading_weight,
            max_steer_rate_per_sec,
            steer_smoothing_alpha,
            current_steering: 0.0,
            current_throttle_brake: 0.0,
            current_target_offset_m: 0.0,
            stop_hold_active: false,
            lateral_pid: ScalarPidCore::new(lat_kp, lat_ki, lat_kd, 5.0),
            longitudinal_pid: ScalarPidCore::new(lon_kp, lon_ki, lon_kd, 5.0),
        }
    }

    pub fn reset(&mut self) {
        self.lateral_pid.reset();
        self.longitudinal_pid.reset();
        self.current_steering = 0.0;
        self.current_throttle_brake = 0.0;
        self.current_target_offset_m = 0.0;
        self.stop_hold_active = false;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_safety_shield_core(
        &mut self,
        raw_maneuver: &str,
        raw_speed_level: i64,
        maneuver_conf: f64,
        aeb_prob: f64,
        lead_distance_m: f64,
        lead_speed_kmh: f64,
        has_lead_speed: bool,
        ped_distance_m: f64,
        current_speed_kmh: f64,
        traffic_light: &str,
        hazard_source: &str,
        _wet_road: bool,
        a_max_mps2: f64,
        unc_margin_m: f64,
        left_lane_available: bool,
        right_lane_available: bool,
        blind_spot_left: bool,
        blind_spot_right: bool,
        road_curvature: f64,
        shield_enabled: bool,
    ) -> ShieldGatedOutput {
        let d_aeb_m = compute_kinematic_aeb_distance_core(
            current_speed_kmh,
            lead_speed_kmh,
            a_max_mps2,
            unc_margin_m,
        );

        let mut gated_maneuver = raw_maneuver.to_string();
        let mut gated_speed_level = raw_speed_level;
        let mut shield_triggered = false;
        let mut shield_reason = "none".to_string();

        if !shield_enabled {
            if aeb_prob >= self.aeb_threshold {
                gated_maneuver = "keep_lane".to_string();
                gated_speed_level = 0;
                shield_triggered = true;
                shield_reason = "model_aeb_only".to_string();
            }
            let mut target_speed_kmh = speed_level_to_kmh(gated_speed_level);
            if lead_distance_m < 38.0 && gated_speed_level > 0 {
                if self.stop_hold_active {
                    if lead_speed_kmh >= 3.5 || lead_distance_m >= 9.5 {
                        self.stop_hold_active = false;
                    }
                } else if lead_speed_kmh < 2.0 && lead_distance_m <= 8.5 {
                    self.stop_hold_active = true;
                }

                if self.stop_hold_active {
                    target_speed_kmh = 0.0;
                } else {
                    let desired_gap_m = 8.0 + 1.2 * (current_speed_kmh / 3.6);
                    let gap_error_m = lead_distance_m - desired_gap_m;
                    let acc_upper_kmh = 35.0_f64.min(lead_speed_kmh + 6.0);
                    target_speed_kmh =
                        (lead_speed_kmh + 0.85 * gap_error_m).clamp(0.0, acc_upper_kmh);
                }
            }
            return ShieldGatedOutput {
                raw_maneuver: raw_maneuver.to_string(),
                raw_speed_level,
                maneuver: gated_maneuver,
                speed_level: gated_speed_level,
                target_speed_kmh,
                mpc_cornering_cap_kmh: 65.0,
                maneuver_confidence: maneuver_conf,
                emergency_brake: aeb_prob,
                shield_triggered,
                shield_reason,
                blind_spot_left,
                blind_spot_right,
            };
        }

        // Rule 1: Emergency Brake Priority
        if aeb_prob >= self.aeb_threshold
            || traffic_light == "red"
            || ped_distance_m <= 16.0
            || (hazard_source == "pedestrian" && lead_distance_m <= 16.0)
            || (lead_distance_m <= d_aeb_m
                && current_speed_kmh > 6.0_f64.max(lead_speed_kmh + 6.0))
        {
            gated_maneuver = "keep_lane".to_string();
            gated_speed_level = 0;
            shield_triggered = true;
            shield_reason = "rule1_emergency_brake".to_string();
        } else {
            // Rule 2: Low Confidence Fallback
            if maneuver_conf < self.confidence_threshold {
                if raw_maneuver == "change_left" || raw_maneuver == "change_right" {
                    gated_maneuver = "keep_lane".to_string();
                }
                gated_speed_level = if raw_speed_level > 0 {
                    1.max(raw_speed_level - 1)
                } else {
                    0
                };
                shield_triggered = true;
                shield_reason = "rule2_low_confidence_fallback".to_string();
            }

            // Boundary & Blind-Spot Guards
            if gated_maneuver == "change_left" && !left_lane_available {
                gated_maneuver = "keep_lane".to_string();
                shield_triggered = true;
                shield_reason = "boundary_left_unavailable".to_string();
            } else if gated_maneuver == "change_right" && !right_lane_available {
                gated_maneuver = "keep_lane".to_string();
                shield_triggered = true;
                shield_reason = "boundary_right_unavailable".to_string();
            } else if gated_maneuver == "change_left" && blind_spot_left {
                gated_maneuver = "keep_lane".to_string();
                shield_triggered = true;
                shield_reason = "blindspot_left_occupied".to_string();
            } else if gated_maneuver == "change_right" && blind_spot_right {
                gated_maneuver = "keep_lane".to_string();
                shield_triggered = true;
                shield_reason = "blindspot_right_occupied".to_string();
            }
        }

        let mut target_speed_kmh = speed_level_to_kmh(gated_speed_level);
        let road_curv_abs = road_curvature.abs();
        let effective_kappa = road_curv_abs * 0.065;
        let mpc_cornering_cap_kmh =
            65.0_f64.min(3.6 * (2.85 / effective_kappa.max(1e-3)).sqrt());
        if road_curv_abs > 0.22 && gated_speed_level > 0 {
            target_speed_kmh = target_speed_kmh.min(20.0_f64.max(mpc_cornering_cap_kmh));
        }

        if gated_maneuver == "follow_lead" && gated_speed_level >= 2 {
            target_speed_kmh = target_speed_kmh.min(30.0);
        }
        if lead_distance_m < 52.0 && gated_speed_level > 0 {
            let smooth_cap = 30.0 + (lead_distance_m - 30.0).max(0.0) * 1.4;
            target_speed_kmh = target_speed_kmh.min(smooth_cap);
        }
        if lead_distance_m < 38.0 && gated_speed_level > 0 {
            let eff_lead_speed_kmh = if has_lead_speed {
                lead_speed_kmh
            } else {
                target_speed_kmh
            };
            if self.stop_hold_active {
                if eff_lead_speed_kmh >= 3.5 || lead_distance_m >= 9.5 {
                    self.stop_hold_active = false;
                }
            } else if eff_lead_speed_kmh < 2.0 && lead_distance_m <= 8.5 {
                self.stop_hold_active = true;
            }

            if self.stop_hold_active {
                target_speed_kmh = 0.0;
            } else {
                let desired_gap_m = 8.0 + 1.2 * (current_speed_kmh / 3.6);
                let gap_error_m = lead_distance_m - desired_gap_m;
                let acc_upper_kmh = 35.0_f64.min(eff_lead_speed_kmh + 6.0);
                target_speed_kmh =
                    (eff_lead_speed_kmh + 0.85 * gap_error_m).clamp(0.0, acc_upper_kmh);
            }
        }

        ShieldGatedOutput {
            raw_maneuver: raw_maneuver.to_string(),
            raw_speed_level,
            maneuver: gated_maneuver,
            speed_level: gated_speed_level,
            target_speed_kmh,
            mpc_cornering_cap_kmh: round_n(mpc_cornering_cap_kmh, 1),
            maneuver_confidence: maneuver_conf,
            emergency_brake: aeb_prob,
            shield_triggered,
            shield_reason,
            blind_spot_left,
            blind_spot_right,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_control_core(
        &mut self,
        raw_maneuver: &str,
        raw_speed_level: i64,
        maneuver_conf: f64,
        aeb_prob: f64,
        wp2_x: f64,
        wp2_y: f64,
        has_wp2: bool,
        lead_distance_m: f64,
        lead_speed_kmh: f64,
        has_lead_speed: bool,
        ped_distance_m: f64,
        current_speed_kmh: f64,
        lane_offset_m: f64,
        heading_error: f64,
        traffic_light: &str,
        hazard_source: &str,
        wet_road: bool,
        a_max_mps2: f64,
        unc_margin_m: f64,
        left_lane_available: bool,
        right_lane_available: bool,
        blind_spot_left: bool,
        blind_spot_right: bool,
        road_curvature: f64,
        dt: f64,
        shield_enabled: bool,
    ) -> ControlStepOutput {
        let gated = self.apply_safety_shield_core(
            raw_maneuver,
            raw_speed_level,
            maneuver_conf,
            aeb_prob,
            lead_distance_m,
            lead_speed_kmh,
            has_lead_speed,
            ped_distance_m,
            current_speed_kmh,
            traffic_light,
            hazard_source,
            wet_road,
            a_max_mps2,
            unc_margin_m,
            left_lane_available,
            right_lane_available,
            blind_spot_left,
            blind_spot_right,
            road_curvature,
            shield_enabled,
        );

        let target_offset_m = if self.use_mock && gated.maneuver == "change_left" {
            self.lane_width * 0.85
        } else if self.use_mock && gated.maneuver == "change_right" {
            -self.lane_width * 0.85
        } else if !self.use_mock
            && road_curvature.abs() > 0.25
            && !left_lane_available
            && right_lane_available
        {
            -0.18
        } else {
            0.0
        };

        let lateral_error =
            (target_offset_m - lane_offset_m) - (self.heading_weight * heading_error);
        let pid_steering = self.lateral_pid.step(lateral_error, dt).clamp(-1.0, 1.0);
        let mpc_ff = if self.use_mock {
            0.0
        } else {
            compute_mpc_feedforward_core(road_curvature, lane_offset_m, wp2_x, wp2_y, has_wp2)
        };
        let raw_target_steering = (pid_steering + 0.12 * mpc_ff).clamp(-1.0, 1.0);

        let steer_from = self.current_steering;
        let road_curv_abs = road_curvature.abs();
        let (max_delta, alpha) = if self.use_mock {
            (1.0, 1.0)
        } else if road_curv_abs >= 0.35 {
            let speed_damping = 1.0 + 0.012 * current_speed_kmh.max(0.0);
            ((3.5 * (1.1 * dt) / speed_damping).min(0.22), 0.82)
        } else {
            let speed_damping = 1.0 + 0.012 * current_speed_kmh.max(0.0);
            ((1.1 * dt) / speed_damping, 0.55)
        };

        let steer_step = ((raw_target_steering - steer_from) * alpha).clamp(-max_delta, max_delta);
        let steering = (steer_from + steer_step).clamp(-1.0, 1.0);
        self.current_steering = steering;

        let accel_from = self.current_throttle_brake;
        let throttle_brake = if gated.shield_reason == "rule1_emergency_brake"
            || gated.shield_reason == "model_aeb_only"
        {
            -1.0
        } else {
            let speed_error = gated.target_speed_kmh - current_speed_kmh;
            let raw_tb = self.longitudinal_pid.step(speed_error, dt).clamp(-1.0, 1.0);
            if self.use_mock {
                raw_tb
            } else {
                let tb_step = (raw_tb - accel_from).clamp(-0.25, 0.20);
                (accel_from + tb_step).clamp(-1.0, 1.0)
            }
        };
        self.current_throttle_brake = throttle_brake;

        let mut steer_ramp_substeps = [0.0_f64; 5];
        for (idx, item) in steer_ramp_substeps.iter_mut().enumerate() {
            let k = (idx + 1) as f64;
            *item = round_n(steer_from + (k / 5.0) * (steering - steer_from), 4);
        }

        ControlStepOutput {
            steering,
            throttle_brake,
            steer_from: round_n(steer_from, 4),
            steer_to: round_n(steering, 4),
            steer_target: round_n(raw_target_steering, 4),
            steer_delta: round_n(steering - steer_from, 4),
            mpc_feedforward: round_n(mpc_ff, 4),
            steer_ramp_substeps,
            gated,
        }
    }
}

#[inline]
pub fn ols_5step_slope_mps_core(window: &[f64], dt: f64) -> f64 {
    let n_w = window.len();
    if n_w < 2 {
        return 0.0;
    }
    let n_f = n_w as f64;
    let t_mean = (n_f - 1.0) * 0.5 * dt;
    let d_mean = window.iter().sum::<f64>() / n_f;
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (i, &d) in window.iter().enumerate() {
        let dt_i = (i as f64) * dt - t_mean;
        num += dt_i * (d - d_mean);
        den += dt_i * dt_i;
    }
    num / den.max(1e-6)
}

#[inline]
pub fn timestamp_ols_slope_mps_core(timestamps: &[f64], distances: &[f64]) -> f64 {
    let n = timestamps.len().min(distances.len());
    if n < 2 {
        return 0.0;
    }
    let n_f = n as f64;
    let t_mean = timestamps[..n].iter().sum::<f64>() / n_f;
    let d_mean = distances[..n].iter().sum::<f64>() / n_f;
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..n {
        let dt_i = timestamps[i] - t_mean;
        num += dt_i * (distances[i] - d_mean);
        den += dt_i * dt_i;
    }
    num / den.max(1e-6)
}

#[derive(Clone, Debug)]
pub struct LeadSpeedOlsEstimatorCore {
    pub window: Vec<(f64, f64)>,
    pub obs_time_s: f64,
    pub prev_est_v_lead: f64,
}

impl LeadSpeedOlsEstimatorCore {
    pub fn new(initial_v_lead_kmh: f64) -> Self {
        Self {
            window: Vec::with_capacity(6),
            obs_time_s: 0.0,
            prev_est_v_lead: initial_v_lead_kmh,
        }
    }

    pub fn reset(&mut self, initial_v_lead_kmh: f64) {
        self.window.clear();
        self.obs_time_s = 0.0;
        self.prev_est_v_lead = initial_v_lead_kmh;
    }

    pub fn update(
        &mut self,
        pred_d: f64,
        pred_rr_mps: f64,
        ego_v_kmh: f64,
        elapsed_steps: usize,
        dt: f64,
    ) -> f64 {
        let steps = elapsed_steps.max(1) as f64;
        self.obs_time_s += steps * dt;

        if let Some(&(_, last_d)) = self.window.last() {
            if (pred_d - last_d).abs() > 6.0 {
                self.window.clear();
            }
        }
        self.window.push((self.obs_time_s, pred_d));
        if self.window.len() > 5 {
            self.window.remove(0);
        }

        let dd_dt_reg = if self.window.len() < 2 {
            pred_rr_mps
        } else {
            let n_f = self.window.len() as f64;
            let t_mean = self.window.iter().map(|pt| pt.0).sum::<f64>() / n_f;
            let d_mean = self.window.iter().map(|pt| pt.1).sum::<f64>() / n_f;
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for &(t_i, d_i) in &self.window {
                let dt_i = t_i - t_mean;
                num += dt_i * (d_i - d_mean);
                den += dt_i * dt_i;
            }
            num / den.max(1e-6)
        };

        let dd_dt_fused = 0.70 * dd_dt_reg + 0.30 * pred_rr_mps;
        let raw_v_lead = ego_v_kmh + 3.6 * dd_dt_fused;
        let est_v_lead = (0.65 * self.prev_est_v_lead + 0.35 * raw_v_lead).clamp(0.0, 60.0);
        self.prev_est_v_lead = est_v_lead;
        est_v_lead
    }
}

pub fn count_sustained_aeb_events_core(
    trace: &[f64],
    threshold: f64,
    min_contiguous_ticks: usize,
    max_gap_bridge: usize,
) -> (usize, usize) {
    let active: Vec<u8> = trace
        .iter()
        .map(|&u| if u <= threshold { 1 } else { 0 })
        .collect();
    let total_ticks: usize = active.iter().map(|&v| v as usize).sum();

    let mut bridged = active;
    let n = bridged.len();
    let mut i = 0;
    while i < n {
        if bridged[i] == 1 {
            let mut j = i + 1;
            while j < n && bridged[j] == 0 && (j - i - 1) <= max_gap_bridge {
                j += 1;
            }
            if j < n && bridged[j] == 1 && (j - i - 1) <= max_gap_bridge {
                for item in bridged.iter_mut().take(j).skip(i + 1) {
                    *item = 1;
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }

    let mut events = 0_usize;
    let mut run_len = 0_usize;
    for val in bridged {
        if val == 1 {
            run_len += 1;
        } else {
            if run_len >= min_contiguous_ticks {
                events += 1;
            }
            run_len = 0;
        }
    }
    if run_len >= min_contiguous_ticks {
        events += 1;
    }
    (events, total_ticks)
}

pub fn compute_polyline_tangents_core(
    xyz_list: &[(f64, f64, f64)],
) -> Vec<(f64, f64, f64, f64)> {
    let n = xyz_list.len();
    if n == 0 {
        return Vec::new();
    }
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y, z) = xyz_list[i];
        let yaw = if i + 1 < n {
            let (nx, ny, _) = xyz_list[i + 1];
            (ny - y).atan2(nx - x)
        } else if i > 0 {
            let (px, py, _) = xyz_list[i - 1];
            (y - py).atan2(x - px)
        } else {
            0.0
        };
        pts.push((x, y, z, yaw));
    }
    pts
}

pub fn quintic_s_curve_blend_core(
    orig_xyz: &[(f64, f64, f64)],
    adj_xyz: &[(f64, f64, f64)],
    n_blend: usize,
) -> Vec<(f64, f64, f64)> {
    let eff_blend = n_blend.min(orig_xyz.len()).min(adj_xyz.len());
    let denom = (eff_blend.saturating_sub(1)).max(1) as f64;
    let mut out = Vec::with_capacity(adj_xyz.len());
    for (k, &(ax, ay, az)) in adj_xyz.iter().enumerate() {
        if k < eff_blend {
            let (ox, oy, oz) = orig_xyz[k];
            let tau = (k as f64) / denom;
            let s = (tau * tau * tau) * (10.0 + tau * (-15.0 + 6.0 * tau));
            let bx = (1.0 - s) * ox + s * ax;
            let by = (1.0 - s) * oy + s * ay;
            let bz = (1.0 - s) * oz + s * az;
            out.push((bx, by, bz));
        } else {
            out.push((ax, ay, az));
        }
    }
    out
}

pub fn sample_continuous_route_pose_core(
    pts: &[(f64, f64, f64, f64)],
    x: f64,
    y: f64,
    start_idx: usize,
    lookahead_m: f64,
) -> (usize, f64, f64, f64, f64, f64, f64) {
    if pts.is_empty() {
        return (0, x, y, 0.0, x + lookahead_m, y, 0.0);
    }
    let n = pts.len();
    let mut best_i = start_idx.min(n - 1);
    let mut best_d = 1e9_f64;
    let end_i = n.min(best_i + 12);
    for (idx, pt) in pts.iter().enumerate().take(end_i).skip(best_i) {
        let d = (x - pt.0).hypot(y - pt.1);
        if d < best_d {
            best_d = d;
            best_i = idx;
        }
    }

    let (proj_x, proj_y, proj_yaw) = if best_i + 1 < n {
        let (x0, y0, _, yaw0) = pts[best_i];
        let (x1, y1, _, _) = pts[best_i + 1];
        let vx = x1 - x0;
        let vy = y1 - y0;
        let seg_len_sq = vx * vx + vy * vy;
        let t_seg = if seg_len_sq > 1e-4 {
            (((x - x0) * vx + (y - y0) * vy) / seg_len_sq).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (x0 + t_seg * vx, y0 + t_seg * vy, yaw0)
    } else {
        let (x0, y0, _, yaw0) = pts[best_i];
        (x0, y0, yaw0)
    };

    let mut rem = lookahead_m;
    let mut cx = proj_x;
    let mut cy = proj_y;
    let mut curr_idx = best_i;
    let mut look_x = pts[n - 1].0;
    let mut look_y = pts[n - 1].1;
    while curr_idx + 1 < n {
        let (nx, ny, _, _) = pts[curr_idx + 1];
        let ds = (nx - cx).hypot(ny - cy);
        if ds >= rem && ds > 1e-4 {
            let ratio = rem / ds;
            look_x = cx + ratio * (nx - cx);
            look_y = cy + ratio * (ny - cy);
            break;
        }
        rem -= ds;
        cx = nx;
        cy = ny;
        curr_idx += 1;
        look_x = cx;
        look_y = cy;
    }

    let ref_yaw = (look_y - proj_y).atan2(look_x - proj_x);
    (best_i, proj_x, proj_y, proj_yaw, look_x, look_y, ref_yaw)
}

// ============================================================================
// PyO3 Python Bindings
// ============================================================================

#[pyclass]
pub struct RustScalarPID {
    inner: ScalarPidCore,
}

#[pymethods]
impl RustScalarPID {
    #[new]
    #[pyo3(signature = (kp, ki, kd, integral_limit=5.0))]
    pub fn new(kp: f64, ki: f64, kd: f64, integral_limit: f64) -> Self {
        Self {
            inner: ScalarPidCore::new(kp, ki, kd, integral_limit),
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[pyo3(signature = (error, dt=0.05))]
    pub fn step(&mut self, error: f64, dt: f64) -> f64 {
        self.inner.step(error, dt)
    }

    #[getter]
    pub fn integral(&self) -> f64 {
        self.inner.integral
    }

    #[setter]
    pub fn set_integral(&mut self, val: f64) {
        self.inner.integral = val;
    }

    #[getter]
    pub fn prev_error(&self) -> Option<f64> {
        self.inner.prev_error
    }

    #[setter]
    pub fn set_prev_error(&mut self, val: Option<f64>) {
        self.inner.prev_error = val;
    }
}

#[pyclass]
pub struct RustLeadSpeedEstimator {
    inner: LeadSpeedOlsEstimatorCore,
}

#[pymethods]
impl RustLeadSpeedEstimator {
    #[new]
    #[pyo3(signature = (initial_v_lead_kmh=35.0))]
    pub fn new(initial_v_lead_kmh: f64) -> Self {
        Self {
            inner: LeadSpeedOlsEstimatorCore::new(initial_v_lead_kmh),
        }
    }

    #[pyo3(signature = (initial_v_lead_kmh=35.0))]
    pub fn reset(&mut self, initial_v_lead_kmh: f64) {
        self.inner.reset(initial_v_lead_kmh);
    }

    #[pyo3(signature = (pred_d, pred_rr_mps, ego_v_kmh, elapsed_steps=1, dt=0.05))]
    pub fn update(
        &mut self,
        pred_d: f64,
        pred_rr_mps: f64,
        ego_v_kmh: f64,
        elapsed_steps: usize,
        dt: f64,
    ) -> f64 {
        self.inner
            .update(pred_d, pred_rr_mps, ego_v_kmh, elapsed_steps, dt)
    }
}

fn extract_f64(dict: &PyDict, key: &str, default: f64) -> f64 {
    match dict.get_item(key) {
        Ok(Some(v)) => v.extract::<f64>().unwrap_or(default),
        _ => default,
    }
}

fn extract_f64_2keys(dict: &PyDict, key1: &str, key2: &str, default: f64) -> f64 {
    if let Ok(Some(v)) = dict.get_item(key1) {
        if let Ok(val) = v.extract::<f64>() {
            return val;
        }
    }
    if let Ok(Some(v)) = dict.get_item(key2) {
        if let Ok(val) = v.extract::<f64>() {
            return val;
        }
    }
    default
}

fn extract_i64(dict: &PyDict, key: &str, default: i64) -> i64 {
    match dict.get_item(key) {
        Ok(Some(v)) => v.extract::<i64>().unwrap_or(default),
        _ => default,
    }
}

fn extract_bool(dict: &PyDict, key: &str, default: bool) -> bool {
    match dict.get_item(key) {
        Ok(Some(v)) => v.extract::<bool>().unwrap_or(default),
        _ => default,
    }
}

fn extract_str(dict: &PyDict, key: &str, default: &str) -> String {
    match dict.get_item(key) {
        Ok(Some(v)) => v.extract::<String>().unwrap_or_else(|_| default.to_string()),
        _ => default.to_string(),
    }
}

fn extract_wp2(decision: &PyDict) -> (f64, f64, bool) {
    if let Ok(Some(traj_any)) = decision.get_item("predicted_trajectory") {
        if let Ok(traj_list) = traj_any.downcast::<PyList>() {
            if traj_list.len() >= 3 {
                if let Ok(wp2_any) = traj_list.get_item(2) {
                    if let Ok(wp2_seq) = wp2_any.extract::<Vec<f64>>() {
                        if wp2_seq.len() >= 2 {
                            return (wp2_seq[0], wp2_seq[1], true);
                        }
                    }
                }
            }
        }
    }
    (0.0, 0.0, false)
}

#[pyclass]
pub struct RustSafetyShieldPID {
    inner: SafetyShieldPidCore,
}

#[pymethods]
impl RustSafetyShieldPID {
    #[new]
    #[pyo3(signature = (
        use_mock=false,
        aeb_threshold=0.70,
        confidence_threshold=0.80,
        lane_width=3.5,
        heading_weight=1.25,
        max_steer_rate_per_sec=3.2,
        steer_smoothing_alpha=0.72,
        lat_kp=0.45,
        lat_ki=0.01,
        lat_kd=0.18,
        lon_kp=0.08,
        lon_ki=0.005,
        lon_kd=0.02
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        use_mock: bool,
        aeb_threshold: f64,
        confidence_threshold: f64,
        lane_width: f64,
        heading_weight: f64,
        max_steer_rate_per_sec: f64,
        steer_smoothing_alpha: f64,
        lat_kp: f64,
        lat_ki: f64,
        lat_kd: f64,
        lon_kp: f64,
        lon_ki: f64,
        lon_kd: f64,
    ) -> Self {
        Self {
            inner: SafetyShieldPidCore::new(
                use_mock,
                aeb_threshold,
                confidence_threshold,
                lane_width,
                heading_weight,
                max_steer_rate_per_sec,
                steer_smoothing_alpha,
                lat_kp,
                lat_ki,
                lat_kd,
                lon_kp,
                lon_ki,
                lon_kd,
            ),
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    pub fn stop_hold_active(&self) -> bool {
        self.inner.stop_hold_active
    }

    #[setter]
    pub fn set_stop_hold_active(&mut self, val: bool) {
        self.inner.stop_hold_active = val;
    }

    #[getter]
    pub fn current_steering(&self) -> f64 {
        self.inner.current_steering
    }

    #[setter]
    pub fn set_current_steering(&mut self, val: f64) {
        self.inner.current_steering = val;
    }

    #[getter]
    pub fn current_throttle_brake(&self) -> f64 {
        self.inner.current_throttle_brake
    }

    #[setter]
    pub fn set_current_throttle_brake(&mut self, val: f64) {
        self.inner.current_throttle_brake = val;
    }

    #[pyo3(signature = (decision, telemetry, shield_enabled=true))]
    pub fn apply_safety_shield<'py>(
        &mut self,
        py: Python<'py>,
        decision: &PyDict,
        telemetry: &PyDict,
        shield_enabled: bool,
    ) -> PyResult<&'py PyDict> {
        let raw_maneuver = extract_str(decision, "maneuver", "keep_lane");
        let raw_speed_level = extract_i64(decision, "speed_level", 2);
        let maneuver_conf = extract_f64(decision, "maneuver_confidence", 1.0);
        let aeb_prob = extract_f64(decision, "emergency_brake", 0.0);

        let lead_distance_m =
            extract_f64_2keys(telemetry, "lead_distance_m", "lead_distance", 60.0);
        let has_lead_speed = telemetry.get_item("lead_speed_kmh").ok().flatten().is_some();
        let lead_speed_kmh = extract_f64(telemetry, "lead_speed_kmh", 60.0);
        let ped_distance_m = extract_f64_2keys(
            telemetry,
            "pedestrian_distance_m",
            "pedestrian_distance",
            100.0,
        );
        let current_speed_kmh = extract_f64(telemetry, "speed_kmh", 0.0);
        let traffic_light = extract_str(telemetry, "traffic_light", "green");
        let hazard_source = extract_str(telemetry, "hazard_source", "none");
        let wet_road = extract_bool(telemetry, "wet_road", false);
        let a_max_mps2 = extract_f64(telemetry, "a_max_mps2", if wet_road { 4.5 } else { 6.0 });
        let unc_margin_m = extract_f64(
            telemetry,
            "uncertainty_margin_m",
            if wet_road { 1.8 } else { 0.0 },
        );
        let left_lane_available = extract_bool(telemetry, "left_lane_available", true);
        let right_lane_available = extract_bool(telemetry, "right_lane_available", true);
        let blind_spot_left = extract_bool(telemetry, "blind_spot_left", false);
        let blind_spot_right = extract_bool(telemetry, "blind_spot_right", false);
        let road_curvature = extract_f64(telemetry, "road_curvature", 0.0);

        let out = self.inner.apply_safety_shield_core(
            &raw_maneuver,
            raw_speed_level,
            maneuver_conf,
            aeb_prob,
            lead_distance_m,
            lead_speed_kmh,
            has_lead_speed,
            ped_distance_m,
            current_speed_kmh,
            &traffic_light,
            &hazard_source,
            wet_road,
            a_max_mps2,
            unc_margin_m,
            left_lane_available,
            right_lane_available,
            blind_spot_left,
            blind_spot_right,
            road_curvature,
            shield_enabled,
        );

        let dict = PyDict::new(py);
        dict.set_item("raw_maneuver", out.raw_maneuver)?;
        dict.set_item("raw_speed_level", out.raw_speed_level)?;
        dict.set_item("maneuver", out.maneuver)?;
        dict.set_item("speed_level", out.speed_level)?;
        dict.set_item("target_speed_kmh", out.target_speed_kmh)?;
        dict.set_item("mpc_cornering_cap_kmh", out.mpc_cornering_cap_kmh)?;
        dict.set_item("maneuver_confidence", out.maneuver_confidence)?;
        dict.set_item("emergency_brake", out.emergency_brake)?;
        dict.set_item("shield_triggered", out.shield_triggered)?;
        dict.set_item("shield_reason", out.shield_reason)?;
        dict.set_item("blind_spot_left", out.blind_spot_left)?;
        dict.set_item("blind_spot_right", out.blind_spot_right)?;
        Ok(dict)
    }

    #[pyo3(signature = (decision, telemetry, dt=0.05, shield_enabled=true))]
    pub fn compute_control<'py>(
        &mut self,
        py: Python<'py>,
        decision: &PyDict,
        telemetry: &PyDict,
        dt: f64,
        shield_enabled: bool,
    ) -> PyResult<((f64, f64), &'py PyDict)> {
        let raw_maneuver = extract_str(decision, "maneuver", "keep_lane");
        let raw_speed_level = extract_i64(decision, "speed_level", 2);
        let maneuver_conf = extract_f64(decision, "maneuver_confidence", 1.0);
        let aeb_prob = extract_f64(decision, "emergency_brake", 0.0);
        let (wp2_x, wp2_y, has_wp2) = extract_wp2(decision);

        let lead_distance_m =
            extract_f64_2keys(telemetry, "lead_distance_m", "lead_distance", 60.0);
        let has_lead_speed = telemetry.get_item("lead_speed_kmh").ok().flatten().is_some();
        let lead_speed_kmh = extract_f64(telemetry, "lead_speed_kmh", 60.0);
        let ped_distance_m = extract_f64_2keys(
            telemetry,
            "pedestrian_distance_m",
            "pedestrian_distance",
            100.0,
        );
        let current_speed_kmh = extract_f64(telemetry, "speed_kmh", 0.0);
        let lane_offset_m = extract_f64(telemetry, "lane_offset_m", 0.0);
        let heading_error = extract_f64(telemetry, "heading_error", 0.0);
        let traffic_light = extract_str(telemetry, "traffic_light", "green");
        let hazard_source = extract_str(telemetry, "hazard_source", "none");
        let wet_road = extract_bool(telemetry, "wet_road", false);
        let a_max_mps2 = extract_f64(telemetry, "a_max_mps2", if wet_road { 4.5 } else { 6.0 });
        let unc_margin_m = extract_f64(
            telemetry,
            "uncertainty_margin_m",
            if wet_road { 1.8 } else { 0.0 },
        );
        let left_lane_available = extract_bool(telemetry, "left_lane_available", true);
        let right_lane_available = extract_bool(telemetry, "right_lane_available", true);
        let blind_spot_left = extract_bool(telemetry, "blind_spot_left", false);
        let blind_spot_right = extract_bool(telemetry, "blind_spot_right", false);
        let road_curvature = extract_f64(telemetry, "road_curvature", 0.0);

        let res = self.inner.compute_control_core(
            &raw_maneuver,
            raw_speed_level,
            maneuver_conf,
            aeb_prob,
            wp2_x,
            wp2_y,
            has_wp2,
            lead_distance_m,
            lead_speed_kmh,
            has_lead_speed,
            ped_distance_m,
            current_speed_kmh,
            lane_offset_m,
            heading_error,
            &traffic_light,
            &hazard_source,
            wet_road,
            a_max_mps2,
            unc_margin_m,
            left_lane_available,
            right_lane_available,
            blind_spot_left,
            blind_spot_right,
            road_curvature,
            dt,
            shield_enabled,
        );

        let dict = PyDict::new(py);
        dict.set_item("raw_maneuver", res.gated.raw_maneuver)?;
        dict.set_item("raw_speed_level", res.gated.raw_speed_level)?;
        dict.set_item("maneuver", res.gated.maneuver)?;
        dict.set_item("speed_level", res.gated.speed_level)?;
        dict.set_item("target_speed_kmh", res.gated.target_speed_kmh)?;
        dict.set_item("mpc_cornering_cap_kmh", res.gated.mpc_cornering_cap_kmh)?;
        dict.set_item("maneuver_confidence", res.gated.maneuver_confidence)?;
        dict.set_item("emergency_brake", res.gated.emergency_brake)?;
        dict.set_item("shield_triggered", res.gated.shield_triggered)?;
        dict.set_item("shield_reason", res.gated.shield_reason)?;
        dict.set_item("blind_spot_left", res.gated.blind_spot_left)?;
        dict.set_item("blind_spot_right", res.gated.blind_spot_right)?;

        dict.set_item("steer_from", res.steer_from)?;
        dict.set_item("steer_to", res.steer_to)?;
        dict.set_item("steer_target", res.steer_target)?;
        dict.set_item("steer_delta", res.steer_delta)?;
        dict.set_item("mpc_feedforward", res.mpc_feedforward)?;
        dict.set_item("steer_ramp_substeps", res.steer_ramp_substeps.to_vec())?;
        dict.set_item("steering", res.steering)?;
        dict.set_item("throttle_brake", res.throttle_brake)?;

        let carla_ctrl = PyDict::new(py);
        let steer_c = res.steering.clamp(-1.0, 1.0);
        let tb_c = res.throttle_brake.clamp(-1.0, 1.0);
        carla_ctrl.set_item("steer", round_n(steer_c, 4))?;
        carla_ctrl.set_item("throttle", round_n(if tb_c >= 0.0 { tb_c } else { 0.0 }, 4))?;
        carla_ctrl.set_item("brake", round_n(if tb_c < 0.0 { -tb_c } else { 0.0 }, 4))?;
        carla_ctrl.set_item("hand_brake", false)?;
        carla_ctrl.set_item("reverse", false)?;
        dict.set_item("carla_control", carla_ctrl)?;

        Ok(((res.steering, res.throttle_brake), dict))
    }
}

#[pyfunction]
#[pyo3(signature = (current_speed_kmh, lead_speed_kmh, a_max_mps2=6.0, uncertainty_margin_m=0.0))]
pub fn compute_kinematic_aeb_distance(
    current_speed_kmh: f64,
    lead_speed_kmh: f64,
    a_max_mps2: f64,
    uncertainty_margin_m: f64,
) -> f64 {
    compute_kinematic_aeb_distance_core(
        current_speed_kmh,
        lead_speed_kmh,
        a_max_mps2,
        uncertainty_margin_m,
    )
}

#[pyfunction]
#[pyo3(signature = (road_curv, lane_offset_m, wp2_x=0.0, wp2_y=0.0, has_wp2=false))]
pub fn compute_mpc_feedforward(
    road_curv: f64,
    lane_offset_m: f64,
    wp2_x: f64,
    wp2_y: f64,
    has_wp2: bool,
) -> f64 {
    compute_mpc_feedforward_core(road_curv, lane_offset_m, wp2_x, wp2_y, has_wp2)
}

#[pyfunction]
#[pyo3(signature = (window, dt=0.05))]
pub fn ols_5step_slope_mps(window: Vec<f64>, dt: f64) -> f64 {
    ols_5step_slope_mps_core(&window, dt)
}

#[pyfunction]
pub fn timestamp_ols_slope_mps(timestamps: Vec<f64>, distances: Vec<f64>) -> f64 {
    timestamp_ols_slope_mps_core(&timestamps, &distances)
}

#[pyfunction]
#[pyo3(signature = (throttle_brake_trace, threshold=-0.50, min_contiguous_ticks=3, max_gap_bridge=2))]
pub fn count_sustained_aeb_events(
    throttle_brake_trace: Vec<f64>,
    threshold: f64,
    min_contiguous_ticks: usize,
    max_gap_bridge: usize,
) -> (usize, usize) {
    count_sustained_aeb_events_core(
        &throttle_brake_trace,
        threshold,
        min_contiguous_ticks,
        max_gap_bridge,
    )
}

#[pyfunction]
pub fn compute_polyline_tangents(
    xyz_list: Vec<(f64, f64, f64)>,
) -> Vec<(f64, f64, f64, f64)> {
    compute_polyline_tangents_core(&xyz_list)
}

#[pyfunction]
#[pyo3(signature = (orig_xyz, adj_xyz, n_blend=14))]
pub fn quintic_s_curve_blend(
    orig_xyz: Vec<(f64, f64, f64)>,
    adj_xyz: Vec<(f64, f64, f64)>,
    n_blend: usize,
) -> Vec<(f64, f64, f64)> {
    quintic_s_curve_blend_core(&orig_xyz, &adj_xyz, n_blend)
}

#[pyfunction]
#[pyo3(signature = (pts, x, y, start_idx=0, lookahead_m=8.0))]
pub fn sample_continuous_route_pose(
    pts: Vec<(f64, f64, f64, f64)>,
    x: f64,
    y: f64,
    start_idx: usize,
    lookahead_m: f64,
) -> (usize, f64, f64, f64, f64, f64, f64) {
    sample_continuous_route_pose_core(&pts, x, y, start_idx, lookahead_m)
}

#[pyfunction]
#[pyo3(signature = (iterations=10000))]
pub fn benchmark_rust_control_loop_us(iterations: usize) -> f64 {
    let n = iterations.max(1);
    let mut ctrl = SafetyShieldPidCore::new(
        false, 0.70, 0.80, 3.5, 1.25, 3.2, 0.72, 0.45, 0.01, 0.18, 0.08, 0.005, 0.02,
    );
    let mut ols = LeadSpeedOlsEstimatorCore::new(24.0);
    let route_xyz: Vec<(f64, f64, f64)> = (0..50).map(|i| (i as f64 * 2.0, 0.0, 0.0)).collect();
    let pts = compute_polyline_tangents_core(&route_xyz);

    let t0 = Instant::now();
    for step in 0..n {
        let d_obs = 22.0 - (step % 5) as f64 * 0.2;
        let v_lead = ols.update(d_obs, -4.0, 28.0, 1, 0.05);
        let (_, _, _, _, lx, ly, _) =
            sample_continuous_route_pose_core(&pts, (step % 40) as f64, 0.05, 0, 8.0);
        let _ = ctrl.compute_control_core(
            "follow_lead",
            2,
            0.95,
            0.02,
            lx,
            ly,
            true,
            d_obs,
            v_lead,
            true,
            100.0,
            28.0,
            0.05,
            0.01,
            "green",
            "vehicle",
            false,
            6.0,
            0.0,
            true,
            true,
            false,
            false,
            0.12,
            0.05,
            true,
        );
    }
    let elapsed_us = t0.elapsed().as_nanos() as f64 / 1000.0;
    elapsed_us / (n as f64)
}

#[pyclass]
pub struct RustExpertLabeler {
    pub lane_change_steer_thresh: f64,
    pub aeb_decel_thresh: f64,
    pub aeb_distance_thresh_m: f64,
    pub follow_distance_thresh_m: f64,
    pub curve_curvature_thresh: f64,
}

#[pymethods]
impl RustExpertLabeler {
    #[new]
    #[pyo3(signature = (
        lane_change_steer_thresh=0.22,
        aeb_decel_thresh=-0.75,
        aeb_distance_thresh_m=13.5,
        follow_distance_thresh_m=30.0,
        curve_curvature_thresh=0.40
    ))]
    pub fn new(
        lane_change_steer_thresh: f64,
        aeb_decel_thresh: f64,
        aeb_distance_thresh_m: f64,
        follow_distance_thresh_m: f64,
        curve_curvature_thresh: f64,
    ) -> Self {
        Self {
            lane_change_steer_thresh,
            aeb_decel_thresh,
            aeb_distance_thresh_m,
            follow_distance_thresh_m,
            curve_curvature_thresh,
        }
    }

    pub fn label_step(
        &self,
        telemetry: &PyDict,
        expert_action: (f64, f64),
        info: &PyDict,
    ) -> (String, i64, i64, i64) {
        let (steering, throttle_brake) = expert_action;
        let speed_kmh = extract_f64(telemetry, "speed_kmh", 0.0);
        let left_avail = extract_bool(telemetry, "left_lane_available", false);
        let right_avail = extract_bool(telemetry, "right_lane_available", false);
        let nav_command = extract_str(telemetry, "nav_command", "go_straight");

        let lead_distance_m = extract_f64(
            info,
            "lead_distance_m",
            extract_f64(telemetry, "lead_distance_m", 60.0),
        );
        let road_curvature = extract_f64(
            info,
            "road_curvature",
            extract_f64(telemetry, "road_curvature", 0.0),
        )
        .abs();
        let scenario_event = extract_str(info, "scenario_event", "");
        let traffic_light = extract_str(
            telemetry,
            "traffic_light",
            &extract_str(info, "traffic_light", "green"),
        );
        let hazard_source = extract_str(
            telemetry,
            "hazard_source",
            &extract_str(info, "hazard_source", "none"),
        );

        let is_real_sim = scenario_event == "real_metadrive" || scenario_event == "real_carla";
        let is_aeb = scenario_event == "emergency_brake"
            || traffic_light == "red"
            || (hazard_source == "pedestrian" && lead_distance_m <= 16.0)
            || (!is_real_sim
                && throttle_brake <= self.aeb_decel_thresh
                && lead_distance_m <= self.aeb_distance_thresh_m)
            || (lead_distance_m < 9.5 && speed_kmh > 15.0);
        let emergency_brake: i64 = if is_aeb { 1 } else { 0 };

        let expert_maneuver_info = extract_str(info, "expert_maneuver", "");
        let expert_maneuver = if !expert_maneuver_info.is_empty() {
            expert_maneuver_info
        } else {
            extract_str(telemetry, "expert_maneuver", "")
        };

        let maneuver = if matches!(
            scenario_event.as_str(),
            "keep_lane" | "change_left" | "change_right" | "follow_lead"
        ) {
            scenario_event.clone()
        } else if expert_maneuver == "change_left" || expert_maneuver == "change_right" {
            expert_maneuver
        } else if is_aeb {
            "follow_lead".to_string()
        } else if !is_real_sim
            && steering >= self.lane_change_steer_thresh
            && left_avail
            && road_curvature < self.curve_curvature_thresh
        {
            "change_left".to_string()
        } else if !is_real_sim
            && steering <= -self.lane_change_steer_thresh
            && right_avail
            && road_curvature < self.curve_curvature_thresh
        {
            "change_right".to_string()
        } else if lead_distance_m <= self.follow_distance_thresh_m {
            "follow_lead".to_string()
        } else {
            "keep_lane".to_string()
        };

        let maneuver_idx: i64 = match maneuver.as_str() {
            "keep_lane" => 0,
            "change_left" => 1,
            "change_right" => 2,
            "follow_lead" => 3,
            _ => 0,
        };

        let speed_level: i64 = if emergency_brake == 1 {
            0
        } else if scenario_event == "sharp_curve" || road_curvature >= self.curve_curvature_thresh {
            1
        } else if matches!(
            maneuver.as_str(),
            "change_left" | "change_right" | "follow_lead"
        ) || matches!(
            nav_command.as_str(),
            "merge" | "turn_left" | "turn_right"
        ) || road_curvature >= 0.22
        {
            if lead_distance_m > 16.0 {
                2
            } else {
                1
            }
        } else if lead_distance_m >= 45.0 && road_curvature < 0.22 {
            3
        } else {
            2
        };

        (maneuver, maneuver_idx, speed_level, emergency_brake)
    }
}

#[pymodule]
fn autonomous_vehicle_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustScalarPID>()?;
    m.add_class::<RustSafetyShieldPID>()?;
    m.add_class::<RustLeadSpeedEstimator>()?;
    m.add_class::<RustExpertLabeler>()?;
    m.add_function(wrap_pyfunction!(compute_kinematic_aeb_distance, m)?)?;
    m.add_function(wrap_pyfunction!(compute_mpc_feedforward, m)?)?;
    m.add_function(wrap_pyfunction!(ols_5step_slope_mps, m)?)?;
    m.add_function(wrap_pyfunction!(timestamp_ols_slope_mps, m)?)?;
    m.add_function(wrap_pyfunction!(count_sustained_aeb_events, m)?)?;
    m.add_function(wrap_pyfunction!(compute_polyline_tangents, m)?)?;
    m.add_function(wrap_pyfunction!(quintic_s_curve_blend, m)?)?;
    m.add_function(wrap_pyfunction!(sample_continuous_route_pose, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_rust_control_loop_us, m)?)?;
    Ok(())
}
