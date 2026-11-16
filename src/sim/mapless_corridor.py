"""Mapless & HD-Map-Fallback Visual/Inertial Drivable Corridor Estimator.

Estimates ego-centric lane offset (`lane_offset_m`), heading error (`heading_error`),
preview curvature (`road_curvature`), and adjacent-lane availability directly from:
  1. Inverse Perspective Mapped (IPM) Bird's-Eye-View white/yellow lane-ridge points
     fitted to a 2nd-order clothoid y_right(x) = c0 + c1*x + 0.5*c2*x^2,
  2. Unmarked-asphalt Mahalanobis drivable-freespace boundary segmentation when lane
     paint disappears (e.g., intersection boxes, construction zones, rural roads),
  3. IMU yaw-rate kinematic bicycle dead-reckoning + 3D radar lead-trail & Laya 6-WP
     E2E trajectory curvature priors during transient visual degradation.

Adds ZERO trainable neural-network parameters (preserving the 4,232,991 invariant).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOCALIZATION_MODES: Tuple[str, ...] = (
    "hd_map",
    "hybrid_fallback",
    "mapless_vision",
)


def _fit_huber_quadratic(
    xs: np.ndarray,
    ys: np.ndarray,
    prior_c0: float = 0.0,
    reg: float = 0.35,
) -> Tuple[float, float, float]:
    """Fit y(x) = c0 + c1*x + 0.5*c2*x^2 with ridge regularization & 1-step Huber reweighting."""
    n = int(xs.shape[0])
    if n < 4:
        return float(prior_c0), 0.0, 0.0

    design = np.column_stack([np.ones_like(xs), xs, 0.5 * (xs ** 2)])
    reg_mat = np.diag([reg * 0.15, reg * 0.75, reg * 1.15])
    rhs_prior = np.array([reg * 0.15 * prior_c0, 0.0, 0.0], dtype=np.float64)

    lhs = design.T @ design + reg_mat
    rhs = design.T @ ys + rhs_prior
    try:
        theta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return float(prior_c0), 0.0, 0.0

    # 1-step Huber IRLS reweighting (delta = 0.35m) to reject outlier curb/shadow pixels
    resid = np.abs(ys - (design @ theta))
    weights = np.where(resid <= 0.35, 1.0, 0.35 / np.maximum(resid, 1e-3))
    w_design = design * weights[:, None]
    lhs_w = w_design.T @ design + reg_mat
    rhs_w = w_design.T @ ys + rhs_prior
    try:
        theta = np.linalg.solve(lhs_w, rhs_w)
    except np.linalg.LinAlgError:
        pass

    c0 = float(np.clip(theta[0], -3.2, 3.2))
    c1 = float(np.clip(theta[1], -0.65, 0.65))
    c2 = float(np.clip(theta[2], -0.125, 0.125))
    return c0, c1, c2


class MaplessVisionCorridorEstimator:
    """Zero-parameter IPM clothoid + asphalt freespace + IMU corridor state estimator."""

    def __init__(
        self,
        lane_width_m: float = 3.5,
        camera_fov_deg: float = 75.0,
        cam_height_m: float = 1.55,
        crop_top_ratio: float = 0.25,
        crop_bottom_ratio: float = 0.10,
    ) -> None:
        self.lane_width_m = float(lane_width_m)
        self.camera_fov_deg = float(camera_fov_deg)
        self.cam_height_m = float(cam_height_m)
        self.crop_top_ratio = float(crop_top_ratio)
        self.crop_bottom_ratio = float(crop_bottom_ratio)

        self._est_offset_m: float = 0.0
        self._est_heading_err: float = 0.0
        self._est_curvature: float = 0.0
        self._vis_curvature: float = 0.0
        self._confidence: float = 0.0
        self._left_avail: bool = False
        self._right_avail: bool = False
        self._initialized: bool = False
        self._cached_shape: Optional[Tuple[int, int]] = None
        self._row_x_m: Optional[np.ndarray] = None
        self._row_m_per_px: Optional[np.ndarray] = None
        self._roi_rows: Optional[np.ndarray] = None
        self._calib_v_horizon: Optional[float] = None
        self._calib_slope: Optional[float] = None
        self._was_in_turn_box: bool = False
        self._turn_box_dir: str = ""
        self._turn_integrated_rad: float = 0.0
        self._turn_box_steps: int = 0
        self._unpainted_advance_m: float = 0.0
        self._post_turn_cooldown_m: float = 0.0
        self._nav_turn_streak: int = 0
        self._sparse_paint_streak: int = 0

    def reset(self) -> None:
        """Reset temporal corridor state at episode boundaries."""
        self._est_offset_m = 0.0
        self._est_heading_err = 0.0
        self._est_curvature = 0.0
        self._vis_curvature = 0.0
        self._confidence = 0.0
        self._left_avail = False
        self._right_avail = False
        self._initialized = False
        self._calib_v_horizon = None
        self._calib_slope = None
        self._was_in_turn_box = False
        self._turn_box_dir = ""
        self._turn_integrated_rad = 0.0
        self._turn_box_steps = 0
        self._unpainted_advance_m = 0.0
        self._post_turn_cooldown_m = 0.0
        self._nav_turn_streak = 0
        self._sparse_paint_streak = 0

    def _ensure_ipm_tables(self, h: int, w: int) -> None:
        """Precompute baseline pinhole-ground-plane IPM row lookup tables for `(h, w)` RGB."""
        if self._cached_shape == (h, w) and self._row_x_m is not None:
            return
        focal_px = (0.5 * float(w)) / math.tan(math.radians(0.5 * self.camera_fov_deg))
        v_horizon = float(h) * 0.36

        r_start = max(int(h * 0.40), int(v_horizon + 8))
        r_end = max(r_start + 8, int(h * 0.96))
        rows = np.arange(r_start, r_end, 2, dtype=np.int32)
        dy_px = np.maximum(rows.astype(np.float64) - v_horizon, 6.0)
        x_m = np.clip((self.cam_height_m * focal_px * 1.15) / dy_px, 3.2, 26.0)
        m_per_px = x_m / max(focal_px, 1.0)

        self._cached_shape = (h, w)
        self._roi_rows = rows
        self._row_x_m = x_m
        self._row_m_per_px = m_per_px

    def update(
        self,
        rgb: Optional[np.ndarray],
        speed_kmh: float = 0.0,
        imu_telemetry: Optional[Dict[str, Any]] = None,
        radar_targets: Optional[Sequence[Dict[str, Any]]] = None,
        predicted_trajectory: Optional[Sequence[Sequence[float]]] = None,
        nav_command: str = "go_straight",
        dt: float = 0.05,
    ) -> Dict[str, Any]:
        """Estimate ego corridor state strictly from RGB camera, IMU, radar & E2E trajectory."""
        dt_eff = max(1e-3, min(0.25, float(dt)))
        v_mps = max(0.0, float(speed_kmh) / 3.6)

        # 1. Kinematic bicycle + IMU dead-reckoning prediction step
        yaw_rate_ccw = 0.0
        if isinstance(imu_telemetry, dict) and "yaw_rate_rps" in imu_telemetry:
            # Note: CARLA IMU gyroscope.z > 0 is clockwise right; MetaDrive dyaw/dt > 0 is CCW left.
            # Damp yaw_rate contribution conservatively so sign conventions never destabilize prior.
            yaw_rate_ccw = float(np.clip(imu_telemetry.get("yaw_rate_rps", 0.0), -1.5, 1.5)) * 0.15

        if self._initialized:
            kappa_prev = (self._est_curvature if self._was_in_turn_box else self._vis_curvature) / 12.0
            pred_off = float(
                np.clip(
                    self._est_offset_m + v_mps * math.sin(self._est_heading_err) * dt_eff,
                    -2.5,
                    2.5,
                )
            )
            pred_herr = float(
                np.clip(
                    (self._est_heading_err + (yaw_rate_ccw - v_mps * kappa_prev) * dt_eff) * 0.94,
                    -0.42,
                    0.42,
                )
            )
            pred_curv = float(self._est_curvature * 0.92)
            pred_vis_curv = float(self._vis_curvature * 0.92)
        else:
            pred_off = 0.0
            pred_herr = 0.0
            pred_curv = 0.0
            pred_vis_curv = 0.0

        if rgb is None or not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.size == 0:
            self._est_offset_m = pred_off
            self._est_heading_err = pred_herr
            self._est_curvature = pred_curv
            self._confidence = 0.10
            return self._pack_output("mapless_imu_dead_reckoning")

        img = (
            rgb
            if rgb.dtype == np.uint8
            else (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        )
        h, w = img.shape[:2]
        if h < 32 or w < 64:
            return self._pack_output("mapless_imu_dead_reckoning")

        mean_luma = float(np.mean(img))
        if mean_luma < 12.0:
            # Camera blackout: rely on IMU + radar trail prior
            self._est_offset_m = pred_off
            self._est_heading_err = pred_herr
            self._est_curvature = pred_curv
            self._confidence = 0.15
            return self._pack_output("mapless_blackout_hold")

        self._ensure_ipm_tables(h, w)
        assert self._roi_rows is not None
        assert self._row_x_m is not None
        assert self._row_m_per_px is not None

        # 2. Extract painted white/yellow lane ridges via row-wise morphological top-hat + HSV yellow
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        k_w = max(7, int(round(w * 0.035)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        yellow_mask = (
            (hsv[:, :, 0] >= 12)
            & (hsv[:, :, 0] <= 38)
            & (hsv[:, :, 1] >= 55)
            & (hsv[:, :, 2] >= 70)
        )
        cand_rows = np.arange(int(h * 0.34), int(h * 0.96), 2, dtype=np.int32)
        pave_gray_med = float(
            np.median(gray[int(h * 0.74) : int(h * 0.93), int(w * 0.36) : int(w * 0.64)])
        )
        bright_floor = max(92.0, min(210.0, pave_gray_med + 14.0))
        white_thresh = max(18.0, float(np.percentile(tophat[cand_rows, :], 91.5)))
        lane_mask = ((tophat >= white_thresh) & (gray >= bright_floor)) | yellow_mask

        half_w_px = 0.5 * float(w)
        focal_eff = (0.5 * float(w)) / math.tan(math.radians(0.5 * 92.0))
        if self.camera_fov_deg >= 85.0:
            vh_default = float(h) * 0.50
            slope_default = 2.18 * (float(w) / 392.0)
            vh_lo, vh_hi = 0.38 * float(h), 0.58 * float(h)
        else:
            vh_default = float(h) * 0.36
            slope_default = 1.75 * (float(w) / 392.0)
            vh_lo, vh_hi = 0.12 * float(h), 0.58 * float(h)

        v_h_curr = (
            float(self._calib_v_horizon)
            if self._calib_v_horizon is not None
            else vh_default
        )
        slope_curr = (
            float(self._calib_slope)
            if self._calib_slope is not None
            else slope_default
        )

        # Bottom-up projective dual-boundary scanline tracker following TRUE visual clothoid prior
        dy_res_m = 0.0
        kappa_prior = pred_vis_curv / 12.0
        tan_prior = math.tan(pred_herr)

        def _y_prior_at(xm_q: float) -> float:
            dx_q = max(0.0, xm_q - 2.0)
            return float(pred_off + tan_prior * dx_q - 0.5 * kappa_prior * (dx_q ** 2))

        nav_c = str(nav_command or "go_straight").strip().lower()
        obs_rows: List[Tuple[float, float, bool]] = []  # (row, c_center_px, is_dual)
        dual_r: List[float] = []
        dual_w: List[float] = []
        left_hits = 0
        right_hits = 0
        left_dashed_gaps = 0
        right_dashed_gaps = 0

        for r_int in cand_rows[::-1]:
            r_f = float(r_int)
            if r_f <= v_h_curr + (10.0 if self.camera_fov_deg >= 85.0 else 8.0):
                break
            w_exp = max(18.0, slope_curr * (r_f - v_h_curr))
            mpp_exp = self.lane_width_m / w_exp
            xm_exp = float(np.clip((focal_eff * self.lane_width_m) / w_exp, 3.0, 25.0))
            c_track = float(
                np.clip(
                    half_w_px + (_y_prior_at(xm_exp) + dy_res_m) / max(mpp_exp, 1e-4),
                    0.06 * w,
                    0.94 * w,
                )
            )

            cols = np.flatnonzero(lane_mask[int(r_int), :])
            if cols.size == 0:
                left_dashed_gaps += 1
                right_dashed_gaps += 1
                continue

            # Reject pedestrian zebra crosswalks & transverse stop bars (dense white stripes inside lane center)
            int_c0 = max(0, int(round(c_track - 0.28 * w_exp)))
            int_c1 = min(w, int(round(c_track + 0.28 * w_exp)))
            if (int_c1 - int_c0) >= 8 and float(np.mean(lane_mask[int(r_int), int_c0:int_c1])) > 0.16:
                continue

            cols_f = cols.astype(np.float64)
            l_lo = c_track - 0.82 * w_exp
            l_hi = c_track - 0.22 * w_exp
            r_lo = c_track + 0.22 * w_exp
            r_hi = c_track + 0.82 * w_exp

            l_cands = cols_f[(cols_f >= l_lo) & (cols_f <= l_hi)]
            r_cands = cols_f[(cols_f >= r_lo) & (cols_f <= r_hi)]

            col_l: Optional[float] = None
            col_r: Optional[float] = None
            if l_cands.size > 0:
                col_l = float(np.max(l_cands)) - min(2.0, 0.5 * float(np.ptp(l_cands[-6:])))
                left_hits += 1
            else:
                left_dashed_gaps += 1

            if r_cands.size > 0:
                col_r = float(np.min(r_cands)) + min(2.0, 0.5 * float(np.ptp(r_cands[:6])))
                right_hits += 1
            else:
                right_dashed_gaps += 1

            c_mid_r: Optional[float] = None
            is_dual_hit = False
            if col_l is not None and col_r is not None and (0.62 * w_exp) <= (col_r - col_l) <= (1.45 * w_exp):
                c_mid_r = 0.5 * (col_l + col_r)
                is_dual_hit = True
            elif col_l is not None and abs(col_l - (c_track - 0.5 * w_exp)) <= 0.24 * w_exp:
                c_mid_r = col_l + 0.5 * w_exp
            elif col_r is not None and abs(col_r - (c_track + 0.5 * w_exp)) <= 0.24 * w_exp:
                c_mid_r = col_r - 0.5 * w_exp

            if c_mid_r is not None:
                y_obs_r = (c_mid_r - half_w_px) * mpp_exp
                innov_r = y_obs_r - (_y_prior_at(xm_exp) + dy_res_m)
                if abs(innov_r) <= 1.25 or not self._initialized or self._was_in_turn_box:
                    obs_rows.append((r_f, c_mid_r, is_dual_hit))
                    if is_dual_hit and col_l is not None and col_r is not None:
                        dual_r.append(r_f)
                        dual_w.append(col_r - col_l)
                    gain_r = 0.45 if is_dual_hit else 0.26
                    dy_res_m = float(np.clip(dy_res_m + gain_r * innov_r, -1.65, 1.65))

        # Online self-calibration of vanishing horizon v_horizon & perspective lane-width slope
        if len(dual_r) >= 4 and (max(dual_r) - min(dual_r)) >= 18.0:
            try:
                r_fit = np.asarray(dual_r, dtype=np.float64)
                w_fit = np.asarray(dual_w, dtype=np.float64)
                a_est, b_est = np.polyfit(r_fit, w_fit, 1)
                if 0.80 <= float(a_est) <= 3.80:
                    vh_est = -float(b_est) / float(a_est)
                    if vh_lo <= vh_est <= vh_hi:
                        if self._calib_v_horizon is None or self._calib_slope is None:
                            self._calib_v_horizon = vh_est
                            self._calib_slope = float(a_est)
                        else:
                            self._calib_v_horizon = 0.80 * self._calib_v_horizon + 0.20 * vh_est
                            self._calib_slope = 0.80 * self._calib_slope + 0.20 * float(a_est)
                        v_h_curr = float(self._calib_v_horizon)
                        slope_curr = float(self._calib_slope)
            except Exception:
                pass

        xs_center: List[float] = []
        ys_center: List[float] = []
        for r_f, c_mid_r, _is_dual in obs_rows:
            dy_cal = max(8.0, r_f - v_h_curr)
            w_cal_px = max(16.0, slope_curr * dy_cal)
            mpp_cal = self.lane_width_m / w_cal_px
            xm_cal = float(np.clip((focal_eff * self.lane_width_m) / w_cal_px, 3.0, 25.0))
            yc_cal = (c_mid_r - half_w_px) * mpp_cal
            xs_center.append(xm_cal)
            ys_center.append(yc_cal)

        n_rows_total = max(1, int(cand_rows.shape[0]))
        active_source = "mapless_lane_clothoid"

        # 3. Asphalt drivable-freespace segmentation fallback when lane markings are sparse
        pavement_patch = img[int(h * 0.74) : int(h * 0.93), int(w * 0.38) : int(w * 0.62)]
        pave_med = np.median(pavement_patch.reshape(-1, 3).astype(np.float32), axis=0)
        pave_mad = np.maximum(
            12.0,
            np.median(
                np.abs(pavement_patch.reshape(-1, 3).astype(np.float32) - pave_med),
                axis=0,
            )
            * 2.2,
        )

        # Check adjacent-lane drivable asphalt presence at mid-range (x ~ 8..14m)
        left_asphalt_scores: List[float] = []
        right_asphalt_scores: List[float] = []
        for r_m in range(int(max(v_h_curr + 20.0, h * 0.56)), int(h * 0.78), 4):
            w_m_px = max(20.0, slope_curr * (float(r_m) - v_h_curr))
            mpp_m = self.lane_width_m / w_m_px
            c_left_adj = int(round(half_w_px + (-self.lane_width_m + pred_off) / max(mpp_m, 1e-4)))
            c_right_adj = int(round(half_w_px + (self.lane_width_m + pred_off) / max(mpp_m, 1e-4)))
            if 6 <= c_left_adj < w - 6:
                patch_l = img[r_m, c_left_adj - 5 : c_left_adj + 6].astype(np.float32)
                dist_l = float(np.mean(np.abs(patch_l - pave_med) / pave_mad))
                left_asphalt_scores.append(dist_l)
            if 6 <= c_right_adj < w - 6:
                patch_r = img[r_m, c_right_adj - 5 : c_right_adj + 6].astype(np.float32)
                dist_r = float(np.mean(np.abs(patch_r - pave_med) / pave_mad))
                right_asphalt_scores.append(dist_r)

        self._left_avail = bool(
            left_asphalt_scores
            and float(np.mean(left_asphalt_scores)) < 1.55
            and left_dashed_gaps >= 2
        )
        self._right_avail = bool(
            right_asphalt_scores
            and float(np.mean(right_asphalt_scores)) < 1.55
            and right_dashed_gaps >= 2
        )

        if len(xs_center) < 6:
            # Unmarked intersection or faded markings: segment drivable asphalt corridor width
            active_source = "mapless_freespace_imu"
            for r_int in range(int(max(v_h_curr + 14.0, h * 0.48)), int(h * 0.94), 4):
                w_fs_px = max(18.0, slope_curr * (float(r_int) - v_h_curr))
                mpp = self.lane_width_m / w_fs_px
                xm = float(np.clip((focal_eff * self.lane_width_m) / w_fs_px, 3.0, 24.0))
                row_rgb = img[r_int, :].astype(np.float32)
                pave_dist = np.mean(np.abs(row_rgb - pave_med[None, :]) / pave_mad[None, :], axis=1)
                drivable = pave_dist < 1.75
                c_mid = int(np.clip(round(half_w_px + pred_off / max(mpp, 1e-4)), 4, w - 5))
                if not drivable[c_mid]:
                    continue
                non_drv_l = np.flatnonzero(~drivable[:c_mid])
                c_l = int(non_drv_l[-1] + 1) if non_drv_l.size > 0 else 0
                non_drv_r = np.flatnonzero(~drivable[c_mid + 1 :])
                c_r = int(c_mid + 1 + non_drv_r[0] - 1) if non_drv_r.size > 0 else (w - 1)
                yl_fs = (float(c_l) - half_w_px) * mpp
                yr_fs = (float(c_r) - half_w_px) * mpp
                span_m = yr_fs - yl_fs
                y_mid_fs = 0.5 * (yl_fs + yr_fs)
                if (
                    non_drv_l.size > 0
                    and non_drv_r.size > 0
                    and 2.8 <= span_m <= 4.6
                    and (not self._initialized or (self._post_turn_cooldown_m <= 0.0 and abs(y_mid_fs - pred_off) <= 0.38))
                ):
                    xs_center.append(xm)
                    ys_center.append(y_mid_fs)

        # Optional lead-vehicle radar trail cue inside unmarked intersections
        if len(xs_center) < 5 and radar_targets:
            for tgt in radar_targets:
                try:
                    tx = float(tgt.get("x_m", 0.0))
                    ty = float(tgt.get("y_m", 0.0))
                    if (
                        str(tgt.get("category", "vehicle")) == "vehicle"
                        and not bool(tgt.get("crossing_conflict", False))
                        and not bool(tgt.get("non_conflicting_cross", False))
                        and 4.0 <= tx <= 24.0
                        and abs(ty) <= 1.8
                    ):
                        xs_center.append(tx)
                        ys_center.append(ty)
                except (TypeError, ValueError):
                    continue

        nav_c = str(nav_command or "go_straight").strip().lower()
        has_imu_gyro = isinstance(imu_telemetry, dict) and "yaw_rate_rps" in imu_telemetry
        raw_yaw_rate = (
            abs(float(imu_telemetry.get("yaw_rate_rps", 0.0)))
            if has_imu_gyro
            else 0.0
        )
        if not has_imu_gyro and v_mps > 0.5 and abs(pred_curv) > 0.25:
            raw_yaw_rate = v_mps * (abs(pred_curv) / 14.0)

        has_close_conflict = any(
            bool(t.get("crossing_conflict", False)) and float(t.get("x_m", 60.0)) < 20.0
            for t in (radar_targets or [])
        )

        # Visual intersection-mouth check: verify drivable cross-street asphalt actually opens up
        # on the target side (x ~ 4..9m, rows 66%..92%) before turning, preventing premature turns
        # into grass/sidewalks when dashed approach lines have a gap 12m before the corner!
        mouth_r0, mouth_r1 = int(0.66 * float(h)), int(0.92 * float(h))
        if nav_c == "turn_left":
            side_patch = img[mouth_r0:mouth_r1, int(0.02 * w) : max(int(0.02 * w) + 4, int(0.14 * w))].astype(np.float32)
            turn_mouth_open = bool(
                side_patch.size > 0
                and float(np.mean(np.mean(np.abs(side_patch - pave_med[None, None, :]) / pave_mad[None, None, :], axis=2) < 1.85)) >= 0.50
            )
        elif nav_c == "turn_right":
            side_patch = img[mouth_r0:mouth_r1, int(0.86 * w) : max(int(0.86 * w) + 4, int(0.98 * w))].astype(np.float32)
            turn_mouth_open = bool(
                side_patch.size > 0
                and float(np.mean(np.mean(np.abs(side_patch - pave_med[None, None, :]) / pave_mad[None, None, :], axis=2) < 1.85)) >= 0.50
            )
        else:
            turn_mouth_open = False

        # VIO Gyro-Guided Intersection Turn State Machine:
        # Once Ego enters an unpainted T/4-way intersection box (len(obs_rows) < 6 AND cross-street mouth open)
        # with an active turn command, integrate IMU yaw rate until ~66-72 deg of heading rotation completes
        # before re-locking onto painted lane lines (prevents latching onto 45-deg cross-street stripes!).
        # VIO Gyro + Wheel-Odometry Intersection Turn State Machine:
        # Latches upon entering an unpainted T/4-way intersection mouth (len(obs_rows) < 6), tracks
        # forward meters driven into the box (`_unpainted_advance_m`) so Ego clears the near corner /
        # oncoming lanes before tightening the arc, and integrates IMU yaw rate to unwind at ~80 deg!
        if self._post_turn_cooldown_m > 0.0:
            self._post_turn_cooldown_m = max(0.0, self._post_turn_cooldown_m - v_mps * dt_eff)
        if nav_c in ("turn_left", "turn_right"):
            self._nav_turn_streak += 1
        else:
            self._nav_turn_streak = 0
        if len(obs_rows) < 5:
            self._sparse_paint_streak += 1
        else:
            self._sparse_paint_streak = 0

        if self._was_in_turn_box:
            if v_mps > 0.35:
                self._turn_box_steps += 1
            if v_mps > 0.25:
                self._turn_integrated_rad += raw_yaw_rate * dt_eff
                self._unpainted_advance_m += v_mps * dt_eff
            turn_dir_chk = (
                self._turn_box_dir
                if self._turn_box_dir in ("turn_left", "turn_right")
                else nav_c
            )
            req_rad = (
                1.45
                if (turn_dir_chk == "turn_right" and self.camera_fov_deg < 85.0)
                else 1.385
            )
            reacquired_lane = bool(self._turn_integrated_rad >= req_rad and len(obs_rows) >= 6)
            aborted_before_turn = bool(
                nav_c not in ("turn_left", "turn_right")
                and self._turn_integrated_rad < 0.18
                and len(obs_rows) >= 6
            )
            if (
                aborted_before_turn
                or reacquired_lane
                or self._turn_integrated_rad >= 1.53
                or self._turn_box_steps > int(14.0 / max(dt_eff, 0.02))
            ):
                self._was_in_turn_box = False
                self._turn_box_dir = ""
                self._turn_integrated_rad = 0.0
                self._turn_box_steps = 0
                self._unpainted_advance_m = 0.0
                self._post_turn_cooldown_m = 28.0
                pred_off = 0.0
                pred_herr = 0.0
                pred_curv = 0.0
                pred_vis_curv = 0.0
        elif (
            nav_c in ("turn_left", "turn_right")
            and self._nav_turn_streak >= 3
            and self._post_turn_cooldown_m <= 0.0
            and len(obs_rows) < 5
            and self._initialized
        ):
            self._was_in_turn_box = True
            self._turn_box_dir = nav_c
            self._turn_integrated_rad = raw_yaw_rate * dt_eff if v_mps > 0.25 else 0.0
            self._unpainted_advance_m = v_mps * dt_eff if v_mps > 0.25 else 0.0
            self._turn_box_steps = 1

        turn_dir = (
            self._turn_box_dir
            if (self._was_in_turn_box and self._turn_box_dir in ("turn_left", "turn_right"))
            else nav_c
        )

        # Stationary kinematic invariance: a stopped vehicle (< 0.8 km/h) cannot drift laterally
        # when cross-traffic occludes the camera!
        if self._initialized and v_mps < 0.22:
            if turn_dir == "turn_left" and (self._was_in_turn_box or self._post_turn_cooldown_m <= 0.0):
                self._est_curvature = max(self._est_curvature, 0.46)
            elif turn_dir == "turn_right" and (self._was_in_turn_box or self._post_turn_cooldown_m <= 0.0):
                self._est_curvature = min(self._est_curvature, -0.46)
            return self._pack_output("mapless_freespace_imu" if self._was_in_turn_box else active_source)

        meas_vis_curv = pred_vis_curv
        if len(xs_center) >= 5 and len(obs_rows) >= 5 and not self._was_in_turn_box:
            x_arr = np.asarray(xs_center, dtype=np.float64)
            y_arr = np.asarray(ys_center, dtype=np.float64)

            # 1) Full-horizon (3..25m) Huber quadratic fit for true visual road curvature
            _b0, _b1, b2 = _fit_huber_quadratic(x_arr - 5.5, y_arr, prior_c0=pred_off)
            raw_curv_vis = float(np.clip(-12.0 * b2, -1.25, 1.25))
            if self._post_turn_cooldown_m > 10.0:
                raw_curv_vis = float(np.clip(raw_curv_vis, -0.24, 0.24))
            meas_vis_curv = raw_curv_vis
            # Temporally smoothed TRUE visual curvature used to "unbend" near-field points
            curv_rect = (0.65 * pred_vis_curv + 0.35 * raw_curv_vis) if self._initialized else raw_curv_vis
            kappa_rect = curv_rect / 12.0

            # 2) Curvature-rectified near-field linear fit: y_rect(x) = y(x) + 0.5*kappa*(x - 2.0)^2 = y0 + psi*(x - 2.0)
            near_mask = x_arr <= 11.5
            if int(np.sum(near_mask)) >= 4:
                xn_rel = x_arr[near_mask] - 2.0
                yn_rect = y_arr[near_mask] + 0.5 * kappa_rect * (xn_rel ** 2)
                xn = x_arr[near_mask] - 4.0
                des_n = np.column_stack([np.ones_like(xn), xn])
                reg_n = np.diag([0.08, 0.55])
                rhs_n = np.array([0.08 * pred_off, 0.0], dtype=np.float64)
                try:
                    th_n = np.linalg.solve(des_n.T @ des_n + reg_n, des_n.T @ yn_rect + rhs_n)
                    a0, a1 = float(th_n[0]), float(th_n[1])
                except np.linalg.LinAlgError:
                    a0, a1 = float(np.median(yn_rect)), 0.0
                raw_off = a0 - 2.0 * a1
                slope_near = a1
            else:
                xr_rel = x_arr - 2.0
                yr_rect = y_arr + 0.5 * kappa_rect * (xr_rel ** 2)
                raw_off = float(np.median(yr_rect))
                slope_near = 0.0

            if self._initialized and self._turn_box_steps == 0:
                max_shift = v_mps * 0.48 * dt_eff + 0.08
                raw_off = pred_off + float(np.clip(raw_off - pred_off, -max_shift, max_shift))
            meas_off = float(np.clip(raw_off, -1.65, 1.65))
            meas_herr = float(np.clip(math.atan(slope_near), -0.42, 0.42))
            meas_curv = raw_curv_vis
            if self._post_turn_cooldown_m <= 0.0:
                if nav_c == "turn_left":
                    curv_floor = 0.72 if len(obs_rows) < 14 else 0.48
                    meas_curv = float(max(meas_curv, curv_floor))
                elif nav_c == "turn_right":
                    curv_ceil = -0.72 if len(obs_rows) < 14 else -0.48
                    meas_curv = float(min(meas_curv, curv_ceil))
            dual_bonus = 0.25 if (left_hits >= 4 and right_hits >= 4) else 0.0
            conf = float(
                np.clip(
                    (len(xs_center) / max(10.0, 0.55 * n_rows_total)) * 0.75 + dual_bonus,
                    0.18,
                    0.98,
                )
            )
        elif len(xs_center) >= 4 and nav_c not in ("turn_left", "turn_right") and not self._was_in_turn_box:
            x_arr = np.asarray(xs_center, dtype=np.float64)
            y_arr = np.asarray(ys_center, dtype=np.float64)
            du_axle = -3.0
            b0, b1, b2 = _fit_huber_quadratic(x_arr - 5.5, y_arr, prior_c0=pred_off)
            meas_off = float(np.clip(b0 + du_axle * b1 + 0.5 * (du_axle ** 2) * b2, -1.65, 1.65))
            meas_herr = float(np.clip(math.atan(b1 + du_axle * b2), -0.42, 0.42))
            meas_curv = float(np.clip(-12.0 * b2, -1.25, 1.25))
            meas_vis_curv = meas_curv
            conf = 0.45
        elif self._was_in_turn_box:
            # Inside unpainted T/4-way urban intersection: compensate near-hood optical blind zone
            # from self-calibrated vanishing horizon (narrow-FOV sees further ahead than wide-FOV)
            active_source = "mapless_freespace_imu"
            meas_vis_curv = 0.0
            blind_hood_m = float(np.clip((0.48 - (v_h_curr / float(h))) * 36.0, 0.0, 6.0))
            adv_l_target = 8.2 if self.camera_fov_deg >= 85.0 else (17.6 + blind_hood_m)
            adv_r_target = 1.8 if self.camera_fov_deg >= 85.0 else (7.9 + blind_hood_m)
            unwind = float(np.clip((1.57 - self._turn_integrated_rad) / 0.30, 0.12, 1.0))
            if turn_dir == "turn_left" and has_close_conflict and self._turn_integrated_rad < 0.25:
                meas_off = 0.0
                meas_herr = 0.0
                meas_curv = 0.72
            elif turn_dir == "turn_left":
                ramp_l = float(np.clip((self._unpainted_advance_m - adv_l_target) / 3.4, 0.0, 1.0)) * unwind
                meas_off = -0.58 * ramp_l
                meas_herr = -0.23 * ramp_l
                meas_curv = 0.80
            elif turn_dir == "turn_right":
                ramp_r = float(np.clip((self._unpainted_advance_m - adv_r_target) / 2.6, 0.0, 1.0)) * unwind
                meas_off = 0.62 * ramp_r
                meas_herr = 0.24 * ramp_r
                meas_curv = -0.86
            else:
                meas_off = pred_off * 0.88
                meas_herr = pred_herr * 0.90
                meas_curv = pred_curv * 0.85
            conf = 0.65
        else:
            # Brief dashed-line gap: hold straight prior!
            active_source = "mapless_freespace_imu"
            meas_off = pred_off * 0.88
            meas_herr = pred_herr * 0.88
            meas_vis_curv = pred_vis_curv * 0.85
            if nav_c == "turn_left" and self._post_turn_cooldown_m <= 0.0:
                meas_curv = max(pred_curv * 0.85, 0.55)
            elif nav_c == "turn_right" and self._post_turn_cooldown_m <= 0.0:
                meas_curv = min(pred_curv * 0.85, -0.55)
            else:
                meas_curv = pred_curv * 0.85
            conf = 0.30

        # Blend Laya 6-WP E2E trajectory curvature hint when distant markings fade
        if predicted_trajectory and len(predicted_trajectory) >= 4 and conf < 0.55:
            try:
                wp3_x = max(4.0, float(predicted_trajectory[3][0]))
                wp3_y = float(predicted_trajectory[3][1]) - meas_off
                traj_kappa = (2.0 * wp3_y) / (wp3_x ** 2 + wp3_y ** 2)
                meas_curv = 0.75 * meas_curv + 0.25 * float(np.clip(12.0 * traj_kappa, -0.65, 0.65))
            except (IndexError, TypeError, ValueError):
                pass

        alpha = float(np.clip(0.32 + 0.48 * conf, 0.25, 0.80)) if self._initialized else 1.0
        new_off = (1.0 - alpha) * pred_off + alpha * meas_off
        new_herr = float(np.clip((1.0 - alpha) * pred_herr + alpha * meas_herr, -0.42, 0.42))
        if not self._was_in_turn_box and abs(self._vis_curvature) < 0.28:
            has_r_pave = bool(right_asphalt_scores and float(np.mean(right_asphalt_scores)) < 1.65)
            has_l_pave = bool(left_asphalt_scores and float(np.mean(left_asphalt_scores)) < 1.65)
            if new_off < -0.48 * self.lane_width_m and new_herr < -0.06 and (self._right_avail or has_r_pave):
                new_off = min(1.45, new_off + self.lane_width_m)
            elif new_off > 0.48 * self.lane_width_m and new_herr > 0.06 and (self._left_avail or has_l_pave):
                new_off = max(-1.45, new_off - self.lane_width_m)
        self._est_offset_m = float(np.clip(new_off, -1.75, 1.75))
        self._est_heading_err = new_herr
        self._est_curvature = float(
            np.clip((1.0 - 0.65 * alpha) * pred_curv + (0.65 * alpha) * meas_curv, -1.25, 1.25)
        )
        self._vis_curvature = float(
            np.clip((1.0 - 0.65 * alpha) * pred_vis_curv + (0.65 * alpha) * meas_vis_curv, -1.25, 1.25)
        )
        self._confidence = round(conf, 3)
        self._initialized = True
        return self._pack_output(active_source)

    def _pack_output(self, source: str) -> Dict[str, Any]:
        return {
            "lane_offset_m": round(float(self._est_offset_m), 4),
            "heading_error": round(float(self._est_heading_err), 4),
            "road_curvature": round(float(self._est_curvature), 4),
            "left_lane_available": bool(self._left_avail),
            "right_lane_available": bool(self._right_avail),
            "corridor_confidence": round(float(self._confidence), 3),
            "localization_source": str(source),
        }

    def resolve_telemetry(
        self,
        localization_mode: str,
        hd_telemetry: Dict[str, Any],
        rgb: Optional[np.ndarray],
        imu_telemetry: Optional[Dict[str, Any]] = None,
        radar_targets: Optional[Sequence[Dict[str, Any]]] = None,
        predicted_trajectory: Optional[Sequence[Sequence[float]]] = None,
        hd_map_dropped: bool = False,
        dt: float = 0.05,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Select or blend HD-map vs Mapless visual/inertial corridor telemetry."""
        mode = str(localization_mode or "hd_map").strip().lower()
        if mode not in LOCALIZATION_MODES:
            mode = "hd_map"

        if mode == "hd_map" and not hd_map_dropped:
            return dict(hd_telemetry), {
                "localization_mode": "hd_map",
                "localization_source": "hd_map",
                "corridor_confidence": 1.0,
                "hd_map_dropped": False,
            }

        vis_est = self.update(
            rgb=rgb,
            speed_kmh=float(hd_telemetry.get("speed_kmh", 0.0)),
            imu_telemetry=imu_telemetry,
            radar_targets=radar_targets,
            predicted_trajectory=predicted_trajectory,
            nav_command=str(hd_telemetry.get("nav_command", "go_straight")),
            dt=dt,
        )

        if mode == "hybrid_fallback" and not hd_map_dropped:
            self._was_in_turn_box = False
            self._turn_integrated_rad = 0.0
            self._turn_box_steps = 0
            self._unpainted_advance_m = 0.0
            # Arm a 14m post-failover settling window so an interrupted HD-map lane change
            # finishes centering onto physical paint before intersection turn-box detection arms!
            self._post_turn_cooldown_m = max(self._post_turn_cooldown_m, 14.0)
            return dict(hd_telemetry), {
                "localization_mode": "hybrid_fallback",
                "localization_source": "hd_map_primary",
                "corridor_confidence": 1.0,
                "hd_map_dropped": False,
            }

        out_tel = dict(hd_telemetry)
        out_tel["lane_offset_m"] = vis_est["lane_offset_m"]
        out_tel["heading_error"] = vis_est["heading_error"]
        out_tel["road_curvature"] = vis_est["road_curvature"]
        if mode == "mapless_vision" or hd_map_dropped:
            out_tel["left_lane_available"] = vis_est["left_lane_available"] and not bool(
                out_tel.get("blind_spot_left", False)
            )
            out_tel["right_lane_available"] = vis_est["right_lane_available"] and not bool(
                out_tel.get("blind_spot_right", False)
            )
            if abs(float(vis_est["road_curvature"])) > 0.42 and str(
                out_tel.get("nav_command", "go_straight")
            ) == "go_straight":
                out_tel["nav_command"] = (
                    "turn_left" if float(vis_est["road_curvature"]) > 0.0 else "turn_right"
                )

        diag = {
            "localization_mode": mode,
            "localization_source": vis_est["localization_source"],
            "corridor_confidence": vis_est["corridor_confidence"],
            "hd_map_dropped": bool(hd_map_dropped),
            "mapless_estimate": vis_est,
        }
        return out_tel, diag


__all__ = [
    "LOCALIZATION_MODES",
    "MaplessVisionCorridorEstimator",
]
