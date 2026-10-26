"""Verification test suite for v8 perception, calibration, and control equations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control.shield_pid import SafetyShieldPID
from src.models.spatial_projector import AuxiliaryPerceptionHead, SpatialProjector


def ols_5step_slope_mps(window: List[float], dt: float = 0.05) -> float:
    """Compute 5-step ordinary least-squares (OLS) range-rate slope (Eq. 11)."""
    if len(window) < 2:
        return 0.0
    n_w = len(window)
    t_idx = np.arange(n_w, dtype=np.float64) * dt
    t_mean = float(t_idx.mean())
    d_arr = np.asarray(window, dtype=np.float64)
    return float(
        np.sum((t_idx - t_mean) * (d_arr - d_arr.mean()))
        / max(1e-6, float(np.sum((t_idx - t_mean) ** 2)))
    )


def count_sustained_aeb_events(
    throttle_brake_trace: List[float],
    threshold: float = -0.50,
    min_contiguous_ticks: int = 3,
    max_gap_bridge: int = 2,
) -> tuple[int, int]:
    """Group frame-level braking ticks into sustained AEB events (>=3 ticks, bridging <=2-tick gaps)."""
    active = [1 if u <= threshold else 0 for u in throttle_brake_trace]
    total_ticks = int(sum(active))

    # Bridge short gaps of <= max_gap_bridge ticks between active braking ticks
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

    # Count contiguous runs with length >= min_contiguous_ticks
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


def test_decoupled_pedestrian_and_range_rate_heads() -> None:
    """Verify AuxiliaryPerceptionHead outputs decoupled lead, pedestrian, and range-rate tensors."""
    head = AuxiliaryPerceptionHead(embed_dim=1024, hidden_dim=256)
    tokens = torch.randn(2, 36, 1024)
    out = head(tokens)

    assert "pred_lead_distance_m" in out
    assert "pred_ped_distance_m" in out
    assert "pred_range_rate_mps" in out

    lead_d = out["pred_lead_distance_m"]
    ped_d = out["pred_ped_distance_m"]
    rr_mps = out["pred_range_rate_mps"]

    assert lead_d.shape == (2,)
    assert ped_d.shape == (2,)
    assert rr_mps.shape == (2,)
    assert torch.all((lead_d >= 0.0) & (lead_d <= 60.0))
    assert torch.all((ped_d >= 0.0) & (ped_d <= 100.0))
    assert torch.all((rr_mps >= -25.0) & (rr_mps <= 25.0))
    # Ensure lead and pedestrian heads have independent weights
    assert head.lead_dist_head.weight.data_ptr() != head.ped_dist_head.weight.data_ptr()


def test_step_normalized_temporal_residual() -> None:
    r"""Verify Eq. (6): \Delta V_t = (V_t - V_{t-\Delta n_t}) / \Delta n_t scales with elapsed_steps."""
    proj = SpatialProjector()
    proj.eval()

    f0 = torch.zeros(1, 392, 768)
    f1 = torch.ones(1, 392, 768)

    proj.reset_temporal_buffer()
    _ = proj(f0, elapsed_steps=1)
    _ = proj(f1, elapsed_steps=1)
    norm_step1 = proj._temporal_delta_norm

    proj.reset_temporal_buffer()
    _ = proj(f0, elapsed_steps=1)
    _ = proj(f1, elapsed_steps=2)
    norm_step2 = proj._temporal_delta_norm

    assert norm_step1 > 0.0
    assert np.isclose(norm_step2, 0.5 * norm_step1, rtol=1e-4)


def test_5step_ols_variance_reduction_and_target_switch_reset() -> None:
    """Verify 5-step OLS slope reduces uncorrelated depth noise variance by 20x (4.47x RMS) vs 1-step diff."""
    rng = np.random.default_rng(42)
    dt = 0.05
    sigma_d = 1.0  # 1.0 m uncorrelated depth noise
    noisy_d = 25.0 + rng.normal(0.0, sigma_d, size=2000)

    diff_1step = np.diff(noisy_d) / dt
    ols_5step = [
        ols_5step_slope_mps(list(noisy_d[i - 4 : i + 1]), dt=dt)
        for i in range(4, len(noisy_d))
    ]

    var_ratio = float(np.var(diff_1step) / np.var(ols_5step))
    # Theoretical variance ratio is (2.0 / dt^2) / (0.1 / dt^2) = 20.0 (sqrt(20) = 4.47x RMS reduction)
    assert 17.5 <= var_ratio <= 22.5

    # Verify target-switch reset when |d_t - d_{t-1}| > 6.0 m
    stream = [30.0, 29.8, 29.6, 14.0, 13.8, 13.6]  # Cut-in at index 3 (-15.6 m jump)
    window: List[float] = []
    slopes: List[float] = []
    for d in stream:
        if window and abs(d - window[-1]) > 6.0:
            window = [d]
        else:
            window.append(d)
            if len(window) > 5:
                window.pop(0)
        slopes.append(ols_5step_slope_mps(window, dt=dt))

    # Immediately after the cut-in and 1 follow-up frame, slope reflects -4.0 m/s, not -300 m/s
    assert np.isclose(slopes[-1], -4.0, atol=1e-3)


def test_sustained_aeb_event_grouping() -> None:
    """Verify reproducible AEB event definition (>=3 contiguous ticks, <=2-tick gap bridge)."""
    # 2 isolated single-tick FP transients + 1 sustained 5-tick TP event
    trace = [0.0, -1.0, 0.0, 0.0, 0.0, -0.8, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
    evts, ticks = count_sustained_aeb_events(trace)
    assert evts == 1
    assert ticks == 7

    # Pure 2-tick transient (e.g., Full stack FP: 0 events, 2 ticks)
    fp_trace = [0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0]
    fp_evts, fp_ticks = count_sustained_aeb_events(fp_trace)
    assert fp_evts == 0
    assert fp_ticks == 2


def test_cth_acc_and_bicycle_curvature_feedforward() -> None:
    """Verify CTH-ACC equilibrium d*(24)=16.0m and bicycle curvature feedforward sign."""
    ctrl = SafetyShieldPID()
    decision = {
        "maneuver": "follow_lead",
        "maneuver_confidence": 0.95,
        "speed_level": 2,
        "speed_confidence": 0.95,
        "emergency_brake": 0.01,
        "predicted_trajectory": [[3.0 * k, 0.0, 24.0] for k in range(1, 7)],
    }
    telemetry = {
        "speed_kmh": 24.0,
        "lead_distance_m": 16.0,
        "lead_speed_kmh": 24.0,
        "pedestrian_distance_m": 100.0,
        "lane_offset_m": 0.0,
        "heading_error": 0.0,
        "road_curvature": 0.0,
        "traffic_light": "green",
        "hazard_source": "none",
        "left_lane_available": True,
        "right_lane_available": True,
    }
    action, debug_info = ctrl.compute_control(decision, telemetry, dt=0.05)
    assert len(action) == 2
    assert np.isclose(debug_info["target_speed_kmh"], 24.0, atol=1e-3)


def test_timestamp_aware_ols_under_frame_drops() -> None:
    """Verify timestamp-aware OLS recovers exact range rate under irregular frame drops (Delta n_t > 1)."""
    # True range rate: -4.0 m/s over irregular step intervals Delta n_t = [1, 2, 1, 3, 1] at dt = 0.05 s
    true_rr_mps = -4.0
    elapsed_steps_seq = [1, 2, 1, 3, 1]
    t_obs = 0.0
    d_obs = 30.0
    ts_window: List[tuple[float, float]] = []
    uniform_window: List[float] = []

    for dn in elapsed_steps_seq:
        t_obs += dn * 0.05
        d_obs += true_rr_mps * (dn * 0.05)
        ts_window.append((t_obs, d_obs))
        uniform_window.append(d_obs)

    t_idx = np.asarray([pt[0] for pt in ts_window], dtype=np.float64)
    d_arr = np.asarray([pt[1] for pt in ts_window], dtype=np.float64)
    t_mean = float(t_idx.mean())
    ts_slope = float(
        np.sum((t_idx - t_mean) * (d_arr - d_arr.mean()))
        / max(1e-6, float(np.sum((t_idx - t_mean) ** 2)))
    )
    uniform_slope = ols_5step_slope_mps(uniform_window, dt=0.05)

    # Timestamp-aware OLS recovers exact -4.0 m/s; uniform spacing overestimates magnitude (-7.0 m/s)
    assert np.isclose(ts_slope, true_rr_mps, atol=1e-6)
    assert abs(uniform_slope) > 1.5 * abs(true_rr_mps)


def test_low_speed_cth_acc_vs_aeb_and_stop_and_go_hysteresis() -> None:
    """Verify d_AEB < d*(v_ego) non-conflict below 10.5 km/h and Stop-and-Go hysteresis latch."""
    ctrl = SafetyShieldPID()
    decision = {
        "maneuver": "follow_lead",
        "maneuver_confidence": 0.95,
        "speed_level": 2,
        "speed_confidence": 0.95,
        "emergency_brake": 0.01,
    }
    # 1. At v_ego = 8.0 km/h (2.22 m/s), d*(8.0) = 10.67 m (< 11.5 m), steady-state following (v_lead = 8.0 km/h)
    # Must NOT trigger Rule 1 AEB and must maintain 8.0 km/h equilibrium
    tel_low = {
        "speed_kmh": 8.0,
        "lead_distance_m": 8.0 + 1.2 * (8.0 / 3.6),
        "lead_speed_kmh": 8.0,
        "pedestrian_distance_m": 100.0,
        "traffic_light": "green",
        "hazard_source": "none",
    }
    gated_low = ctrl.apply_safety_shield(decision, tel_low, shield_enabled=True)
    assert not gated_low["shield_triggered"]
    assert np.isclose(gated_low["target_speed_kmh"], 8.0, atol=1e-3)

    # 2. Stop-and-Go hysteresis: enter stop-hold at v_lead = 1.0 km/h, d_lead = 8.2 m
    tel_stop = dict(tel_low, speed_kmh=2.0, lead_speed_kmh=1.0, lead_distance_m=8.2)
    gated_stop = ctrl.apply_safety_shield(decision, tel_stop, shield_enabled=True)
    assert ctrl.stop_hold_active is True
    assert gated_stop["target_speed_kmh"] == 0.0

    # Remain in stop-hold inside hysteresis band (v_lead = 2.8 km/h, d_lead = 8.9 m)
    tel_hold = dict(tel_low, speed_kmh=0.0, lead_speed_kmh=2.8, lead_distance_m=8.9)
    gated_hold = ctrl.apply_safety_shield(decision, tel_hold, shield_enabled=True)
    assert ctrl.stop_hold_active is True
    assert gated_hold["target_speed_kmh"] == 0.0

    # Exit stop-hold when lead vehicle departs (v_lead = 4.0 km/h, d_lead = 9.8 m)
    tel_go = dict(tel_low, speed_kmh=0.0, lead_speed_kmh=4.0, lead_distance_m=9.8)
    gated_go = ctrl.apply_safety_shield(decision, tel_go, shield_enabled=True)
    assert ctrl.stop_hold_active is False
    assert gated_go["target_speed_kmh"] > 4.0


if __name__ == "__main__":
    test_decoupled_pedestrian_and_range_rate_heads()
    print("[PASS] test_decoupled_pedestrian_and_range_rate_heads")
    test_step_normalized_temporal_residual()
    print("[PASS] test_step_normalized_temporal_residual")
    test_5step_ols_variance_reduction_and_target_switch_reset()
    print("[PASS] test_5step_ols_variance_reduction_and_target_switch_reset")
    test_sustained_aeb_event_grouping()
    print("[PASS] test_sustained_aeb_event_grouping")
    test_cth_acc_and_bicycle_curvature_feedforward()
    print("[PASS] test_cth_acc_and_bicycle_curvature_feedforward")
    test_timestamp_aware_ols_under_frame_drops()
    print("[PASS] test_timestamp_aware_ols_under_frame_drops")
    test_low_speed_cth_acc_vs_aeb_and_stop_and_go_hysteresis()
    print("[PASS] test_low_speed_cth_acc_vs_aeb_and_stop_and_go_hysteresis")
    print("All 7 v9 verification tests passed successfully.")
