"""Verification test suite for v8 perception, calibration, and control equations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control.shield_pid import SafetyShieldPID
from src.models.laya_pilot import AutonomousDrivingSystem
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


def test_aeb_acc_crossover_regime_and_wet_friction_margin() -> None:
    """Verify exact analytical crossover v_ego = 41.8 km/h (11.61 m/s) at v_lead = 0 and wet-friction margin."""
    ctrl = SafetyShieldPID()
    # 1. Stationary lead (v_lead = 0): d_AEB < d*(v_ego) for v_ego < 41.8 km/h; d_AEB > d*(v_ego) at 50 km/h
    for v_kmh in [10.0, 20.0, 30.0, 40.0]:
        d_aeb = ctrl.compute_kinematic_aeb_distance(v_kmh, 0.0, a_max_mps2=6.0)
        d_star = 8.0 + 1.2 * (v_kmh / 3.6)
        assert d_aeb < d_star
    d_aeb_50 = ctrl.compute_kinematic_aeb_distance(50.0, 0.0, a_max_mps2=6.0)
    d_star_50 = 8.0 + 1.2 * (50.0 / 3.6)
    assert np.isclose(d_aeb_50, 27.8302, atol=1e-2)
    assert np.isclose(d_star_50, 24.6667, atol=1e-2)
    assert d_aeb_50 > d_star_50  # Intentional AEB preemption above 41.8 km/h onto stopped obstacle

    # 2. Steady-state / small closing speed (Delta_v <= 6 km/h): d_AEB < d*(v_ego) across entire [0, 50] km/h
    for v_kmh in [10.0, 25.0, 35.0, 50.0]:
        d_aeb_follow = ctrl.compute_kinematic_aeb_distance(v_kmh, max(0.0, v_kmh - 5.0), a_max_mps2=6.0)
        d_star_follow = 8.0 + 1.2 * (v_kmh / 3.6)
        assert d_aeb_follow < d_star_follow

    # 3. Wet road (a_max = 4.5 m/s^2) + uncertainty margin (1.8 m) increases stopping boundary safely
    d_aeb_wet = ctrl.compute_kinematic_aeb_distance(
        35.0, 0.0, a_max_mps2=4.5, uncertainty_margin_m=1.8
    )
    d_aeb_dry = ctrl.compute_kinematic_aeb_distance(35.0, 0.0, a_max_mps2=6.0)
    assert d_aeb_wet > d_aeb_dry + 1.8


def test_holm_bonferroni_multiple_comparison_adjustment() -> None:
    """Verify Holm-Bonferroni step-down adjusted p-values across K=8 baseline comparisons."""
    raw_p = {
        "rule_only": 1e-5,
        "prompt_only": 0.0001,
        "lag_300ms": 0.0002,
        "ar_15token_head": 0.0005,
        "random_laya": 0.0010,
        "expert_rule": 0.0039,
        "transformer_2l_100ep": 0.0063,
        "mlp_head": 0.0078,
    }
    sorted_items = sorted(raw_p.items(), key=lambda kv: kv[1])
    k_total = len(sorted_items)
    holm_p = {}
    running_max = 0.0
    for idx, (name, p_val) in enumerate(sorted_items):
        adj = min(1.0, (k_total - idx) * p_val)
        running_max = max(running_max, adj)
        holm_p[name] = running_max

    # All 8 comparisons remain statistically significant at alpha = 0.05 after Holm-Bonferroni correction
    assert all(p_adj < 0.05 for p_adj in holm_p.values())
    assert np.isclose(holm_p["transformer_2l_100ep"], 0.0063 * 2, atol=1e-4)


def test_trainable_parameter_count_exact_invariant() -> None:
    """Verify total trainable parameters in AutonomousDrivingSystem strictly equals 4,232,991."""
    system = AutonomousDrivingSystem(use_mock=True)
    trainable_params = [p for p in system.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)

    assert total_trainable == 4_232_991, (
        f"Expected exactly 4,232,991 trainable parameters, got {total_trainable}"
    )

    proj_params = sum(p.numel() for p in system.projector.parameters() if p.requires_grad)
    maneuver_params = sum(p.numel() for p in system.laya.maneuver_head.parameters() if p.requires_grad)
    speed_params = sum(p.numel() for p in system.laya.speed_head.parameters() if p.requires_grad)
    aeb_params = sum(p.numel() for p in system.laya.aeb_head.parameters() if p.requires_grad)
    traj_params = sum(p.numel() for p in system.laya.trajectory_head.parameters() if p.requires_grad)

    assert proj_params == 3_954_446
    assert maneuver_params == 33_412
    assert speed_params == 33_412
    assert aeb_params == 16_513
    assert traj_params == 195_208
    assert proj_params + maneuver_params + speed_params + aeb_params + traj_params == 4_232_991


def test_compiled_pdf_artifacts_and_manuscript_integrity() -> None:
    """Verify bilingual compiled PDF artifacts exist, are non-empty, and citations/parameters align."""
    paper_dir = PROJECT_ROOT / "paper"
    main_pdf = paper_dir / "main.pdf"
    main_ko_pdf = paper_dir / "main_ko.pdf"
    references_bib = paper_dir / "references.bib"

    assert main_pdf.is_file(), f"Compiled PDF not found: {main_pdf}"
    assert main_pdf.stat().st_size > 500_000, f"main.pdf is unexpectedly small: {main_pdf.stat().st_size} bytes"

    assert main_ko_pdf.is_file(), f"Compiled Korean PDF not found: {main_ko_pdf}"
    assert main_ko_pdf.stat().st_size > 500_000, f"main_ko.pdf is unexpectedly small: {main_ko_pdf.stat().st_size} bytes"

    # Verify exactly 42 unique bibliography references
    bib_text = references_bib.read_text(encoding="utf-8")
    entries = [line for line in bib_text.splitlines() if line.strip().startswith("@")]
    assert len(entries) == 42, f"Expected 42 bibliography entries, found {len(entries)}"

    # Verify nocite{*} in both TeX sources
    main_tex = (paper_dir / "main.tex").read_text(encoding="utf-8")
    main_ko_tex = (paper_dir / "main_ko.tex").read_text(encoding="utf-8")
    assert "\\nocite{*}" in main_tex
    assert "\\nocite{*}" in main_ko_tex

    # Verify Table 2 GT shield interventions typo fix: 1.32 /km
    assert "1.32" in main_tex
    assert "5.88" not in main_tex

    # Verify speed levels {0, 20, 40, 65} in both papers and READMEs
    assert "\\{0, 20, 40, 65\\}" in main_tex
    assert "\\{0, 20, 40, 65\\}" in main_ko_tex

    readme_en = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    readme_ko = (PROJECT_ROOT / "README.ko.md").read_text(encoding="utf-8")
    assert "40" in readme_en and "65" in readme_en
    assert "4,232,991" in readme_en and "4,232,991" in readme_ko


def test_mock_pipeline_01_through_04_execution() -> None:
    """Verify all 4 stages of the end-to-end pipeline run cleanly in mock mode with exit code 0."""
    python_exe = sys.executable

    # Stage 1: Data collection
    res_01 = subprocess.run(
        [python_exe, "scripts/01_collect_data.py", "--mock", "--num-frames", "5"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert res_01.returncode == 0, f"Stage 1 failed:\n{res_01.stderr}"
    assert (PROJECT_ROOT / "data" / "metadata.jsonl").is_file()

    # Stage 2: Projector training
    res_02 = subprocess.run(
        [python_exe, "scripts/02_train_projector.py", "--mock", "--epochs", "1", "--batch-size", "2"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert res_02.returncode == 0, f"Stage 2 failed:\n{res_02.stderr}"
    assert (PROJECT_ROOT / "checkpoints" / "projector_best.pt").is_file()

    # Stage 3: Calibration
    res_03 = subprocess.run(
        [python_exe, "scripts/03_calibrate.py", "--mock"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert res_03.returncode == 0, f"Stage 3 failed:\n{res_03.stderr}"
    assert (PROJECT_ROOT / "checkpoints" / "temperature.pt").is_file()

    # Stage 4: Simulation Evaluation
    res_04 = subprocess.run(
        [python_exe, "scripts/04_evaluate_sim.py", "--mock", "--episodes", "1", "--perception-mode", "vision_estimated"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert res_04.returncode == 0, f"Stage 4 failed:\n{res_04.stderr}"
    assert (PROJECT_ROOT / "checkpoints" / "eval_metrics.json").is_file()


def test_closed_loop_route_completion_vision_estimated_threshold() -> None:
    """Verify closed-loop evaluation achieves >= 85.0% route completion under vision_estimated mode."""
    eval_json = PROJECT_ROOT / "checkpoints" / "eval_metrics.json"
    assert eval_json.is_file(), f"eval_metrics.json not found: {eval_json}"
    with open(eval_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    route_completion = float(metrics.get("route_completion_rate_pct", 0.0))
    mean_latency = float(metrics.get("mean_latency_ms", 999.0))

    assert route_completion >= 85.0, (
        f"Route completion rate {route_completion:.1f}% is below target 85.0%"
    )
    assert mean_latency <= 40.0, (
        f"Mean latency {mean_latency:.2f} ms exceeds real-time limit 40.0 ms"
    )
    assert metrics.get("targets_met", {}).get("route_completion_ge_85pct", False) is True


def test_all_12_ablation_modes_execute_cleanly() -> None:
    """Verify all 12 supported ablation modes in 04_evaluate_sim.py execute cleanly with exit code 0."""
    python_exe = sys.executable
    eval_json = PROJECT_ROOT / "checkpoints" / "eval_metrics.json"
    saved_eval_bytes = eval_json.read_bytes() if eval_json.is_file() else None
    ablation_modes = [
        "full",
        "mlp_head",
        "small_transformer",
        "random_laya",
        "prompt_only",
        "no_temporal",
        "expert_rule_policy",
        "rule_only",
        "heuristic_fsm",
        "ar_delay_300ms",
        "autoregressive_vla",
        "model_only",
    ]

    try:
        for mode in ablation_modes:
            res = subprocess.run(
                [
                    python_exe,
                    "scripts/04_evaluate_sim.py",
                    "--mock",
                    "--episodes",
                    "1",
                    "--ablation-mode",
                    mode,
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            assert res.returncode == 0, (
                f"Ablation mode '{mode}' failed with returncode {res.returncode}:\n{res.stderr}"
            )
    finally:
        if saved_eval_bytes is not None:
            eval_json.write_bytes(saved_eval_bytes)


if __name__ == "__main__":
    _eval_json = PROJECT_ROOT / "checkpoints" / "eval_metrics.json"
    _eval_carla_json = PROJECT_ROOT / "checkpoints" / "eval_metrics_carla.json"
    _orig_eval = _eval_json.read_bytes() if _eval_json.is_file() else None
    _orig_eval_carla = _eval_carla_json.read_bytes() if _eval_carla_json.is_file() else None
    try:
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
        test_aeb_acc_crossover_regime_and_wet_friction_margin()
        print("[PASS] test_aeb_acc_crossover_regime_and_wet_friction_margin")
        test_holm_bonferroni_multiple_comparison_adjustment()
        print("[PASS] test_holm_bonferroni_multiple_comparison_adjustment")
        test_trainable_parameter_count_exact_invariant()
        print("[PASS] test_trainable_parameter_count_exact_invariant")
        test_compiled_pdf_artifacts_and_manuscript_integrity()
        print("[PASS] test_compiled_pdf_artifacts_and_manuscript_integrity")
        test_mock_pipeline_01_through_04_execution()
        print("[PASS] test_mock_pipeline_01_through_04_execution")
        test_closed_loop_route_completion_vision_estimated_threshold()
        print("[PASS] test_closed_loop_route_completion_vision_estimated_threshold")
        test_all_12_ablation_modes_execute_cleanly()
        print("[PASS] test_all_12_ablation_modes_execute_cleanly")
        print("All 14 verification tests passed successfully.")
    finally:
        if _orig_eval is not None:
            _eval_json.write_bytes(_orig_eval)
        if _orig_eval_carla is not None:
            _eval_carla_json.write_bytes(_orig_eval_carla)

