"""NAVSIM PDMS Scorer, Isolated Decoder Open-Loop Evaluator & Counterfactual Obstacle Sensitivity (Track 1-B).

Implements:
  1. Official NAVSIM Predictive Driver Model Score (PDMS):
       PDMS = NC * DAC * (5 * EP + 5 * TTC + 2 * C) / 12
  2. Isolated neural decoder open-loop benchmark: `ADE_3s`, `FDE_3s`, `Lat_ADE_3s`,
     3.0s OBB collision rate `CR_traj`, pure-neural accuracies, and Expected Calibration Error (ECE).
  3. Counterfactual Obstacle Lateral Shift Sensitivity test: renders a synthetic vehicle at 20m
     shifted laterally by `Delta y_obs in {-1.0, 0.0, +1.0} m` under fixed neutral telemetry and
     quantifies whether the neural decoder actively deflects its predicted endpoint `y_hat_6`
     away from the obstacle.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.laya_pilot import AutonomousDrivingSystem
from src.sim.e2e_losses import compute_open_loop_trajectory_metrics
from src.utils.calibration import compute_binary_ece, compute_multiclass_ece


def compute_episode_pdms(
    no_collision: float,
    drivable_area_compliance: float,
    ego_progress: float,
    ttc_within_bound: float,
    comfort: float,
) -> Dict[str, float]:
    """Compute official NAVSIM PDMS = NC * DAC * (5*EP + 5*TTC + 2*C) / 12 in [0.0, 1.0]."""
    nc = float(np.clip(no_collision, 0.0, 1.0))
    dac = float(np.clip(drivable_area_compliance, 0.0, 1.0))
    ep = float(np.clip(ego_progress, 0.0, 1.0))
    ttc = float(np.clip(ttc_within_bound, 0.0, 1.0))
    c = float(np.clip(comfort, 0.0, 1.0))

    weighted = (5.0 * ep + 5.0 * ttc + 2.0 * c) / 12.0
    pdms = nc * dac * weighted
    return {
        "pdms": round(pdms, 4),
        "pdms_pct": round(pdms * 100.0, 2),
        "nc": round(nc, 4),
        "dac": round(dac, 4),
        "ep": round(ep, 4),
        "ttc": round(ttc, 4),
        "comfort": round(c, 4),
    }


def compute_pdms_from_episode_summary(
    summary: Dict[str, Any],
    crashed: bool = False,
    out_of_road: bool = False,
) -> Dict[str, float]:
    """Derive official NAVSIM PDMS subscores directly from `EpisodeSafetyComfortTracker.summarize()`."""
    infr = summary.get("infraction_counts") if isinstance(summary.get("infraction_counts"), dict) else summary
    veh_cols = int(infr.get("vehicle_collisions", summary.get("vehicle_collisions", 0)))
    ped_cols = int(infr.get("pedestrian_collisions", summary.get("pedestrian_collisions", 0)))
    oor_events = int(infr.get("off_road_events", summary.get("off_road_events", 0))) + (
        1 if out_of_road else 0
    )
    lane_inv = int(infr.get("lane_invasions", summary.get("lane_invasions", 0)))

    has_at_fault_collision = bool(ped_cols > 0 or veh_cols > 0 or (crashed and not out_of_road))
    nc = 0.0 if has_at_fault_collision else 1.0

    if oor_events > 0:
        dac = 0.0
    else:
        dac = max(0.0, 1.0 - 0.15 * float(lane_inv))

    ep = float(np.clip(float(summary.get("route_progress_pct", 0.0)) / 100.0, 0.0, 1.0))

    min_ttc = float(summary.get("min_ttc_s", 99.0))
    if min_ttc >= 1.0:
        ttc_score = 1.0
    else:
        ttc_score = float(np.clip((min_ttc - 0.10) / 0.90, 0.0, 1.0))

    comfort_pct = float(
        summary.get("iso_comfort_rate_pct", summary.get("comfort_rate_pct", 100.0))
    )
    comfort = float(np.clip(comfort_pct / 100.0, 0.0, 1.0))

    return compute_episode_pdms(
        no_collision=nc,
        drivable_area_compliance=dac,
        ego_progress=ep,
        ttc_within_bound=ttc_score,
        comfort=comfort,
    )


@torch.no_grad()
def evaluate_isolated_decoder(
    system: AutonomousDrivingSystem,
    val_loader: Any,
    device: torch.device,
    use_amp: bool = False,
    aux_use_prior: bool = False,
) -> Dict[str, float]:
    """Evaluate isolated decoder open-loop accuracy, CR_traj, ECE, temporal gate & latency."""
    system.eval()
    m_probs_l: List[torch.Tensor] = []
    s_probs_l: List[torch.Tensor] = []
    a_probs_l: List[torch.Tensor] = []
    m_tgt_l: List[torch.Tensor] = []
    s_tgt_l: List[torch.Tensor] = []
    a_tgt_l: List[torch.Tensor] = []

    ade_sum = 0.0
    fde_sum = 0.0
    lat_ade_sum = 0.0
    cr_sum = 0.0
    n_total = 0

    for batch in val_loader:
        bsz = len(batch["telemetries"])
        tels = batch["telemetries"]
        el_steps = batch["elapsed_steps"].to(device=device, dtype=torch.float32)

        if "patch_tokens" in batch:
            cur_p = batch["patch_tokens"].to(device=device, dtype=torch.float32)
            prev_p = batch["prev_patch_tokens"].to(device=device, dtype=torch.float32)
        else:
            cur_i = batch["images"].to(device=device, dtype=torch.float32)
            prev_i = batch["prev_images"].to(device=device, dtype=torch.float32)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                cur_p = system.dino(cur_i)
                prev_p = system.dino(prev_i)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            prev_v = system.projector(prev_p, use_temporal=False)
            out = system.forward_from_patch_tokens(
                patch_tokens=cur_p,
                telemetries=tels,
                elapsed_steps=el_steps,
                prev_visual_tokens=prev_v,
                aux_use_prior=aux_use_prior,
            )

        ol_m = compute_open_loop_trajectory_metrics(
            pred_traj=out["predicted_trajectory"],
            gt_traj=batch["future_trajectory"].to(device),
            obstacles=batch["obstacles"].to(device),
            obstacle_mask=batch["obstacle_mask"].to(device),
        )
        ade_sum += ol_m["ade_3s_m"] * bsz
        fde_sum += ol_m["fde_3s_m"] * bsz
        lat_ade_sum += ol_m["lat_ade_3s_m"] * bsz
        cr_sum += ol_m["cr_traj"] * bsz
        n_total += bsz

        m_probs_l.append(out["maneuver_probs"].float().cpu())
        s_probs_l.append(out["speed_probs"].float().cpu())
        a_probs_l.append(out["emergency_brake"].float().cpu())
        m_tgt_l.append(batch["maneuver_idx"].long().cpu())
        s_tgt_l.append(batch["speed_level"].long().cpu())
        a_tgt_l.append(batch["emergency_brake"].float().cpu())

    denom = max(1, n_total)
    m_probs = torch.cat(m_probs_l, dim=0)
    s_probs = torch.cat(s_probs_l, dim=0)
    a_probs = torch.cat(a_probs_l, dim=0)
    m_tgt = torch.cat(m_tgt_l, dim=0)
    s_tgt = torch.cat(s_tgt_l, dim=0)
    a_tgt = torch.cat(a_tgt_l, dim=0)

    man_ece = float(compute_multiclass_ece(m_probs, m_tgt, num_bins=15))
    spd_ece = float(compute_multiclass_ece(s_probs, s_tgt, num_bins=15))
    aeb_ece = float(compute_binary_ece(a_probs, a_tgt, num_bins=15))
    mean_ece = (man_ece + spd_ece + aeb_ece) / 3.0

    man_acc = float((m_probs.argmax(dim=-1) == m_tgt).float().mean().item())
    spd_acc = float((s_probs.argmax(dim=-1) == s_tgt).float().mean().item())
    aeb_acc = float(((a_probs >= 0.5).float() == a_tgt).float().mean().item())

    # Single-frame end-to-end latency probe (batch_size=1)
    dummy_img = torch.zeros((1, 3, 196, 392), dtype=torch.float32, device=device)
    dummy_tel = [{"speed_kmh": 35.0, "lane_offset_m": 0.0, "nav_command": "go_straight"}]
    for _ in range(2):
        _ = system(dummy_img, dummy_tel, aux_use_prior=aux_use_prior)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_runs = 10
    for _ in range(n_runs):
        _ = system(dummy_img, dummy_tel, aux_use_prior=aux_use_prior)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0 / n_runs
    system.reset_temporal_buffer()

    return {
        "samples": n_total,
        "decision_mode": system.laya.decision_mode,
        "trajectory_mode": system.laya.trajectory_mode,
        "ade_3s_m": round(ade_sum / denom, 4),
        "fde_3s_m": round(fde_sum / denom, 4),
        "lat_ade_3s_m": round(lat_ade_sum / denom, 4),
        "cr_traj_pct": round(100.0 * cr_sum / denom, 2),
        "maneuver_acc_pct": round(100.0 * man_acc, 2),
        "speed_acc_pct": round(100.0 * spd_acc, 2),
        "aeb_acc_pct": round(100.0 * aeb_acc, 2),
        "mean_ece": round(mean_ece, 4),
        "abs_temporal_gate": round(
            float(system.projector.temporal_gate.detach().abs().item()), 6
        ),
        "latency_ms": round(latency_ms, 2),
    }


def _render_counterfactual_vehicle(
    rgb: np.ndarray,
    dist_m: float = 20.0,
    lateral_left_m: float = 0.0,
) -> np.ndarray:
    """Overlay a perspective rear-view obstacle vehicle at `(dist_m, lateral_left_m)` (`+y = LEFT`)."""
    out = rgb.copy()
    h, w = out.shape[:2]
    scale = float(np.clip(20.0 / max(6.0, dist_m), 0.35, 2.0))
    box_w = int(round(58 * scale))
    box_h = int(round(44 * scale))

    # In camera pixels, +lateral_left_m shifts vehicle LEFT (smaller pixel column x)
    cx = int(round((w * 0.5) - lateral_left_m * 34.0 * scale))
    cy = int(round(h * 0.56 + 10.0 * scale))

    x1 = max(0, cx - box_w // 2)
    x2 = min(w - 1, cx + box_w // 2)
    y1 = max(0, cy - box_h)
    y2 = min(h - 1, cy)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return out

    # Shadow + vehicle body + windshield + red brake lights (strong visual signature for DINOv2)
    cv2.rectangle(out, (x1 + 2, y2 - 3), (x2 - 2, min(h - 1, y2 + 4)), (22, 22, 26), -1)
    cv2.rectangle(out, (x1, y1 + box_h // 4), (x2, y2), (185, 38, 38), -1)
    cv2.rectangle(
        out,
        (x1 + box_w // 8, y1),
        (x2 - box_w // 8, y1 + box_h // 2),
        (55, 70, 88),
        -1,
    )
    # Bright tail-lights
    lw = max(4, box_w // 6)
    lh = max(3, box_h // 7)
    ly = y1 + int(box_h * 0.52)
    cv2.rectangle(out, (x1 + 3, ly), (x1 + 3 + lw, ly + lh), (255, 45, 45), -1)
    cv2.rectangle(out, (x2 - 3 - lw, ly), (x2 - 3, ly + lh), (255, 45, 45), -1)
    cv2.rectangle(out, (x1, y1), (x2, y2), (30, 30, 35), 1)
    return out


@torch.no_grad()
def evaluate_counterfactual_obstacle_sensitivity(
    system: AutonomousDrivingSystem,
    base_frames: Sequence[np.ndarray],
    device: torch.device,
    dist_m: float = 20.0,
    shift_m: float = 1.0,
    use_amp: bool = False,
) -> Dict[str, Any]:
    """Measure lateral trajectory deflection when a 20m obstacle shifts `Delta y_obs in {-1, 0, +1} m`.

    Telemetry is held strictly fixed and neutral so 100% of trajectory shift originates from vision.
    """
    system.eval()
    neutral_tel = {
        "speed_kmh": 35.0,
        "lane_offset_m": 0.0,
        "heading_error": 0.0,
        "left_lane_available": True,
        "right_lane_available": True,
        "nav_command": "go_straight",
        "lead_distance_m": 60.0,
        "lead_speed_kmh": 45.0,
        "pedestrian_distance_m": 100.0,
        "road_curvature": 0.0,
        "traffic_light": "green",
        "hazard_source": "none",
    }

    abs_shifts: List[float] = []
    signed_avoidances: List[float] = []
    center_y6s: List[float] = []
    consistent_count = 0

    for rgb in base_frames:
        img_left = _render_counterfactual_vehicle(rgb, dist_m=dist_m, lateral_left_m=+shift_m)
        img_center = _render_counterfactual_vehicle(rgb, dist_m=dist_m, lateral_left_m=0.0)
        img_right = _render_counterfactual_vehicle(rgb, dist_m=dist_m, lateral_left_m=-shift_m)

        batch_np = np.stack([img_left, img_center, img_right], axis=0)
        batch_t = (
            torch.from_numpy(batch_np)
            .permute(0, 3, 1, 2)
            .to(device=device, dtype=torch.float32)
            / 255.0
        )
        system.reset_temporal_buffer()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = system(
                batch_t,
                [neutral_tel, neutral_tel, neutral_tel],
                aux_use_prior=False,
            )
        traj = out["predicted_trajectory"].float().cpu().numpy()  # (3, 6, 3)
        y6_obs_left = float(traj[0, -1, 1])    # obstacle at +1.0m (LEFT) -> ego should steer RIGHT (smaller y)
        y6_obs_center = float(traj[1, -1, 1])
        y6_obs_right = float(traj[2, -1, 1])   # obstacle at -1.0m (RIGHT) -> ego should steer LEFT (larger y)

        # Positive avoidance_diff means ego shifts LEFT when obstacle is RIGHT, and RIGHT when obstacle is LEFT
        avoidance_diff = y6_obs_right - y6_obs_left
        abs_shifts.append(abs(avoidance_diff))
        signed_avoidances.append(avoidance_diff)
        center_y6s.append(y6_obs_center)
        if avoidance_diff > 0.0:
            consistent_count += 1

    n = max(1, len(base_frames))
    return {
        "num_frames": len(base_frames),
        "obstacle_dist_m": float(dist_m),
        "obstacle_shift_m": float(shift_m),
        "mean_abs_y6_shift_m": round(float(np.mean(abs_shifts)) if abs_shifts else 0.0, 4),
        "mean_signed_avoidance_m": round(
            float(np.mean(signed_avoidances)) if signed_avoidances else 0.0, 4
        ),
        "avoidance_direction_consistency_pct": round(100.0 * consistent_count / n, 1),
        "mean_center_y6_m": round(float(np.mean(center_y6s)) if center_y6s else 0.0, 4),
    }


__all__ = [
    "compute_episode_pdms",
    "compute_pdms_from_episode_summary",
    "evaluate_counterfactual_obstacle_sensitivity",
    "evaluate_isolated_decoder",
]
