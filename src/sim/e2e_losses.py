"""Differentiable Obstacle-Aware End-to-End Trajectory & Pure-Neural Supervision Losses.

Implements the composite Track 1-A objective:
    L_E2E = w_traj * L_SmoothL1(tau_hat, tau_real)
          + w_coll * L_coll(tau_hat, O_t)
          + w_lane * L_lane(tau_hat)
          + w_jerk * L_jerk(tau_hat)
          + w_dec  * L_pure_neural(mlp_maneuver, mlp_speed, mlp_aeb)
          + w_aux  * L_aux(pure_vision_aux)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from src.sim.e2e_dataset import EGO_HALF_LENGTH_M, EGO_HALF_WIDTH_M

DEFAULT_E2E_LOSS_WEIGHTS: Dict[str, float] = {
    "traj": 1.5,
    "coll": 3.0,
    "lane": 1.2,
    "jerk": 0.15,
    "maneuver": 1.0,
    "speed_level": 1.0,
    "emergency_brake": 2.0,
    "aux_perception": 0.25,
}


def compute_collision_repulsion_loss(
    pred_traj: torch.Tensor,
    obstacles: torch.Tensor,
    obstacle_mask: torch.Tensor,
    step_sec: float = 0.5,
    long_margin_m: float = 1.40,
    lat_margin_m: float = 0.55,
    ped_radius_mult: float = 1.50,
) -> torch.Tensor:
    """Differentiable spatiotemporal elliptical obstacle repulsion loss over the 6 future waypoints.

    Args:
        pred_traj: `(B, 6, 3)` predicted waypoints `[x_fwd, y_left, v_kmh]`.
        obstacles: `(B, M, 7)` obstacle OBBs `[x, y, vx, vy, half_l, half_w, is_ped]`
            in pose-`t` ego frame (`+x` forward, `+y` LEFT) with absolute velocities.
        obstacle_mask: `(B, M)` float mask (`1.0` valid obstacle, `0.0` padding).
    """
    bsz, horizon, _ = pred_traj.shape
    device = pred_traj.device
    dtype = pred_traj.dtype

    if obstacles.numel() == 0 or float(obstacle_mask.sum().item()) <= 0.0:
        return torch.zeros((), device=device, dtype=dtype)

    t_k = (
        torch.arange(1, horizon + 1, device=device, dtype=dtype) * float(step_sec)
    ).view(1, horizon, 1)  # (1, 6, 1)

    ox = obstacles[:, :, 0].unsqueeze(1)  # (B, 1, M)
    oy = obstacles[:, :, 1].unsqueeze(1)
    ovx = obstacles[:, :, 2].unsqueeze(1)
    ovy = obstacles[:, :, 3].unsqueeze(1)
    half_l = obstacles[:, :, 4].unsqueeze(1)
    half_w = obstacles[:, :, 5].unsqueeze(1)
    is_ped = obstacles[:, :, 6].unsqueeze(1)

    # Constant-velocity future obstacle centers in pose-t ego frame: (B, 6, M)
    ox_t = ox + ovx * t_k
    oy_t = oy + ovy * t_k

    # Safe elliptical semi-axes (enlarged 1.5x for vulnerable pedestrians)
    ped_scale = 1.0 + (float(ped_radius_mult) - 1.0) * is_ped
    rx = (EGO_HALF_LENGTH_M + half_l + float(long_margin_m)) * ped_scale
    ry = (EGO_HALF_WIDTH_M + half_w + float(lat_margin_m)) * ped_scale

    px = pred_traj[:, :, 0].unsqueeze(-1)  # (B, 6, 1)
    py = pred_traj[:, :, 1].unsqueeze(-1)  # (B, 6, 1)

    dx_norm = (px - ox_t) / rx.clamp_min(0.5)
    dy_norm = (py - oy_t) / ry.clamp_min(0.4)
    d_norm = torch.sqrt(dx_norm * dx_norm + dy_norm * dy_norm + 1e-6)

    penetration = F.relu(1.0 - d_norm)
    valid = obstacle_mask.unsqueeze(1).to(dtype=dtype)  # (B, 1, M)
    # Sum over obstacles, average over horizon and batch
    per_sample = (penetration.pow(2) * valid).sum(dim=-1).mean(dim=-1)  # (B,)
    return per_sample.mean()


def compute_lane_boundary_loss(
    pred_traj: torch.Tensor,
    drivable_bounds: torch.Tensor,
) -> torch.Tensor:
    """Differentiable penalty when predicted lateral waypoints exit `[right_min_y, left_max_y]`."""
    y_pred = pred_traj[:, :, 1]  # (B, 6)
    right_min = drivable_bounds[:, 0].unsqueeze(1)  # (B, 1)
    left_max = drivable_bounds[:, 1].unsqueeze(1)  # (B, 1)
    over_left = F.relu(y_pred - left_max)
    over_right = F.relu(right_min - y_pred)
    return (over_left.pow(2) + over_right.pow(2)).mean()


def compute_trajectory_jerk_loss(pred_traj: torch.Tensor) -> torch.Tensor:
    """Minimum-jerk 3rd finite-difference smoothness penalty on predicted 6-WP polyline."""
    bsz = pred_traj.shape[0]
    origin_xy = torch.zeros((bsz, 1, 2), device=pred_traj.device, dtype=pred_traj.dtype)
    xy_seq = torch.cat([origin_xy, pred_traj[:, :, :2]], dim=1)  # (B, 7, 2)
    d1 = xy_seq[:, 1:, :] - xy_seq[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    d3 = d2[:, 1:, :] - d2[:, :-1, :]  # (B, 4, 2)
    lat_jerk = d3[:, :, 1].pow(2).mean()
    long_jerk = d3[:, :, 0].pow(2).mean()
    return lat_jerk + 0.10 * long_jerk


def compute_obstacle_aware_e2e_loss(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute composite Track 1-A obstacle-aware trajectory + pure-neural readout + aux loss."""
    w = dict(DEFAULT_E2E_LOSS_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items()})

    pred_traj = outputs["predicted_trajectory"].float()  # (B, 6, 3)
    device = pred_traj.device

    gt_traj = batch["future_trajectory"].to(device=device, dtype=torch.float32)
    obstacles = batch["obstacles"].to(device=device, dtype=torch.float32)
    obstacle_mask = batch["obstacle_mask"].to(device=device, dtype=torch.float32)
    drivable_bounds = batch["drivable_bounds"].to(device=device, dtype=torch.float32)

    # 1. Trajectory Smooth-L1 loss (extra lateral precision weighting)
    l_x = F.smooth_l1_loss(pred_traj[:, :, 0], gt_traj[:, :, 0], beta=1.0)
    l_y = F.smooth_l1_loss(pred_traj[:, :, 1], gt_traj[:, :, 1], beta=0.35)
    l_v = F.smooth_l1_loss(pred_traj[:, :, 2] / 10.0, gt_traj[:, :, 2] / 10.0, beta=0.5)
    loss_traj = 1.0 * l_x + 2.5 * l_y + 0.6 * l_v

    # 2. Spatiotemporal obstacle repulsion loss
    loss_coll = compute_collision_repulsion_loss(
        pred_traj=pred_traj,
        obstacles=obstacles,
        obstacle_mask=obstacle_mask,
    )

    # 3. Drivable corridor boundary loss
    loss_lane = compute_lane_boundary_loss(pred_traj, drivable_bounds)

    # 4. Minimum-jerk smoothness loss
    loss_jerk = compute_trajectory_jerk_loss(pred_traj)

    # 5. Pure-neural discrete decision readout supervision (zero mock_tel_readout)
    m_tgt = batch["maneuver_idx"].to(device=device, dtype=torch.long)
    s_tgt = batch["speed_level"].to(device=device, dtype=torch.long)
    a_tgt = batch["emergency_brake"].to(device=device, dtype=torch.float32)

    mlp_m = outputs["mlp_maneuver_logits"].float()
    mlp_s = outputs["mlp_speed_logits"].float()
    mlp_a = outputs["mlp_aeb_logits"].float().squeeze(-1)

    loss_maneuver = F.cross_entropy(mlp_m, m_tgt)
    loss_speed = F.cross_entropy(mlp_s, s_tgt)
    pos_w = torch.tensor(3.0, device=device, dtype=torch.float32)
    loss_aeb = F.binary_cross_entropy_with_logits(mlp_a, a_tgt, pos_weight=pos_w)

    # 6. Pure-vision auxiliary perception loss (when aux_use_prior=False)
    aux = outputs.get("aux_perception")
    if aux is not None:
        gt_lead = batch["lead_distance_m"].to(device=device, dtype=torch.float32)
        gt_ped = batch["pedestrian_distance_m"].to(device=device, dtype=torch.float32)
        gt_off = batch["lane_offset_m"].to(device=device, dtype=torch.float32)
        gt_tl = batch["traffic_light_idx"].to(device=device, dtype=torch.long)
        gt_hz = batch["hazard_idx"].to(device=device, dtype=torch.long)
        gt_bs = batch["blindspot_targets"].to(device=device, dtype=torch.float32)

        l_aux_lead = F.smooth_l1_loss(
            aux["pred_lead_distance_m"].float() / 10.0, gt_lead / 10.0
        )
        l_aux_ped = F.smooth_l1_loss(
            aux["pred_ped_distance_m"].float() / 10.0, gt_ped / 10.0
        )
        l_aux_off = F.smooth_l1_loss(aux["pred_lane_offset_m"].float(), gt_off)
        l_aux_tl = F.cross_entropy(aux["pred_traffic_light_logits"].float(), gt_tl)
        l_aux_hz = F.cross_entropy(aux["pred_hazard_logits"].float(), gt_hz)
        l_aux_bs = F.binary_cross_entropy_with_logits(
            aux["blind_spot_logits"].float(), gt_bs
        )
        loss_aux = (
            l_aux_lead
            + 0.5 * l_aux_ped
            + l_aux_off
            + 0.5 * l_aux_tl
            + 0.7 * l_aux_hz
            + 0.5 * l_aux_bs
        )
    else:
        loss_aux = torch.zeros((), device=device, dtype=torch.float32)

    total = (
        w["traj"] * loss_traj
        + w["coll"] * loss_coll
        + w["lane"] * loss_lane
        + w["jerk"] * loss_jerk
        + w["maneuver"] * loss_maneuver
        + w["speed_level"] * loss_speed
        + w["emergency_brake"] * loss_aeb
        + w["aux_perception"] * loss_aux
    )

    return {
        "total": total,
        "traj": loss_traj.detach(),
        "coll": loss_coll.detach(),
        "lane": loss_lane.detach(),
        "jerk": loss_jerk.detach(),
        "maneuver": loss_maneuver.detach(),
        "speed": loss_speed.detach(),
        "aeb": loss_aeb.detach(),
        "aux": loss_aux.detach(),
    }


@torch.no_grad()
def compute_open_loop_trajectory_metrics(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    obstacles: torch.Tensor,
    obstacle_mask: torch.Tensor,
    step_sec: float = 0.5,
) -> Dict[str, float]:
    """Compute open-loop ADE_3s, FDE_3s, lateral ADE, and 3.0s OBB collision rate `CR_traj`."""
    pred_xy = pred_traj[:, :, :2].float()
    gt_xy = gt_traj[:, :, :2].float()
    l2 = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # (B, 6)
    ade_3s = float(l2.mean().item())
    fde_3s = float(l2[:, -1].mean().item())
    lat_ade = float((pred_xy[:, :, 1] - gt_xy[:, :, 1]).abs().mean().item())

    # Physical 2D bounding-box overlap check over t = 0.5s .. 3.0s
    bsz, horizon, _ = pred_traj.shape
    device = pred_traj.device
    t_k = (
        torch.arange(1, horizon + 1, device=device, dtype=torch.float32) * float(step_sec)
    ).view(1, horizon, 1)

    ox_t = obstacles[:, :, 0].unsqueeze(1).float() + obstacles[:, :, 2].unsqueeze(1).float() * t_k
    oy_t = obstacles[:, :, 1].unsqueeze(1).float() + obstacles[:, :, 3].unsqueeze(1).float() * t_k
    half_l = obstacles[:, :, 4].unsqueeze(1).float()
    half_w = obstacles[:, :, 5].unsqueeze(1).float()

    dx = (pred_xy[:, :, 0].unsqueeze(-1) - ox_t).abs()
    dy = (pred_xy[:, :, 1].unsqueeze(-1) - oy_t).abs()
    coll_box = (dx < (EGO_HALF_LENGTH_M + half_l)) & (dy < (EGO_HALF_WIDTH_M + half_w))
    valid = obstacle_mask.unsqueeze(1).to(device=device) > 0.5
    sample_collided = (coll_box & valid).any(dim=-1).any(dim=-1)  # (B,)
    cr_traj = float(sample_collided.float().mean().item())

    return {
        "ade_3s_m": ade_3s,
        "fde_3s_m": fde_3s,
        "lat_ade_3s_m": lat_ade,
        "cr_traj": cr_traj,
    }


__all__ = [
    "DEFAULT_E2E_LOSS_WEIGHTS",
    "compute_collision_repulsion_loss",
    "compute_lane_boundary_loss",
    "compute_obstacle_aware_e2e_loss",
    "compute_open_loop_trajectory_metrics",
    "compute_trajectory_jerk_loss",
]
