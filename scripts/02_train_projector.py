"""Step 2: Stage 1 Alignment Training of the 2D Spatial Projector (~2M Params)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.laya_pilot import AutonomousDrivingSystem
from src.utils.dataset import build_dataloaders


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def compute_aux_and_traj_losses(
    out: Dict[str, Any],
    telemetries: list[Dict[str, Any]],
    target_traj: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Smooth-L1 6-waypoint trajectory loss L_traj and auxiliary perception loss L_aux."""
    pred_traj = out["predicted_trajectory"]
    loss_traj = F.smooth_l1_loss(pred_traj, target_traj.to(device))

    aux = out.get("aux_perception", {})
    if not aux:
        return loss_traj, torch.tensor(0.0, device=device)

    tl_map = {"green": 0, "yellow": 1, "red": 2}
    haz_map = {"none": 0, "vehicle": 1, "pedestrian": 2, "red_light": 3}
    gt_lead = torch.tensor(
        [float(t.get("lead_distance_m", 60.0)) for t in telemetries],
        dtype=torch.float32,
        device=device,
    )
    gt_ped = torch.tensor(
        [float(t.get("pedestrian_distance_m", 100.0)) for t in telemetries],
        dtype=torch.float32,
        device=device,
    )
    gt_rr_mps = torch.tensor(
        [
            (float(t.get("lead_speed_kmh", 35.0)) - float(t.get("speed_kmh", 35.0))) / 3.6
            for t in telemetries
        ],
        dtype=torch.float32,
        device=device,
    )
    gt_off = torch.tensor(
        [float(t.get("lane_offset_m", 0.0)) for t in telemetries],
        dtype=torch.float32,
        device=device,
    )
    gt_tl = torch.tensor(
        [tl_map.get(str(t.get("traffic_light", "green")).lower(), 0) for t in telemetries],
        dtype=torch.long,
        device=device,
    )
    gt_haz = torch.tensor(
        [haz_map.get(str(t.get("hazard_source", "none")).lower(), 0) for t in telemetries],
        dtype=torch.long,
        device=device,
    )
    gt_bs = torch.tensor(
        [
            [float(bool(t.get("blind_spot_left", False))), float(bool(t.get("blind_spot_right", False)))]
            for t in telemetries
        ],
        dtype=torch.float32,
        device=device,
    )

    pred_ped = aux.get("pred_ped_distance_m", aux.get("pedestrian_distance_m"))
    loss_ped = (
        0.05 * F.smooth_l1_loss(pred_ped, gt_ped)
        if pred_ped is not None
        else torch.tensor(0.0, device=device)
    )
    pred_rr = aux.get("pred_range_rate_mps", aux.get("range_rate_mps"))
    loss_rr = (
        0.10 * F.smooth_l1_loss(pred_rr, gt_rr_mps)
        if pred_rr is not None
        else torch.tensor(0.0, device=device)
    )

    pred_lead = aux.get("pred_lead_distance_m", aux.get("lead_distance_m"))
    pred_off = aux.get("pred_lane_offset_m", aux.get("lane_offset_m"))
    tl_logits = aux.get("traffic_light_logits", aux.get("pred_traffic_light_logits"))
    haz_logits = aux.get("hazard_logits", aux.get("pred_hazard_logits"))

    if "blind_spot_logits" in aux:
        loss_bs = 0.5 * F.binary_cross_entropy_with_logits(aux["blind_spot_logits"], gt_bs)
    elif "pred_blindspot_probs" in aux:
        loss_bs = 0.5 * F.binary_cross_entropy(torch.clamp(aux["pred_blindspot_probs"], 1e-6, 1.0 - 1e-6), gt_bs)
    elif "blind_spot_probs" in aux:
        loss_bs = 0.5 * F.binary_cross_entropy(torch.clamp(aux["blind_spot_probs"], 1e-6, 1.0 - 1e-6), gt_bs)
    else:
        loss_bs = torch.tensor(0.0, device=device)

    loss_aux = (
        0.05 * F.smooth_l1_loss(pred_lead, gt_lead)
        + loss_ped
        + loss_rr
        + F.smooth_l1_loss(pred_off, gt_off)
        + 0.5 * F.cross_entropy(tl_logits, gt_tl)
        + 0.5 * F.cross_entropy(haz_logits, gt_haz)
        + loss_bs
    )
    return loss_traj, loss_aux


def evaluate(
    system: AutonomousDrivingSystem,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    system.eval()
    total_loss = 0.0
    total_traj_mae = 0.0
    correct_m = 0
    correct_s = 0
    correct_a = 0
    total_samples = 0

    w_m = float(loss_weights.get("maneuver", 1.0))
    w_s = float(loss_weights.get("speed_level", 1.0))
    w_a = float(loss_weights.get("emergency_brake", 2.0))
    w_traj = float(loss_weights.get("trajectory", 0.5))
    w_aux = float(loss_weights.get("aux_perception", 0.2))

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            telemetries = batch["telemetries"]
            target_m = batch["maneuver_idx"].to(device)
            target_s = batch["speed_level"].to(device)
            target_a = batch["emergency_brake"].to(device)
            target_traj = batch["target_trajectory"].to(device)

            out = system(images, telemetries)
            loss_m = F.cross_entropy(out["maneuver_logits"], target_m)
            loss_s = F.cross_entropy(out["speed_logits"], target_s)
            loss_a = F.binary_cross_entropy_with_logits(
                out["aeb_logits"].squeeze(-1), target_a
            )
            loss_traj, loss_aux = compute_aux_and_traj_losses(
                out, telemetries, target_traj, device
            )
            loss = (
                w_m * loss_m
                + w_s * loss_s
                + w_a * loss_a
                + w_traj * loss_traj
                + w_aux * loss_aux
            )

            bsz = images.size(0)
            total_loss += float(loss.item()) * bsz
            total_traj_mae += float((out["predicted_trajectory"] - target_traj).abs().mean().item()) * bsz
            correct_m += int((out["maneuver_idx"] == target_m).sum().item())
            correct_s += int((out["speed_level"] == target_s).sum().item())
            pred_a = (out["emergency_brake"] >= 0.5).long()
            correct_a += int((pred_a == target_a.long()).sum().item())
            total_samples += bsz

    denom = max(1, total_samples)
    return {
        "val_loss": total_loss / denom,
        "traj_mae": total_traj_mae / denom,
        "maneuver_acc": correct_m / denom,
        "speed_acc": correct_s / denom,
        "aeb_acc": correct_a / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the 2D Spatial Projector while keeping DINOv2 and Laya frozen."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use lightweight mock DINOv2 & Laya backbones for fast local verification.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_mock = bool(args.mock or cfg.get("project", {}).get("use_mock", False))
    device = resolve_device(cfg.get("project", {}).get("device", "auto"))

    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 15))
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else train_cfg.get("batch_size", 32)
    )
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    loss_weights = train_cfg.get(
        "loss_weights", {"maneuver": 1.0, "speed_level": 1.0, "emergency_brake": 2.0}
    )

    data_dir = Path(
        args.data_dir
        if args.data_dir is not None
        else PROJECT_ROOT / data_cfg.get("data_dir", "data")
    )
    ckpt_path = PROJECT_ROOT / train_cfg.get(
        "projector_ckpt", "checkpoints/projector_best.pt"
    )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(
        data_dir=data_dir,
        train_split=float(data_cfg.get("train_split", 0.85)),
        batch_size=batch_size,
        num_workers=int(train_cfg.get("num_workers", 0)),
        oversample_cfg=data_cfg.get("oversampling", {}),
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )

    system = AutonomousDrivingSystem(config=cfg, use_mock=use_mock).to(device)
    system.dino.freeze()
    system.laya.freeze_brain()

    # Unfreeze linear readout heads & trajectory head on LayaPilot while keeping 421M transformer layers frozen
    for head_mod in (
        system.laya.maneuver_head,
        system.laya.speed_head,
        system.laya.aeb_head,
        system.laya.trajectory_head,
    ):
        for p in head_mod.parameters():
            p.requires_grad = True

    trainable_params = [
        p
        for p in list(system.projector.parameters())
        + list(system.laya.maneuver_head.parameters())
        + list(system.laya.speed_head.parameters())
        + list(system.laya.aeb_head.parameters())
        + list(system.laya.trajectory_head.parameters())
        if p.requires_grad
    ]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_core = system.projector.count_trainable_parameters()
    total_frozen = sum(
        p.numel() for p in system.parameters() if not p.requires_grad
    )

    print(
        f"[02_train_projector] Device: {device} | Mock: {use_mock}\n"
        f"  Trainable Core Spatial MLP params : {num_core:,} (~{num_core/1e6:.2f}M)\n"
        f"  Trainable Total Adapter+Heads     : {num_trainable:,} (~{num_trainable/1e6:.2f}M)\n"
        f"  Frozen DINOv2 + Laya params       : {total_frozen:,}"
    )

    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )

    w_m = float(loss_weights.get("maneuver", 1.0))
    w_s = float(loss_weights.get("speed_level", 1.0))
    w_a = float(loss_weights.get("emergency_brake", 2.0))
    w_traj = float(loss_weights.get("trajectory", 0.5))
    w_aux = float(loss_weights.get("aux_perception", 0.2))

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        system.projector.train()
        system.laya.trajectory_head.train()
        running_loss = 0.0
        n_seen = 0

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d}", leave=False
        ):
            images = batch["images"].to(device)
            telemetries = batch["telemetries"]
            target_m = batch["maneuver_idx"].to(device)
            target_s = batch["speed_level"].to(device)
            target_a = batch["emergency_brake"].to(device)
            target_traj = batch["target_trajectory"].to(device)

            optimizer.zero_grad()
            out = system(images, telemetries)

            loss_m = F.cross_entropy(out["maneuver_logits"], target_m)
            loss_s = F.cross_entropy(out["speed_logits"], target_s)
            loss_a = F.binary_cross_entropy_with_logits(
                out["aeb_logits"].squeeze(-1), target_a
            )
            loss_traj, loss_aux = compute_aux_and_traj_losses(
                out, telemetries, target_traj, device
            )
            loss = (
                w_m * loss_m
                + w_s * loss_s
                + w_a * loss_a
                + w_traj * loss_traj
                + w_aux * loss_aux
            )

            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            bsz = images.size(0)
            running_loss += float(loss.item()) * bsz
            n_seen += bsz

        scheduler.step()
        train_loss = running_loss / max(1, n_seen)
        val_metrics = evaluate(system, val_loader, device, loss_weights)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['val_loss']:.4f} | "
            f"traj_mae={val_metrics['traj_mae']:.3f}m | "
            f"maneuver_acc={val_metrics['maneuver_acc']*100:.1f}% | "
            f"speed_acc={val_metrics['speed_acc']*100:.1f}% | "
            f"aeb_acc={val_metrics['aeb_acc']*100:.1f}%"
        )

        if val_metrics["val_loss"] <= best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "projector_state_dict": system.projector.state_dict(),
                    "trajectory_head_state_dict": system.laya.trajectory_head.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                ckpt_path,
            )

    print(f"\n[02_train_projector] Saved best SpatialProjector checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
