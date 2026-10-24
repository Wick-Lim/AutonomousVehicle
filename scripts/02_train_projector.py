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


def evaluate(
    system: AutonomousDrivingSystem,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    system.eval()
    total_loss = 0.0
    correct_m = 0
    correct_s = 0
    correct_a = 0
    total_samples = 0

    w_m = float(loss_weights.get("maneuver", 1.0))
    w_s = float(loss_weights.get("speed_level", 1.0))
    w_a = float(loss_weights.get("emergency_brake", 2.0))

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            telemetries = batch["telemetries"]
            target_m = batch["maneuver_idx"].to(device)
            target_s = batch["speed_level"].to(device)
            target_a = batch["emergency_brake"].to(device)

            out = system(images, telemetries)
            loss_m = F.cross_entropy(out["maneuver_logits"], target_m)
            loss_s = F.cross_entropy(out["speed_logits"], target_s)
            loss_a = F.binary_cross_entropy_with_logits(
                out["aeb_logits"].squeeze(-1), target_a
            )
            loss = w_m * loss_m + w_s * loss_s + w_a * loss_a

            bsz = images.size(0)
            total_loss += float(loss.item()) * bsz
            correct_m += int((out["maneuver_idx"] == target_m).sum().item())
            correct_s += int((out["speed_level"] == target_s).sum().item())
            pred_a = (out["emergency_brake"] >= 0.5).long()
            correct_a += int((pred_a == target_a.long()).sum().item())
            total_samples += bsz

    denom = max(1, total_samples)
    return {
        "val_loss": total_loss / denom,
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

    trainable_params = [p for p in system.projector.parameters() if p.requires_grad]
    num_trainable = system.projector.count_trainable_parameters()
    total_frozen = sum(
        p.numel() for p in system.parameters() if not p.requires_grad
    )

    print(
        f"[02_train_projector] Device: {device} | Mock: {use_mock}\n"
        f"  Trainable SpatialProjector params : {num_trainable:,} (~{num_trainable/1e6:.2f}M)\n"
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

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        system.projector.train()
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

            optimizer.zero_grad()
            out = system(images, telemetries)

            loss_m = F.cross_entropy(out["maneuver_logits"], target_m)
            loss_s = F.cross_entropy(out["speed_logits"], target_s)
            loss_a = F.binary_cross_entropy_with_logits(
                out["aeb_logits"].squeeze(-1), target_a
            )
            loss = w_m * loss_m + w_s * loss_s + w_a * loss_a

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
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                ckpt_path,
            )

    print(f"\n[02_train_projector] Saved best SpatialProjector checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
