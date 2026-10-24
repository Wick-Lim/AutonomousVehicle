"""Step 3: Stage 2 Temperature Scaling Calibration on Validation Set (Target ECE <= 0.05)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.laya_pilot import AutonomousDrivingSystem
from src.utils.calibration import TemperatureScaler
from src.utils.dataset import build_dataloaders


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Laya decision probabilities via Temperature Scaling (ECE minimization)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--projector-ckpt", type=str, default=None)
    parser.add_argument("--output-ckpt", type=str, default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_mock = bool(args.mock or cfg.get("project", {}).get("use_mock", False))
    device = resolve_device(cfg.get("project", {}).get("device", "auto"))

    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    cal_cfg = cfg.get("calibration", {})

    data_dir = Path(
        args.data_dir
        if args.data_dir is not None
        else PROJECT_ROOT / data_cfg.get("data_dir", "data")
    )
    proj_ckpt_path = Path(
        args.projector_ckpt
        if args.projector_ckpt is not None
        else PROJECT_ROOT / train_cfg.get("projector_ckpt", "checkpoints/projector_best.pt")
    )
    temp_ckpt_path = Path(
        args.output_ckpt
        if args.output_ckpt is not None
        else PROJECT_ROOT / train_cfg.get("temperature_ckpt", "checkpoints/temperature.pt")
    )
    temp_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    _, val_loader = build_dataloaders(
        data_dir=data_dir,
        train_split=float(data_cfg.get("train_split", 0.85)),
        batch_size=int(train_cfg.get("batch_size", 32)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        oversample_cfg=data_cfg.get("oversampling", {}),
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )

    system = AutonomousDrivingSystem(config=cfg, use_mock=use_mock).to(device)
    if proj_ckpt_path.exists():
        ckpt = torch.load(proj_ckpt_path, map_location=device, weights_only=False)
        system.projector.load_state_dict(ckpt["projector_state_dict"])
        print(f"[03_calibrate] Loaded trained SpatialProjector from: {proj_ckpt_path}")
    else:
        print(
            f"[03_calibrate] WARNING: Projector checkpoint not found at {proj_ckpt_path}; calibrating current weights."
        )

    system.eval()
    m_logits_list: List[torch.Tensor] = []
    s_logits_list: List[torch.Tensor] = []
    a_logits_list: List[torch.Tensor] = []
    m_targets_list: List[torch.Tensor] = []
    s_targets_list: List[torch.Tensor] = []
    a_targets_list: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            telemetries = batch["telemetries"]
            out = system(images, telemetries)

            m_logits_list.append(out["maneuver_logits"])
            s_logits_list.append(out["speed_logits"])
            a_logits_list.append(out["aeb_logits"])
            m_targets_list.append(batch["maneuver_idx"].to(device))
            s_targets_list.append(batch["speed_level"].to(device))
            a_targets_list.append(batch["emergency_brake"].to(device))

    m_logits = torch.cat(m_logits_list, dim=0)
    s_logits = torch.cat(s_logits_list, dim=0)
    a_logits = torch.cat(a_logits_list, dim=0)
    m_targets = torch.cat(m_targets_list, dim=0)
    s_targets = torch.cat(s_targets_list, dim=0)
    a_targets = torch.cat(a_targets_list, dim=0)

    scaler = TemperatureScaler().to(device)
    num_bins = int(cal_cfg.get("num_bins", 15))
    before_metrics, after_metrics = scaler.fit(
        maneuver_logits=m_logits,
        maneuver_targets=m_targets,
        speed_logits=s_logits,
        speed_targets=s_targets,
        aeb_logits=a_logits,
        aeb_targets=a_targets,
        num_bins=num_bins,
    )

    temps = scaler.get_temperatures()
    system.laya.set_temperatures(
        temps["t_maneuver"],
        temps["t_speed"],
        temps["t_aeb"],
        temps.get("b_aeb", 0.0),
    )

    torch.save(
        {
            "temperatures": temps,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
        },
        temp_ckpt_path,
    )

    print("\n" + "=" * 68)
    print("Stage 2 Temperature Calibration Report")
    print("-" * 68)
    print(
        f"  Before Calibration | Maneuver ECE: {before_metrics['maneuver_ece']:.4f} | "
        f"Speed ECE: {before_metrics['speed_ece']:.4f} | "
        f"AEB ECE: {before_metrics['aeb_ece']:.4f} | "
        f"Mean ECE: {before_metrics['mean_ece']:.4f}"
    )
    print(
        f"  After Calibration  | Maneuver ECE: {after_metrics['maneuver_ece']:.4f} | "
        f"Speed ECE: {after_metrics['speed_ece']:.4f} | "
        f"AEB ECE: {after_metrics['aeb_ece']:.4f} | "
        f"Mean ECE: {after_metrics['mean_ece']:.4f}"
    )
    print(
        f"  Optimal Temperatures: T_maneuver={temps['t_maneuver']:.3f}, "
        f"T_speed={temps['t_speed']:.3f}, T_aeb={temps['t_aeb']:.3f}"
    )
    print(f"  Saved temperature checkpoint to: {temp_ckpt_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
