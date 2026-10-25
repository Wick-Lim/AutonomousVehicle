"""PyTorch Dataset, Collator, and Class-Balancing Oversampler for Expert Driving Data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.env.expert_labeler import MANEUVER_TO_IDX


class DrivingDataset(Dataset):
    """Dataset of (196x392 RGB image, structured Telemetry JSON, Expert Decision Labels)."""

    def __init__(
        self,
        data_dir: str | Path,
        records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        if records is not None:
            self.records = records
        else:
            meta_path = self.data_dir / "metadata.jsonl"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"Dataset metadata not found at {meta_path}. Run `scripts/01_collect_data.py` first."
                )
            with open(meta_path, "r", encoding="utf-8") as f:
                self.records = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        img_path = self.data_dir / rec["image_path"]
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image at {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        labels = rec["labels"]
        maneuver_idx = int(
            labels.get("maneuver_idx", MANEUVER_TO_IDX.get(labels.get("maneuver", "keep_lane"), 0))
        )
        speed_level = int(labels["speed_level"])
        emergency_brake = float(labels["emergency_brake"])

        # Build 6-waypoint (3.0s, dt=0.5s) ground-truth trajectory target [x_m, y_m, v_kmh]
        speed_map = {0: 0.0, 1: 20.0, 2: 35.0, 3: 50.0}
        v_tgt_kmh = 0.0 if emergency_brake >= 0.5 else float(speed_map.get(speed_level, 35.0))
        v_tgt_ms = v_tgt_kmh / 3.6
        lat_shift = 3.5 if maneuver_idx == 1 else (-3.5 if maneuver_idx == 2 else 0.0)
        gt_traj: List[List[float]] = []
        for k in range(1, 7):
            tau = k / 6.0
            s_q = (tau ** 3) * (10.0 - 15.0 * tau + 6.0 * tau * tau)
            gt_traj.append([v_tgt_ms * (0.5 * k), lat_shift * s_q, v_tgt_kmh])

        return {
            "image": img_tensor,
            "telemetry": rec["telemetry"],
            "maneuver_idx": maneuver_idx,
            "speed_level": speed_level,
            "emergency_brake": emergency_brake,
            "target_trajectory": torch.tensor(gt_traj, dtype=torch.float32),
        }

    def compute_sample_weights(
        self, oversample_cfg: Optional[Dict[str, float]] = None
    ) -> torch.DoubleTensor:
        """Compute inverse-frequency / priority sample weights to balance rare maneuvers & AEB."""
        cfg = oversample_cfg or {}
        base_weights = {
            0: float(cfg.get("keep_lane_weight", 1.0)),
            1: float(cfg.get("change_left_weight", 4.5)),
            2: float(cfg.get("change_right_weight", 4.5)),
            3: float(cfg.get("follow_lead_weight", 2.5)),
        }
        aeb_bonus = float(cfg.get("aeb_bonus_weight", 5.0))

        # Compute square-root inverse-frequency weights so rare maneuvers are balanced without starving keep_lane
        counts = np.zeros(4, dtype=np.float64)
        for rec in self.records:
            m_idx = int(rec["labels"].get("maneuver_idx", 0))
            counts[m_idx] += 1.0
        safe_counts = np.maximum(1.0, counts)
        inv_freq = np.where(
            counts > 0, np.sqrt(len(self.records) / (4.0 * safe_counts)), 1.0
        )

        weights: List[float] = []
        for rec in self.records:
            lbl = rec["labels"]
            m_idx = int(lbl.get("maneuver_idx", 0))
            aeb = int(lbl.get("emergency_brake", 0))
            w = float(inv_freq[m_idx]) * (1.25 if m_idx in (1, 2) else 1.0)
            if aeb == 1:
                w *= 1.5
            weights.append(max(1e-4, w))

        return torch.DoubleTensor(weights)


def collate_driving_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function preserving telemetry dicts alongside stacked image and label tensors."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    telemetries = [item["telemetry"] for item in batch]
    maneuver_idx = torch.tensor(
        [item["maneuver_idx"] for item in batch], dtype=torch.long
    )
    speed_level = torch.tensor(
        [item["speed_level"] for item in batch], dtype=torch.long
    )
    emergency_brake = torch.tensor(
        [item["emergency_brake"] for item in batch], dtype=torch.float32
    )
    target_trajectory = torch.stack([item["target_trajectory"] for item in batch], dim=0)
    return {
        "images": images,
        "telemetries": telemetries,
        "maneuver_idx": maneuver_idx,
        "speed_level": speed_level,
        "emergency_brake": emergency_brake,
        "target_trajectory": target_trajectory,
    }


def build_dataloaders(
    data_dir: str | Path,
    train_split: float = 0.85,
    batch_size: int = 32,
    num_workers: int = 0,
    oversample_cfg: Optional[Dict[str, float]] = None,
    seed: int = 42,
    split_by_episode: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Split dataset into train/val by contiguous episode blocks (preventing adjacent-frame leakage)."""
    full_dataset = DrivingDataset(data_dir=data_dir)
    n_total = len(full_dataset)

    n_train = max(1, int(round(n_total * train_split)))
    if n_total - n_train < 1 and n_total >= 2:
        n_train = n_total - 1

    if split_by_episode:
        # Group by episode_id (or contiguous trajectory blocks) so no episode spans both train and val
        ep_groups: Dict[int, List[Dict[str, Any]]] = {}
        for idx, rec in enumerate(full_dataset.records):
            ep_id = int(rec.get("episode_id", idx // 100))
            ep_groups.setdefault(ep_id, []).append(rec)
        ep_ids = sorted(ep_groups.keys())
        if len(ep_ids) >= 2:
            n_train_eps = max(1, min(len(ep_ids) - 1, int(round(len(ep_ids) * train_split))))
            train_records = [r for eid in ep_ids[:n_train_eps] for r in ep_groups[eid]]
            val_records = [r for eid in ep_ids[n_train_eps:] for r in ep_groups[eid]]
        else:
            train_records = full_dataset.records[:n_train]
            val_records = full_dataset.records[n_train:]
    else:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_total).tolist()
        train_records = [full_dataset.records[i] for i in indices[:n_train]]
        val_records = [full_dataset.records[i] for i in indices[n_train:]]
    if not val_records:
        val_records = train_records

    train_ds = DrivingDataset(data_dir=data_dir, records=train_records)
    val_ds = DrivingDataset(data_dir=data_dir, records=val_records)

    sample_weights = train_ds.compute_sample_weights(oversample_cfg=oversample_cfg)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_driving_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_driving_batch,
    )
    return train_loader, val_loader
