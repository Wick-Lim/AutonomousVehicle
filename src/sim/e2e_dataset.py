r"""Obstacle-Aware Sequence Dataset & Retrospective Future-Trajectory Projection (Track 1-A).

Stores and loads contiguous driving episodes (`data_e2e/metadata.jsonl`) containing:
  - Authentic executed ego poses `(X, Y, yaw)` projected retrospectively into 6 future
    ego-frame waypoints `{(x_k, y_k, v_k)}_{k=1}^6` at `t_k = 0.5k s` (`+x` forward, `+y` LEFT).
  - Dynamic obstacle oriented bounding boxes (OBBs) in pose-`t` ego frame (`+x` forward, `+y` LEFT)
    with absolute pose-`t` velocities `(vx, vy)` for spatiotemporal collision supervision.
  - Drivable lateral bounds `[right_min_y, left_max_y]` in `+y = LEFT` coordinates.
  - Contiguous `(prev_image, image, elapsed_steps)` pairs within the same episode with
    random frame-drop augmentation (`\Delta n in {1, 2, 3}`) to train `SpatialProjector`'s
    `1.05M` temporal motion residual parameters.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.models.spatial_projector import HAZARD_SOURCE_CLASSES, TRAFFIC_LIGHT_CLASSES

MAX_OBSTACLES: int = 8
EGO_HALF_LENGTH_M: float = 2.35
EGO_HALF_WIDTH_M: float = 0.95


def project_future_trajectory_ego_frame(
    poses_xy_yaw: Sequence[Tuple[float, float, float]],
    speeds_kmh: Sequence[float],
    step_dt: float = 0.10,
    horizon_steps: int = 6,
    step_sec: float = 0.5,
    handedness: str = "right_handed",
) -> Tuple[List[List[List[float]]], List[int]]:
    """Retrospectively project an episode's executed world trajectory into ego-frame 6-WP ground truth.

    Args:
        poses_xy_yaw: List of `(X, Y, yaw_rad)` world poses for each step `0..T-1`.
        speeds_kmh: List of ego speeds (km/h) for each step `0..T-1`.
        step_dt: Elapsed simulated seconds between consecutive recorded steps.
        horizon_steps: Number of future waypoints (`6` for 0.5s .. 3.0s).
        step_sec: Time delta between waypoints (`0.5 s`).
        handedness: `"right_handed"` for MetaDrive (`+yaw` CCW, `+y_left = -dx*sin + dy*cos`),
            `"left_handed"` for CARLA Unreal (`+yaw` CW, `+y_left = dx*sin - dy*cos`).

    Returns:
        `(trajectories, valid_horizon_counts)` where `trajectories[t]` has shape `(6, 3)`
        containing `[x_forward_m, y_left_m, v_kmh]`.
    """
    n = len(poses_xy_yaw)
    if n == 0:
        return [], []
    dt = max(1e-3, float(step_dt))
    poses = np.asarray(poses_xy_yaw, dtype=np.float64)
    spds = np.asarray(speeds_kmh, dtype=np.float64)
    is_left_handed = str(handedness).lower() == "left_handed"

    trajectories: List[List[List[float]]] = []
    valid_counts: List[int] = []

    for t in range(n):
        x0, y0, yaw0 = float(poses[t, 0]), float(poses[t, 1]), float(poses[t, 2])
        c0, s0 = math.cos(yaw0), math.sin(yaw0)
        wps: List[List[float]] = []
        valid_k = 0

        for k in range(1, horizon_steps + 1):
            t_offset_steps = (k * float(step_sec)) / dt
            f_idx = float(t) + t_offset_steps
            if f_idx <= float(n - 1):
                valid_k = k
                i0 = int(math.floor(f_idx))
                i1 = min(n - 1, i0 + 1)
                alpha = f_idx - float(i0)
                wx = (1.0 - alpha) * float(poses[i0, 0]) + alpha * float(poses[i1, 0])
                wy = (1.0 - alpha) * float(poses[i0, 1]) + alpha * float(poses[i1, 1])
                vk = (1.0 - alpha) * float(spds[i0]) + alpha * float(spds[i1])
            else:
                # Constant-velocity extrapolation along final recorded heading
                extra_s = (f_idx - float(n - 1)) * dt
                v_end_mps = max(0.0, float(spds[-1]) / 3.6)
                yaw_end = float(poses[-1, 2])
                wx = float(poses[-1, 0]) + v_end_mps * math.cos(yaw_end) * extra_s
                wy = float(poses[-1, 1]) + v_end_mps * math.sin(yaw_end) * extra_s
                vk = float(spds[-1])

            dx_w = wx - x0
            dy_w = wy - y0
            x_ego = dx_w * c0 + dy_w * s0
            if is_left_handed:
                y_ego = dx_w * s0 - dy_w * c0
            else:
                y_ego = -dx_w * s0 + dy_w * c0

            # Clamp tiny numerical backward jitter at standstill to monotonic >= 0
            x_ego = max(0.0, float(x_ego))
            y_ego = float(np.clip(y_ego, -4.45, 4.45))
            vk = float(np.clip(vk, 0.0, 70.0))
            wps.append([round(x_ego, 3), round(y_ego, 3), round(vk, 2)])

        # Enforce strict forward monotonicity on x_k
        for k in range(1, len(wps)):
            if wps[k][0] < wps[k - 1][0]:
                wps[k][0] = wps[k - 1][0]

        trajectories.append(wps)
        valid_counts.append(valid_k)

    return trajectories, valid_counts


def extract_obstacles_ego_left_frame(
    radar_targets: Optional[Sequence[Dict[str, Any]]],
    ego_speed_kmh: float,
    telemetry: Optional[Dict[str, Any]] = None,
    max_obstacles: int = MAX_OBSTACLES,
) -> List[Dict[str, float]]:
    """Convert `radar_targets` (`+x` forward, `+y` RIGHT) into pose-`t` `+x` forward, `+y` LEFT OBBs.

    Velocities `(vx, vy)` are stored as **absolute velocities expressed in pose-`t` ego frame**
    so an obstacle's future position at `t + t_k` in pose-`t` coordinates is simply:
      `(x + vx * t_k, y + vy * t_k)`.
    """
    ego_v_mps = max(0.0, float(ego_speed_kmh) / 3.6)
    obstacles: List[Dict[str, float]] = []

    if radar_targets:
        for rad in radar_targets:
            x_fwd = float(rad.get("x_m", 99.0))
            # radar_targets uses +y = RIGHT; invert sign to unified +y = LEFT
            y_left = -float(rad.get("y_m", 0.0))
            if not (-6.0 <= x_fwd <= 55.0 and abs(y_left) <= 12.0):
                continue

            cat = str(rad.get("category", "vehicle")).lower()
            is_ped = 1.0 if ("ped" in cat or "walk" in cat) else 0.0
            vx_rel = float(rad.get("vx_rel_mps", 0.0))
            vy_rel_right = float(rad.get("vy_rel_mps", 0.0))
            vx_abs = float(np.clip(vx_rel + ego_v_mps, -15.0, 35.0))
            vy_abs_left = float(np.clip(-vy_rel_right, -10.0, 10.0))

            obstacles.append(
                {
                    "x": round(x_fwd, 3),
                    "y": round(y_left, 3),
                    "vx": round(vx_abs, 3),
                    "vy": round(vy_abs_left, 3),
                    "half_l": 0.45 if is_ped > 0.5 else 2.35,
                    "half_w": 0.45 if is_ped > 0.5 else 0.95,
                    "is_ped": is_ped,
                }
            )

    # Fallback synthetic obstacle from structured telemetry when radar_targets is absent (e.g., mock mode)
    if not obstacles and telemetry is not None:
        lead_d = float(telemetry.get("lead_distance_m", 60.0))
        lead_v_mps = max(0.0, float(telemetry.get("lead_speed_kmh", 40.0)) / 3.6)
        ped_d = float(telemetry.get("pedestrian_distance_m", 100.0))
        if ped_d < 45.0:
            obstacles.append(
                {
                    "x": round(ped_d, 3),
                    "y": 0.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "half_l": 0.45,
                    "half_w": 0.45,
                    "is_ped": 1.0,
                }
            )
        if lead_d < 50.0:
            hz = str(telemetry.get("hazard_source", "vehicle")).lower()
            is_ped = 1.0 if hz == "pedestrian" else 0.0
            obstacles.append(
                {
                    "x": round(lead_d, 3),
                    "y": 0.0,
                    "vx": round(0.0 if is_ped > 0.5 else lead_v_mps, 3),
                    "vy": 0.0,
                    "half_l": 0.45 if is_ped > 0.5 else 2.35,
                    "half_w": 0.45 if is_ped > 0.5 else 0.95,
                    "is_ped": is_ped,
                }
            )

    obstacles.sort(key=lambda o: math.hypot(o["x"], o["y"]))
    return obstacles[: max(1, int(max_obstacles))]


def compute_drivable_lane_bounds(
    telemetry: Dict[str, Any], lane_width: float = 3.5
) -> Tuple[float, float]:
    """Compute ego-frame drivable lateral corridor `[right_min_y, left_max_y]` (`+y` = LEFT)."""
    lane_off = float(telemetry.get("lane_offset_m", 0.0))
    # If ego is +lane_off to the LEFT of current lane center, lane center is at y = -lane_off
    center_y = -lane_off
    half_w = 0.5 * float(lane_width)
    margin = 0.35

    left_max = center_y + half_w - margin
    right_min = center_y - half_w + margin

    if bool(telemetry.get("left_lane_available", False)):
        left_max += float(lane_width)
    if bool(telemetry.get("right_lane_available", False)):
        right_min -= float(lane_width)

    right_min = float(np.clip(right_min, -4.4, 4.4))
    left_max = float(np.clip(left_max, -4.4, 4.4))
    if right_min > left_max:
        mid = 0.5 * (right_min + left_max)
        right_min, left_max = mid - 0.5, mid + 0.5
    return round(right_min, 3), round(left_max, 3)


def encode_obstacles_tensor(
    obstacles: Optional[Sequence[Dict[str, Any]]],
    max_obstacles: int = MAX_OBSTACLES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack variable-length obstacle dicts into `(max_obstacles, 7)` tensor + `(max_obstacles,)` mask."""
    arr = np.zeros((max_obstacles, 7), dtype=np.float32)
    mask = np.zeros((max_obstacles,), dtype=np.float32)
    if obstacles:
        for i, obs in enumerate(obstacles[:max_obstacles]):
            arr[i, 0] = float(obs.get("x", 0.0))
            arr[i, 1] = float(obs.get("y", 0.0))
            arr[i, 2] = float(obs.get("vx", 0.0))
            arr[i, 3] = float(obs.get("vy", 0.0))
            arr[i, 4] = float(obs.get("half_l", 2.35))
            arr[i, 5] = float(obs.get("half_w", 0.95))
            arr[i, 6] = float(obs.get("is_ped", 0.0))
            mask[i] = 1.0
    return torch.from_numpy(arr), torch.from_numpy(mask)


class ObstacleAwareSequenceDataset(Dataset):
    """Contiguous episode-sequence dataset for Stage-1 Obstacle-Aware E2E trajectory learning."""

    def __init__(
        self,
        data_dir: str | Path,
        records: Optional[List[Dict[str, Any]]] = None,
        augment_frame_drop: bool = True,
        max_frame_drop: int = 3,
        patch_cache_path: Optional[str | Path] = None,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.augment_frame_drop = bool(augment_frame_drop)
        self.max_frame_drop = max(1, int(max_frame_drop))
        self._rng = np.random.default_rng(int(seed))

        if records is not None:
            self.records = list(records)
        else:
            meta_path = self.data_dir / "metadata.jsonl"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"E2E sequence metadata not found at {meta_path}. Run `scripts/07_train_e2e_trajectory.py --collect` first."
                )
            self.records = []
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))

        # Index contiguous positions within each episode for temporal (t - delta_n, t) pairing
        ep_to_indices: Dict[Any, List[Tuple[int, int]]] = defaultdict(list)
        for idx, rec in enumerate(self.records):
            ep_id = rec.get("episode_id", 0)
            step = int(rec.get("step", idx))
            ep_to_indices[ep_id].append((step, idx))

        self._ep_pos: List[Tuple[Any, int]] = [("", 0)] * len(self.records)
        self._ep_ordered: Dict[Any, List[int]] = {}
        for ep_id, items in ep_to_indices.items():
            items.sort(key=lambda pair: pair[0])
            ordered_idx = [pair[1] for pair in items]
            self._ep_ordered[ep_id] = ordered_idx
            for pos, global_idx in enumerate(ordered_idx):
                self._ep_pos[global_idx] = (ep_id, pos)

        # Optional memory-mapped frozen DINOv2 FP16 patch token cache
        self._patch_cache: Optional[np.ndarray] = None
        if patch_cache_path is not None and Path(patch_cache_path).is_file():
            self._patch_cache = np.load(str(patch_cache_path), mmap_mode="r")

    def __len__(self) -> int:
        return len(self.records)

    def _load_image_tensor(self, rel_path: str) -> torch.Tensor:
        img_path = self.data_dir / rel_path
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            rgb = np.zeros((196, 392, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (196, 392):
                rgb = cv2.resize(rgb, (392, 196), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    def _select_prev_index(self, idx: int) -> Tuple[int, int, bool]:
        ep_id, pos = self._ep_pos[idx]
        if pos <= 0:
            return idx, 1, False
        if self.augment_frame_drop:
            max_drop = min(self.max_frame_drop, pos)
            delta_n = int(self._rng.integers(1, max_drop + 1))
        else:
            delta_n = 1
        prev_idx = self._ep_ordered[ep_id][pos - delta_n]
        return prev_idx, delta_n, True

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        prev_idx, elapsed_steps, has_prev = self._select_prev_index(idx)
        prev_rec = self.records[prev_idx]

        tel = rec["telemetry"]
        labels = rec["labels"]

        if self._patch_cache is not None and "global_idx" in rec and "global_idx" in prev_rec:
            g_cur = int(rec["global_idx"])
            g_prev = int(prev_rec["global_idx"])
            patch_tokens = torch.from_numpy(np.array(self._patch_cache[g_cur], dtype=np.float32))
            prev_patch_tokens = torch.from_numpy(np.array(self._patch_cache[g_prev], dtype=np.float32))
            image = None
            prev_image = None
        else:
            image = self._load_image_tensor(rec["image_path"])
            prev_image = (
                self._load_image_tensor(prev_rec["image_path"])
                if has_prev
                else image.clone()
            )
            patch_tokens = None
            prev_patch_tokens = None

        future_traj = torch.tensor(rec["future_trajectory_6wp"], dtype=torch.float32)
        obs_tensor, obs_mask = encode_obstacles_tensor(rec.get("obstacles", []))
        bounds = rec.get("drivable_bounds")
        if bounds is None or len(bounds) != 2:
            bounds = list(compute_drivable_lane_bounds(tel))
        drivable_bounds = torch.tensor(bounds, dtype=torch.float32)

        tl_str = str(tel.get("traffic_light", "green")).lower()
        tl_idx = TRAFFIC_LIGHT_CLASSES.index(tl_str) if tl_str in TRAFFIC_LIGHT_CLASSES else 0
        hz_str = str(tel.get("hazard_source", "none")).lower()
        hz_idx = HAZARD_SOURCE_CLASSES.index(hz_str) if hz_str in HAZARD_SOURCE_CLASSES else 0

        item: Dict[str, Any] = {
            "telemetry": tel,
            "elapsed_steps": int(elapsed_steps),
            "has_prev": bool(has_prev),
            "maneuver_idx": torch.tensor(int(labels["maneuver_idx"]), dtype=torch.long),
            "speed_level": torch.tensor(int(labels["speed_level"]), dtype=torch.long),
            "emergency_brake": torch.tensor(
                float(labels["emergency_brake"]), dtype=torch.float32
            ),
            "future_trajectory": future_traj,
            "obstacles": obs_tensor,
            "obstacle_mask": obs_mask,
            "drivable_bounds": drivable_bounds,
            "lead_distance_m": torch.tensor(
                float(tel.get("lead_distance_m", 60.0)), dtype=torch.float32
            ),
            "pedestrian_distance_m": torch.tensor(
                float(tel.get("pedestrian_distance_m", 100.0)), dtype=torch.float32
            ),
            "lane_offset_m": torch.tensor(
                float(tel.get("lane_offset_m", 0.0)), dtype=torch.float32
            ),
            "traffic_light_idx": torch.tensor(tl_idx, dtype=torch.long),
            "hazard_idx": torch.tensor(hz_idx, dtype=torch.long),
            "blindspot_targets": torch.tensor(
                [
                    1.0 if bool(tel.get("blind_spot_left", False)) else 0.0,
                    1.0 if bool(tel.get("blind_spot_right", False)) else 0.0,
                ],
                dtype=torch.float32,
            ),
            "episode_id": rec.get("episode_id", 0),
        }
        if patch_tokens is not None:
            item["patch_tokens"] = patch_tokens
            item["prev_patch_tokens"] = prev_patch_tokens
        else:
            item["image"] = image
            item["prev_image"] = prev_image
        return item

    def compute_sample_weights(self) -> List[float]:
        """Upweight obstacle avoidance, braking, lane-change, pedestrian, and curve frames."""
        weights: List[float] = []
        for rec in self.records:
            tel = rec["telemetry"]
            lbl = rec["labels"]
            w = 1.0
            m_idx = int(lbl.get("maneuver_idx", 0))
            if m_idx in (1, 2):
                w = max(w, 3.5)
            elif m_idx == 3:
                w = max(w, 2.0)
            if float(lbl.get("emergency_brake", 0.0)) >= 0.5:
                w = max(w, 4.5)
            lead_d = float(tel.get("lead_distance_m", 60.0))
            ped_d = float(tel.get("pedestrian_distance_m", 100.0))
            if ped_d < 30.0 or lead_d < 22.0:
                w = max(w, 3.0)
            if abs(float(tel.get("road_curvature", 0.0))) >= 0.25:
                w = max(w, 2.2)
            # Upweight frames with strong real lateral avoidance in future trajectory
            traj = rec.get("future_trajectory_6wp", [])
            if traj and max(abs(float(wp[1])) for wp in traj) >= 1.2:
                w = max(w, 3.2)
            weights.append(float(w))
        return weights


def collate_e2e_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate sequence samples while keeping telemetry dicts as Python list."""
    out: Dict[str, Any] = {
        "telemetries": [b["telemetry"] for b in batch],
        "elapsed_steps": torch.tensor(
            [b["elapsed_steps"] for b in batch], dtype=torch.float32
        ),
        "has_prev": torch.tensor([b["has_prev"] for b in batch], dtype=torch.bool),
        "maneuver_idx": torch.stack([b["maneuver_idx"] for b in batch], dim=0),
        "speed_level": torch.stack([b["speed_level"] for b in batch], dim=0),
        "emergency_brake": torch.stack([b["emergency_brake"] for b in batch], dim=0),
        "future_trajectory": torch.stack([b["future_trajectory"] for b in batch], dim=0),
        "obstacles": torch.stack([b["obstacles"] for b in batch], dim=0),
        "obstacle_mask": torch.stack([b["obstacle_mask"] for b in batch], dim=0),
        "drivable_bounds": torch.stack([b["drivable_bounds"] for b in batch], dim=0),
        "lead_distance_m": torch.stack([b["lead_distance_m"] for b in batch], dim=0),
        "pedestrian_distance_m": torch.stack(
            [b["pedestrian_distance_m"] for b in batch], dim=0
        ),
        "lane_offset_m": torch.stack([b["lane_offset_m"] for b in batch], dim=0),
        "traffic_light_idx": torch.stack(
            [b["traffic_light_idx"] for b in batch], dim=0
        ),
        "hazard_idx": torch.stack([b["hazard_idx"] for b in batch], dim=0),
        "blindspot_targets": torch.stack(
            [b["blindspot_targets"] for b in batch], dim=0
        ),
    }
    if "patch_tokens" in batch[0]:
        out["patch_tokens"] = torch.stack([b["patch_tokens"] for b in batch], dim=0)
        out["prev_patch_tokens"] = torch.stack(
            [b["prev_patch_tokens"] for b in batch], dim=0
        )
    else:
        out["images"] = torch.stack([b["image"] for b in batch], dim=0)
        out["prev_images"] = torch.stack([b["prev_image"] for b in batch], dim=0)
    return out


def split_records_by_episode(
    records: List[Dict[str, Any]],
    train_split: float = 0.85,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split sequence records strictly by `episode_id` to prevent temporal frame leakage."""
    ep_groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for g_idx, rec in enumerate(records):
        rec_copy = dict(rec)
        rec_copy.setdefault("global_idx", g_idx)
        ep_groups[rec_copy.get("episode_id", 0)].append(rec_copy)

    ep_ids = sorted(ep_groups.keys(), key=str)
    if len(ep_ids) <= 1:
        all_recs = [r for grp in ep_groups.values() for r in grp]
        cut = max(1, int(len(all_recs) * train_split))
        return all_recs[:cut], all_recs[cut:] or all_recs[:1]

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(ep_ids))
    n_train_eps = max(1, min(len(ep_ids) - 1, int(round(len(ep_ids) * train_split))))
    train_ep_set = {ep_ids[i] for i in perm[:n_train_eps]}

    train_recs: List[Dict[str, Any]] = []
    val_recs: List[Dict[str, Any]] = []
    for ep_id in ep_ids:
        if ep_id in train_ep_set:
            train_recs.extend(ep_groups[ep_id])
        else:
            val_recs.extend(ep_groups[ep_id])
    return train_recs, val_recs


def build_e2e_dataloaders(
    data_dir: str | Path,
    train_split: float = 0.85,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    patch_cache_path: Optional[str | Path] = None,
) -> Tuple[DataLoader, DataLoader, ObstacleAwareSequenceDataset, ObstacleAwareSequenceDataset]:
    """Create episode-disjoint train/val DataLoaders for Stage-1 Obstacle-Aware E2E training."""
    data_path = Path(data_dir)
    meta_path = data_path / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    all_records: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for g_idx, line in enumerate(f):
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["global_idx"] = g_idx
                all_records.append(rec)

    train_recs, val_recs = split_records_by_episode(
        all_records, train_split=train_split, seed=seed
    )
    train_ds = ObstacleAwareSequenceDataset(
        data_dir=data_path,
        records=train_recs,
        augment_frame_drop=True,
        max_frame_drop=3,
        patch_cache_path=patch_cache_path,
        seed=seed,
    )
    val_ds = ObstacleAwareSequenceDataset(
        data_dir=data_path,
        records=val_recs,
        augment_frame_drop=False,
        max_frame_drop=1,
        patch_cache_path=patch_cache_path,
        seed=seed + 1,
    )

    sample_weights = train_ds.compute_sample_weights()
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_e2e_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_e2e_batch,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_ds, val_ds


__all__ = [
    "EGO_HALF_LENGTH_M",
    "EGO_HALF_WIDTH_M",
    "MAX_OBSTACLES",
    "ObstacleAwareSequenceDataset",
    "build_e2e_dataloaders",
    "collate_e2e_batch",
    "compute_drivable_lane_bounds",
    "encode_obstacles_tensor",
    "extract_obstacles_ego_left_frame",
    "project_future_trajectory_ego_frame",
    "split_records_by_episode",
]
