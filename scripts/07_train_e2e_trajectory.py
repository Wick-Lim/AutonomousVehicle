"""Stage 1 (Track 1-A): Obstacle-Aware Pure End-to-End Trajectory & Pure-Neural Learning.

Two-phase pipeline:
  1. Collects crash-free contiguous episodes from real 3D MetaDrive (train seeds 1000..1199,
     strictly disjoint from val 4000..4005 & eval 5000+) into `data_e2e/`, retrospectively
     projecting authentic executed ego poses into 6 future waypoints `{(x_k, y_k, v_k)}_{k=1}^6`
     alongside dynamic obstacle OBBs and drivable lateral bounds.
  2. Trains the exact `4,232,991`-parameter adapter (`SpatialProjector` + `ReadoutHeads` +
     `TrajectoryHead`) in `decision_mode="pure_neural"` & `trajectory_mode="obstacle_aware_e2e"`
     with `enable_temporal_learning()` active, optimizing the differentiable obstacle-repulsion,
     lane-boundary, minimum-jerk, pure-neural readout, and pure-vision auxiliary losses.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_validated_config
from src.control.shield_pid import SafetyShieldPID
from src.env.expert_labeler import ExpertLabeler
from src.env.metadrive_wrapper import MetaDriveWrapper
from src.models.laya_pilot import AutonomousDrivingSystem
from src.sim.e2e_dataset import (
    build_e2e_dataloaders,
    compute_drivable_lane_bounds,
    extract_obstacles_ego_left_frame,
    project_future_trajectory_ego_frame,
)
from src.sim.e2e_losses import (
    compute_obstacle_aware_e2e_loss,
    compute_open_loop_trajectory_metrics,
)
from src.sim.scenarios import SCENARIO_CATALOG, RealisticSimEnvWrapper
from src.utils.seed import seed_everything

TRAIN_SCENARIO_SCHEDULE: Tuple[str, ...] = (
    "nominal_urban_cruise",
    "nominal_urban_cruise",
    "aggressive_cut_in",
    "sudden_lead_ebrake",
    "blindspot_highspeed_overtaker",
    "occluded_jaywalker_dartout",
    "sotif_heavy_rain_glare_dropout",
)


def _resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def collect_e2e_sequence_dataset(
    cfg: Dict[str, Any],
    data_dir: Path,
    num_frames: int = 12000,
    start_seed: int = 1000,
    num_scenarios: int = 200,
    use_mock: bool = False,
    max_steps_per_ep: int = 360,
    simulator: str = "metadrive",
) -> int:
    """Collect crash-free contiguous episodes with retrospective 6-WP ground-truth trajectories."""
    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "metadata.jsonl"

    # Build real MetaDrive (or mock) base environment strictly on training seeds [1000..1199]
    collect_cfg = json.loads(json.dumps(cfg))
    collect_cfg.setdefault("env", {})["start_seed"] = int(start_seed)
    collect_cfg["env"]["max_episode_steps"] = int(max_steps_per_ep)
    collect_cfg["env"]["traffic_density"] = float(
        collect_cfg["env"].get("traffic_density", 0.12)
    )

    if simulator == "carla" and not use_mock:
        from src.env.carla_wrapper import CarlaWrapper

        collect_cfg["env"].setdefault("carla", {})["town"] = "Town01_Opt"
        base_env: Any = CarlaWrapper(
            config=collect_cfg,
            use_mock=False,
            start_seed=start_seed,
            num_scenarios=num_scenarios,
        )
        handedness = "left_handed"
    else:
        base_env = MetaDriveWrapper(
            config=collect_cfg,
            use_mock=use_mock,
            start_seed=start_seed,
            num_scenarios=num_scenarios,
        )
        handedness = "right_handed"

    env = RealisticSimEnvWrapper(
        base_env=base_env,
        scenario="nominal_urban_cruise",
        actuator_realism=True,
        localization_mode="hd_map",
    )
    expert_ctrl = SafetyShieldPID(config=collect_cfg, use_mock=use_mock)
    labeler = ExpertLabeler()

    total_saved = 0
    ep_idx = 0
    kept_episodes = 0
    dropped_episodes = 0
    t_start = time.perf_counter()

    try:
        with open(meta_path, "w", encoding="utf-8") as meta_f:
            while total_saved < num_frames and ep_idx < max(20, num_frames // 10):
                sc_name = TRAIN_SCENARIO_SCHEDULE[ep_idx % len(TRAIN_SCENARIO_SCHEDULE)]
                if sc_name not in SCENARIO_CATALOG:
                    sc_name = "nominal_urban_cruise"
                ep_seed = int(start_seed) + (ep_idx % max(1, int(num_scenarios)))

                env.set_scenario(sc_name)
                rgb, tel, info = env.reset(seed=ep_seed)
                expert_ctrl.reset()

                step_dt = float(env.step_dt)
                ep_frames: List[Dict[str, Any]] = []
                poses: List[Tuple[float, float, float]] = []
                speeds: List[float] = []

                # Synthetic fallback integrator for mock mode where nav_state may be absent
                sim_x, sim_y, sim_yaw = 0.0, float(tel.get("lane_offset_m", 0.0)), 0.0
                crashed = False

                for step_i in range(max_steps_per_ep):
                    # Extract exact world pose (X, Y, yaw)
                    nav_state = info.get("nav_state") if isinstance(info, dict) else None
                    if (
                        isinstance(nav_state, dict)
                        and isinstance(nav_state.get("ego_xy_yaw"), (list, tuple))
                        and len(nav_state["ego_xy_yaw"]) >= 3
                    ):
                        ex, ey, eyaw = (
                            float(nav_state["ego_xy_yaw"][0]),
                            float(nav_state["ego_xy_yaw"][1]),
                            float(nav_state["ego_xy_yaw"][2]),
                        )
                    else:
                        ex, ey, eyaw = sim_x, sim_y, sim_yaw

                    ego_v_kmh = float(tel.get("speed_kmh", 0.0))
                    exp_act = (
                        env.base_env.get_expert_action(info)
                        if hasattr(env.base_env, "get_expert_action")
                        else (0.0, 0.0)
                    )
                    labels = labeler.label_step(tel, exp_act, info).to_dict()
                    obstacles = extract_obstacles_ego_left_frame(
                        radar_targets=info.get("radar_targets"),
                        ego_speed_kmh=ego_v_kmh,
                        telemetry=tel,
                    )
                    bounds = compute_drivable_lane_bounds(
                        tel, lane_width=float(collect_cfg["env"].get("lane_width", 3.5))
                    )

                    poses.append((ex, ey, eyaw))
                    speeds.append(ego_v_kmh)
                    ep_frames.append(
                        {
                            "rgb": rgb.copy(),
                            "step": step_i,
                            "telemetry": dict(tel),
                            "labels": dict(labels),
                            "obstacles": obstacles,
                            "drivable_bounds": [bounds[0], bounds[1]],
                        }
                    )

                    # Expert safe control: ShieldPID tracking expert decision
                    exp_dec = {
                        "maneuver": labels["maneuver"],
                        "maneuver_idx": labels["maneuver_idx"],
                        "maneuver_confidence": 0.99,
                        "speed_level": labels["speed_level"],
                        "target_speed_kmh": {0: 0.0, 1: 20.0, 2: 40.0, 3: 65.0}.get(
                            int(labels["speed_level"]), 40.0
                        ),
                        "speed_confidence": 0.99,
                        "emergency_brake": float(labels["emergency_brake"]),
                    }
                    (steer, tb), gated = expert_ctrl.compute_control(
                        exp_dec, tel, dt=step_dt, shield_enabled=True
                    )
                    setattr(env, "_last_gated_maneuver", str(gated.get("maneuver", "keep_lane")))
                    setattr(
                        env.base_env,
                        "_last_gated_maneuver",
                        str(gated.get("maneuver", "keep_lane")),
                    )

                    rgb, tel, _rew, done, info = env.step((steer, tb))

                    # Advance mock fallback pose if needed
                    v_mps = float(tel.get("speed_kmh", 0.0)) / 3.6
                    sim_yaw = float(tel.get("heading_error", 0.0))
                    sim_x += v_mps * math.cos(sim_yaw) * step_dt
                    sim_y = float(tel.get("lane_offset_m", 0.0))

                    if bool(info.get("crash", False)) or bool(info.get("out_of_road", False)):
                        crashed = True
                        break
                    if done:
                        break

                ep_idx += 1

                # Keep only crash-free episodes with sufficient length for authentic 3.0s future projection
                min_ep_len = max(12, int(round(1.5 / max(step_dt, 0.02))))
                if crashed or len(ep_frames) < min_ep_len:
                    dropped_episodes += 1
                    continue

                fut_trajs, valid_counts = project_future_trajectory_ego_frame(
                    poses_xy_yaw=poses,
                    speeds_kmh=speeds,
                    step_dt=step_dt,
                    horizon_steps=6,
                    step_sec=0.5,
                    handedness=handedness,
                )

                # Trim tail frames that lack at least 1.5s of authentic executed future motion
                usable = [
                    i
                    for i in range(len(ep_frames))
                    if valid_counts[i] >= 3
                ]
                if not usable:
                    usable = list(range(len(ep_frames)))

                ep_id = f"{simulator}_{ep_seed:04d}_e{kept_episodes:04d}"
                for i in usable:
                    if total_saved >= num_frames:
                        break
                    fr = ep_frames[i]
                    rel_img = f"images/{ep_id}_s{fr['step']:04d}.jpg"
                    bgr = cv2.cvtColor(fr["rgb"], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(
                        str(data_dir / rel_img),
                        bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                    )
                    rec = {
                        "episode_id": ep_id,
                        "seed": ep_seed,
                        "simulator": simulator,
                        "scenario": sc_name,
                        "step": int(fr["step"]),
                        "step_dt": round(step_dt, 4),
                        "image_path": rel_img,
                        "telemetry": fr["telemetry"],
                        "labels": fr["labels"],
                        "future_trajectory_6wp": fut_trajs[i],
                        "future_valid_steps": int(valid_counts[i]),
                        "obstacles": fr["obstacles"],
                        "drivable_bounds": fr["drivable_bounds"],
                    }
                    meta_f.write(json.dumps(rec) + "\n")
                    total_saved += 1

                kept_episodes += 1
                if kept_episodes % 5 == 0 or total_saved >= num_frames:
                    elapsed = time.perf_counter() - t_start
                    fps = total_saved / max(elapsed, 1e-3)
                    print(
                        f"[07_collect] frames={total_saved}/{num_frames} | "
                        f"kept_eps={kept_episodes} | dropped_eps={dropped_episodes} | "
                        f"last_sc={sc_name} | rate={fps:.1f} fps",
                        flush=True,
                    )
    finally:
        env.close()

    print(
        f"[07_collect] Completed: {total_saved} frames across {kept_episodes} crash-free episodes "
        f"(dropped={dropped_episodes}) -> {meta_path}",
        flush=True,
    )
    return total_saved


@torch.no_grad()
def build_or_load_dino_patch_cache(
    system: AutonomousDrivingSystem,
    data_dir: Path,
    device: torch.device,
    batch_size: int = 64,
    rebuild: bool = False,
) -> Path:
    """Pre-extract frozen DINOv2 FP16 patch tokens `(N, 392, 768)` to accelerate multi-epoch GPU training."""
    meta_path = data_dir / "metadata.jsonl"
    cache_path = data_dir / "dino_patch_cache.fp16.npy"

    img_paths: List[Path] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                img_paths.append(data_dir / rec["image_path"])

    n_total = len(img_paths)
    expected_shape = (n_total, int(system.dino.num_patches), int(system.dino.embed_dim))
    if not rebuild and cache_path.is_file():
        try:
            existing = np.load(str(cache_path), mmap_mode="r")
            if tuple(existing.shape) == expected_shape and existing.dtype == np.float16:
                print(
                    f"[07_cache] Reusing verified DINOv2 FP16 cache: {cache_path} {expected_shape}",
                    flush=True,
                )
                return cache_path
        except Exception:
            pass

    print(
        f"[07_cache] Extracting frozen DINOv2 patch tokens for {n_total} frames on {device}...",
        flush=True,
    )
    t0 = time.perf_counter()
    mm = np.lib.format.open_memmap(
        str(cache_path),
        mode="w+",
        dtype=np.float16,
        shape=expected_shape,
    )
    system.dino.eval()
    use_bf16 = device.type == "cuda"

    for start in range(0, n_total, batch_size):
        end = min(n_total, start + batch_size)
        imgs: List[np.ndarray] = []
        for p in img_paths[start:end]:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                rgb = np.zeros((196, 392, 3), dtype=np.uint8)
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != (196, 392):
                    rgb = cv2.resize(rgb, (392, 196), interpolation=cv2.INTER_AREA)
            imgs.append(rgb)
        batch_np = np.stack(imgs, axis=0)  # (B, 196, 392, 3)
        batch_t = (
            torch.from_numpy(batch_np)
            .permute(0, 3, 1, 2)
            .to(device=device, dtype=torch.float32)
            / 255.0
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            patches = system.dino(batch_t)  # (B, 392, 768)
        mm[start:end] = patches.detach().cpu().to(torch.float16).numpy()

        if ((start // batch_size) % 25 == 0) or end == n_total:
            dt = max(1e-3, time.perf_counter() - t0)
            print(
                f"  cached {end}/{n_total} ({end / dt:.1f} img/s)",
                flush=True,
            )

    mm.flush()
    del mm
    print(
        f"[07_cache] Saved DINOv2 FP16 patch cache ({time.perf_counter() - t0:.1f}s) -> {cache_path}",
        flush=True,
    )
    return cache_path


def _forward_e2e_batch(
    system: AutonomousDrivingSystem,
    batch: Dict[str, Any],
    device: torch.device,
    use_amp: bool,
    prompt_offset_dropout: float = 0.0,
) -> Dict[str, Any]:
    """Run contiguous `(prev, cur)` temporal-aware forward pass in `pure_neural` + `obstacle_aware_e2e` mode."""
    raw_tels = batch["telemetries"]
    if prompt_offset_dropout > 0.0 and system.training:
        telemetries = []
        for t in raw_tels:
            if float(torch.rand(()).item()) < prompt_offset_dropout:
                tc = dict(t)
                tc["lane_offset_m"] = 0.0
                telemetries.append(tc)
            else:
                telemetries.append(t)
    else:
        telemetries = raw_tels
    elapsed_steps = batch["elapsed_steps"].to(device=device, dtype=torch.float32)

    if "patch_tokens" in batch:
        cur_patch = batch["patch_tokens"].to(device=device, dtype=torch.float32)
        prev_patch = batch["prev_patch_tokens"].to(device=device, dtype=torch.float32)
    else:
        cur_imgs = batch["images"].to(device=device, dtype=torch.float32)
        prev_imgs = batch["prev_images"].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                cur_patch = system.dino(cur_imgs)
                prev_patch = system.dino(prev_imgs)

    # Encode previous frame spatial tokens without mutating module buffer (matches online ring buffer)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            prev_vis = system.projector(prev_patch, use_temporal=False).detach()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        outputs = system.forward_from_patch_tokens(
            patch_tokens=cur_patch,
            telemetries=telemetries,
            elapsed_steps=elapsed_steps,
            prev_visual_tokens=prev_vis,
            aux_use_prior=False,  # 100% pure visual supervision
        )
    return outputs


@torch.no_grad()
def evaluate_e2e_epoch(
    system: AutonomousDrivingSystem,
    val_loader: Any,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """Evaluate Stage-1 obstacle-aware E2E model on episode-disjoint validation split."""
    system.eval()
    totals = {
        "loss": 0.0,
        "traj": 0.0,
        "coll": 0.0,
        "lane": 0.0,
        "ade_3s_m": 0.0,
        "fde_3s_m": 0.0,
        "lat_ade_3s_m": 0.0,
        "cr_traj": 0.0,
        "man_acc": 0.0,
        "spd_acc": 0.0,
        "aeb_acc": 0.0,
    }
    n_samples = 0

    for batch in val_loader:
        bsz = len(batch["telemetries"])
        out = _forward_e2e_batch(system, batch, device=device, use_amp=use_amp)
        losses = compute_obstacle_aware_e2e_loss(out, batch)
        metrics = compute_open_loop_trajectory_metrics(
            pred_traj=out["predicted_trajectory"],
            gt_traj=batch["future_trajectory"].to(device),
            obstacles=batch["obstacles"].to(device),
            obstacle_mask=batch["obstacle_mask"].to(device),
        )

        m_pred = out["mlp_maneuver_logits"].argmax(dim=-1)
        s_pred = out["mlp_speed_logits"].argmax(dim=-1)
        a_pred = (torch.sigmoid(out["mlp_aeb_logits"].squeeze(-1)) >= 0.5).float()

        m_acc = float((m_pred == batch["maneuver_idx"].to(device)).float().mean().item())
        s_acc = float((s_pred == batch["speed_level"].to(device)).float().mean().item())
        a_acc = float((a_pred == batch["emergency_brake"].to(device)).float().mean().item())

        totals["loss"] += float(losses["total"].item()) * bsz
        totals["traj"] += float(losses["traj"].item()) * bsz
        totals["coll"] += float(losses["coll"].item()) * bsz
        totals["lane"] += float(losses["lane"].item()) * bsz
        totals["ade_3s_m"] += metrics["ade_3s_m"] * bsz
        totals["fde_3s_m"] += metrics["fde_3s_m"] * bsz
        totals["lat_ade_3s_m"] += metrics["lat_ade_3s_m"] * bsz
        totals["cr_traj"] += metrics["cr_traj"] * bsz
        totals["man_acc"] += m_acc * bsz
        totals["spd_acc"] += s_acc * bsz
        totals["aeb_acc"] += a_acc * bsz
        n_samples += bsz

    denom = max(1, n_samples)
    res = {k: round(v / denom, 5) for k, v in totals.items()}
    res["abs_temporal_gate"] = round(
        float(system.projector.temporal_gate.detach().abs().item()), 6
    )
    res["temporal_proj_norm"] = round(
        float(system.projector.temporal_proj.weight.detach().norm().item()), 4
    )
    return res


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 (Track 1-A): Obstacle-Aware Pure E2E Trajectory & Temporal Learning."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument("--simulator", type=str, default="metadrive", choices=("metadrive", "carla"))
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data_e2e"))
    parser.add_argument("--collect", action="store_true", help="Collect fresh contiguous 3D sequence data.")
    parser.add_argument("--num-frames", type=int, default=12000)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--num-scenarios", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--init-ckpt", type=str, default=None)
    parser.add_argument(
        "--output-ckpt",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "projector_e2e_stage1.pt"),
    )
    parser.add_argument(
        "--history-json",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "stage1_e2e_train_history.json"),
    )
    parser.add_argument("--no-cache-dino", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    cfg = load_validated_config(args.config).to_dict()
    seed = int(cfg.get("project", {}).get("seed", 42))
    seed_everything(seed)

    use_mock = bool(args.mock or cfg.get("project", {}).get("use_mock", False))
    device = _resolve_device(str(cfg.get("project", {}).get("device", "auto")))
    data_dir = Path(args.data_dir)
    out_ckpt = Path(args.output_ckpt)
    hist_path = Path(args.history_json)
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Collect authentic sequence dataset if requested or absent
    meta_path = data_dir / "metadata.jsonl"
    if args.collect or not meta_path.is_file():
        collect_e2e_sequence_dataset(
            cfg=cfg,
            data_dir=data_dir,
            num_frames=int(args.num_frames),
            start_seed=int(args.start_seed),
            num_scenarios=int(args.num_scenarios),
            use_mock=use_mock,
            simulator=args.simulator,
        )

    # 2. Instantiate full AutonomousDrivingSystem (exact 4,232,991 trainable params)
    system = AutonomousDrivingSystem(config=cfg, use_mock=use_mock).to(device)
    if args.init_ckpt and Path(args.init_ckpt).is_file():
        system.load_adapter_checkpoint(
            args.init_ckpt, map_location=device, strict=False, restore_modes=False
        )
        print(f"[07_train] Warm-started adapter weights from: {args.init_ckpt}", flush=True)

    system.set_policy_modes(
        decision_mode="pure_neural",
        trajectory_mode="obstacle_aware_e2e",
    )
    unlocked = system.enable_temporal_learning(gain=0.08)

    trainable_params = [p for p in system.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    assert n_trainable == 4_232_991, f"Trainable parameter invariant violated: {n_trainable}"
    print(
        f"[07_train] Device={device} | Mock={use_mock} | Trainable={n_trainable:,} (INVARIANT PASS) | "
        f"Modes=({system.laya.decision_mode}, {system.laya.trajectory_mode}) | "
        f"TemporalUnlocked={unlocked}",
        flush=True,
    )

    # 3. Pre-cache frozen DINOv2 patch tokens on GPU for 5x faster multi-epoch training
    patch_cache_path: Optional[Path] = None
    if not args.no_cache_dino and not use_mock:
        patch_cache_path = build_or_load_dino_patch_cache(
            system=system,
            data_dir=data_dir,
            device=device,
            batch_size=max(32, int(args.batch_size) * 2),
            rebuild=bool(args.rebuild_cache),
        )

    train_loader, val_loader, train_ds, val_ds = build_e2e_dataloaders(
        data_dir=data_dir,
        train_split=float(cfg.get("data", {}).get("train_split", 0.85)),
        batch_size=int(args.batch_size),
        num_workers=0,
        seed=seed,
        patch_cache_path=patch_cache_path,
    )
    print(
        f"[07_train] Episode-disjoint split: train={len(train_ds)} frames, val={len(val_ds)} frames",
        flush=True,
    )

    # Give temporal_gate a slightly stronger LR so motion residual activates cleanly
    gate_params = [system.projector.temporal_gate]
    gate_ids = {id(p) for p in gate_params}
    other_params = [p for p in trainable_params if id(p) not in gate_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": float(args.lr), "weight_decay": float(args.weight_decay)},
            {"params": gate_params, "lr": float(args.lr) * 5.0, "weight_decay": 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(args.epochs)), eta_min=float(args.lr) * 0.05
    )

    use_amp = device.type == "cuda"
    best_score = float("inf")
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(args.epochs) + 1):
        ep_t0 = time.perf_counter()
        system.train()
        # Keep frozen DINOv2 and Laya encoder in eval() so dropout is disabled on frozen weights
        system.dino.eval()
        if not use_mock:
            system.laya.laya_model.eval()
        else:
            system.laya.encoder.eval()

        tr_loss = 0.0
        tr_coll = 0.0
        tr_traj = 0.0
        tr_samples = 0

        for batch in train_loader:
            bsz = len(batch["telemetries"])
            optimizer.zero_grad(set_to_none=True)
            out = _forward_e2e_batch(
                system,
                batch,
                device=device,
                use_amp=use_amp,
                prompt_offset_dropout=0.50,
            )
            losses = compute_obstacle_aware_e2e_loss(out, batch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
            optimizer.step()

            tr_loss += float(losses["total"].item()) * bsz
            tr_traj += float(losses["traj"].item()) * bsz
            tr_coll += float(losses["coll"].item()) * bsz
            tr_samples += bsz

        scheduler.step()
        val_metrics = evaluate_e2e_epoch(
            system=system, val_loader=val_loader, device=device, use_amp=use_amp
        )
        ep_sec = time.perf_counter() - ep_t0

        denom = max(1, tr_samples)
        row = {
            "epoch": epoch,
            "lr": round(float(optimizer.param_groups[0]["lr"]), 6),
            "elapsed_s": round(ep_sec, 2),
            "train_loss": round(tr_loss / denom, 5),
            "train_traj": round(tr_traj / denom, 5),
            "train_coll": round(tr_coll / denom, 5),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        # Selection score balances open-loop accuracy, safety collision rate, and pure-neural decision quality
        sel_score = (
            float(val_metrics["ade_3s_m"])
            + 8.0 * float(val_metrics["cr_traj"])
            + 0.25 * float(val_metrics["loss"])
        )
        is_best = sel_score < best_score
        if is_best:
            best_score = sel_score
            system.save_adapter_checkpoint(
                out_ckpt,
                epoch=epoch,
                val_metrics=val_metrics,
                config=cfg,
                training_stage="stage1_obstacle_aware_e2e",
            )

        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] "
            f"tr_loss={row['train_loss']:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"ADE_3s={val_metrics['ade_3s_m']:.3f}m | FDE_3s={val_metrics['fde_3s_m']:.3f}m | "
            f"CR_traj={val_metrics['cr_traj']*100:.2f}% | "
            f"Man/Spd/AEB={val_metrics['man_acc']*100:.1f}/{val_metrics['spd_acc']*100:.1f}/{val_metrics['aeb_acc']*100:.1f}% | "
            f"|gate|={val_metrics['abs_temporal_gate']:.4f} | {ep_sec:.1f}s"
            + (" *BEST*" if is_best else ""),
            flush=True,
        )

    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(out_ckpt),
                "trainable_parameters": n_trainable,
                "train_frames": len(train_ds),
                "val_frames": len(val_ds),
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"[07_train] Saved Stage-1 E2E checkpoint -> {out_ckpt}", flush=True)
    print(f"[07_train] Saved training history -> {hist_path}", flush=True)


if __name__ == "__main__":
    main()
