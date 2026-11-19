"""Stage 2 (Track 3): Closed-Loop Reinforcement Learning Fine-Tuning (PPO / GRPO + KL Anchor).

Trains the Stage-1 obstacle-aware E2E policy (`checkpoints/projector_e2e_stage1.pt`) directly
inside closed-loop 3D MetaDrive (`ActuatorRealism=ON`, `Pure E2E Trajectory + Shield OFF`)
on strictly disjoint training seeds (`1000..1499`), validates on holdout seeds (`4000..4005`,
completely disjoint from unseen test seeds `5000+`), saves the best `4,232,991`-parameter
adapter to `checkpoints/projector_e2e_rl.pt`, and auto-calibrates `checkpoints/temperature_e2e_rl.pt`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_validated_config
from src.control.e2e_trajectory_tracker import PureE2ETrajectoryTracker
from src.env.expert_labeler import IDX_TO_MANEUVER, SPEED_LEVEL_KMH
from src.env.metadrive_wrapper import MetaDriveWrapper
from src.models.laya_pilot import AutonomousDrivingSystem
from src.rl.closed_loop_rl import (
    AdapterRLPolicyWrapper,
    CachedFeatureRolloutBuffer,
    ClosedLoopRLTrainer,
    FrozenReferenceAdapter,
)
from src.rl.rewards import ClosedLoopDrivingReward
from src.sim.e2e_dataset import build_e2e_dataloaders
from src.sim.metrics import EpisodeSafetyComfortTracker
from src.sim.scenarios import RealisticSimEnvWrapper
from src.utils.calibration import TemperatureScaler
from src.utils.seed import seed_everything

EARLY_CURRICULUM_SCENARIOS: Tuple[str, ...] = (
    "nominal_urban_cruise",
    "nominal_urban_cruise",
    "sudden_lead_ebrake",
)

FULL_CURRICULUM_SCENARIOS: Tuple[str, ...] = (
    "nominal_urban_cruise",
    "sudden_lead_ebrake",
    "aggressive_cut_in",
    "occluded_jaywalker_dartout",
    "blindspot_highspeed_overtaker",
    "sotif_heavy_rain_glare_dropout",
)

VAL_SCENARIOS: Tuple[str, ...] = (
    "nominal_urban_cruise",
    "sudden_lead_ebrake",
    "aggressive_cut_in",
    "occluded_jaywalker_dartout",
    "blindspot_highspeed_overtaker",
    "sotif_heavy_rain_glare_dropout",
)


def _resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def _img_to_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )


def _build_decision_dict(act_out: Dict[str, Any]) -> Dict[str, Any]:
    m_idx = int(act_out["man_act"][0].item())
    s_lvl = int(act_out["spd_act"][0].item())
    aeb_prob = float(act_out["aeb_prob"][0].item())
    traj = act_out["exec_traj"][0].cpu().tolist()
    return {
        "maneuver": IDX_TO_MANEUVER[min(3, max(0, m_idx))],
        "maneuver_idx": m_idx,
        "maneuver_confidence": 0.9,
        "speed_level": s_lvl,
        "target_speed_kmh": SPEED_LEVEL_KMH.get(s_lvl, 40.0),
        "speed_confidence": 0.9,
        "emergency_brake": aeb_prob,
        "predicted_trajectory": traj,
    }


@torch.no_grad()
def evaluate_holdout_pure_e2e(
    policy: AdapterRLPolicyWrapper,
    env: RealisticSimEnvWrapper,
    val_seeds: Sequence[int],
    device: torch.device,
    max_steps: int = 260,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Evaluate policy deterministically on holdout seeds (`4000..4005`) with Pure E2E + Shield OFF."""
    policy.eval()
    tracker = PureE2ETrajectoryTracker()
    crash_free = 0
    arrived = 0
    driving_scores: List[float] = []
    progresses: List[float] = []
    rewards_sum: List[float] = []

    for i, seed in enumerate(val_seeds):
        sc = VAL_SCENARIOS[i % len(VAL_SCENARIOS)]
        env.set_scenario(sc)
        rgb, tel, info = env.reset(seed=int(seed))
        tracker.reset()
        rew_fn = ClosedLoopDrivingReward(dt=env.step_dt)
        sc_tracker = EpisodeSafetyComfortTracker(dt=env.step_dt)
        prev_vis: Optional[torch.Tensor] = None

        ep_ret = 0.0
        crashed = False
        ep_arrived = False
        last_comp = 0.0

        for _step in range(max_steps):
            img_t = _img_to_tensor(rgb, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                patch = policy.system.dino(img_t)
            act_out = policy.act(
                patch_tokens=patch,
                telemetry=tel,
                prev_visual_tokens=prev_vis,
                deterministic=True,
                use_amp=use_amp,
            )
            prev_vis = act_out["visual_tokens"]

            dec = _build_decision_dict(act_out)
            (steer, tb), gated = tracker.compute_control(
                dec, speed_kmh=float(tel.get("speed_kmh", 0.0)), dt=env.step_dt
            )
            setattr(env, "_last_gated_maneuver", str(dec["maneuver"]))
            setattr(env.base_env, "_last_gated_maneuver", str(dec["maneuver"]))

            rgb, tel, _env_r, done, info = env.step((steer, tb))
            r_step, _ = rew_fn.compute(tel, info, dt=env.step_dt)
            ep_ret += r_step
            sc_tracker.step(tel, (steer, tb), gated_info=gated, extra_info=info, dt=env.step_dt)

            last_comp = float(info.get("route_completion", last_comp))
            if bool(info.get("crash", False)) or bool(info.get("out_of_road", False)):
                crashed = True
                break
            if bool(info.get("arrive_dest", False)):
                ep_arrived = True
                last_comp = 1.0
                break
            if done:
                break

        summary = sc_tracker.summarize(
            route_completion=last_comp, arrived=ep_arrived, crashed=crashed
        )
        if not crashed:
            crash_free += 1
        if ep_arrived:
            arrived += 1
        driving_scores.append(float(summary["driving_score_pct"]))
        progresses.append(float(summary["route_progress_pct"]))
        rewards_sum.append(float(ep_ret))

    n = max(1, len(val_seeds))
    return {
        "val_crash_free_pct": round(100.0 * crash_free / n, 2),
        "val_arrival_pct": round(100.0 * arrived / n, 2),
        "val_driving_score_pct": round(float(np.mean(driving_scores)), 2),
        "val_progress_pct": round(float(np.mean(progresses)), 2),
        "val_mean_return": round(float(np.mean(rewards_sum)), 2),
    }


@torch.no_grad()
def calibrate_e2e_rl_temperatures(
    system: AutonomousDrivingSystem,
    data_dir: Path,
    output_temp_ckpt: Path,
    device: torch.device,
    seed: int = 42,
    use_amp: bool = False,
) -> Optional[Dict[str, Any]]:
    """Auto-calibrate `pure_neural` decision temperatures on `data_e2e` validation split."""
    meta_path = data_dir / "metadata.jsonl"
    if not meta_path.is_file():
        print(
            f"[08_rl_cal] Skipping calibration: {meta_path} not found.",
            flush=True,
        )
        return None

    cache_path = data_dir / "dino_patch_cache.fp16.npy"
    _, val_loader, _, _ = build_e2e_dataloaders(
        data_dir=data_dir,
        train_split=0.85,
        batch_size=32,
        num_workers=0,
        seed=seed,
        patch_cache_path=cache_path if cache_path.is_file() else None,
    )

    system.eval()
    system.set_policy_modes(
        decision_mode="pure_neural",
        trajectory_mode="obstacle_aware_e2e",
    )

    m_logits_l: List[torch.Tensor] = []
    s_logits_l: List[torch.Tensor] = []
    a_logits_l: List[torch.Tensor] = []
    m_tgt_l: List[torch.Tensor] = []
    s_tgt_l: List[torch.Tensor] = []
    a_tgt_l: List[torch.Tensor] = []

    for batch in val_loader:
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
                aux_use_prior=False,
            )
        m_logits_l.append(out["maneuver_logits"].float())
        s_logits_l.append(out["speed_logits"].float())
        a_logits_l.append(out["aeb_logits"].float())
        m_tgt_l.append(batch["maneuver_idx"].to(device))
        s_tgt_l.append(batch["speed_level"].to(device))
        a_tgt_l.append(batch["emergency_brake"].to(device))

    scaler = TemperatureScaler().to(device)
    before_m, after_m = scaler.fit(
        maneuver_logits=torch.cat(m_logits_l, dim=0),
        maneuver_targets=torch.cat(m_tgt_l, dim=0),
        speed_logits=torch.cat(s_logits_l, dim=0),
        speed_targets=torch.cat(s_tgt_l, dim=0),
        aeb_logits=torch.cat(a_logits_l, dim=0),
        aeb_targets=torch.cat(a_tgt_l, dim=0),
        num_bins=15,
    )
    temps = scaler.get_temperatures()
    system.laya.set_temperatures(
        temps["t_maneuver"],
        temps["t_speed"],
        temps["t_aeb"],
        temps.get("b_aeb", 0.0),
    )

    output_temp_ckpt.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = output_temp_ckpt.with_suffix(output_temp_ckpt.suffix + ".tmp")
    torch.save(
        {
            "temperatures": temps,
            "before_metrics": before_m,
            "after_metrics": after_m,
        },
        tmp_p,
    )
    os.replace(tmp_p, output_temp_ckpt)
    print(
        f"[08_rl_cal] Calibrated temperatures -> {output_temp_ckpt} | "
        f"Mean ECE: {before_m['mean_ece']:.4f} -> {after_m['mean_ece']:.4f}",
        flush=True,
    )
    return {"temperatures": temps, "before_metrics": before_m, "after_metrics": after_m}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 (Track 3): Closed-Loop RL Fine-Tuning (PPO / GRPO + IL-KL Anchor)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument("--algo", type=str, default="ppo", choices=("ppo", "grpo"))
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "projector_e2e_stage1.pt"),
    )
    parser.add_argument(
        "--output-ckpt",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "projector_e2e_rl.pt"),
    )
    parser.add_argument(
        "--temp-ckpt",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "temperature_e2e_rl.pt"),
    )
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data_e2e"))
    parser.add_argument("--total-steps", type=int, default=24000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--grpo-group-size", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--lr-actor", type=float, default=3e-5)
    parser.add_argument("--lr-critic", type=float, default=1e-4)
    parser.add_argument("--beta-kl", type=float, default=0.04)
    parser.add_argument("--eval-interval", type=int, default=3)
    parser.add_argument("--max-ep-steps", type=int, default=260)
    parser.add_argument(
        "--history-json",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "stage2_rl_train_history.json"),
    )
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    cfg = load_validated_config(args.config).to_dict()
    seed = int(cfg.get("project", {}).get("seed", 42))
    seed_everything(seed)
    rng = np.random.default_rng(seed + 1337)

    use_mock = bool(args.mock or cfg.get("project", {}).get("use_mock", False))
    device = _resolve_device(str(cfg.get("project", {}).get("device", "auto")))
    use_amp = device.type == "cuda"

    out_ckpt = Path(args.output_ckpt)
    temp_ckpt = Path(args.temp_ckpt)
    hist_path = Path(args.history_json)
    hist_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Stage-1 Obstacle-Aware E2E model & verify exact 4,232,991 trainable invariant
    system = AutonomousDrivingSystem(config=cfg, use_mock=use_mock).to(device)
    init_path = Path(args.init_ckpt)
    if init_path.is_file():
        system.load_adapter_checkpoint(init_path, map_location=device, strict=False, restore_modes=False)
        print(f"[08_rl] Loaded Stage-1 E2E checkpoint: {init_path}", flush=True)
    else:
        print(
            f"[08_rl] WARNING: Stage-1 checkpoint {init_path} not found; initializing fresh adapter.",
            flush=True,
        )
    system.set_policy_modes(
        decision_mode="pure_neural",
        trajectory_mode="obstacle_aware_e2e",
    )
    system.enable_temporal_learning(gain=0.08)

    n_trainable = sum(p.numel() for p in system.parameters() if p.requires_grad)
    assert n_trainable == 4_232_991, f"Parameter invariant violated: {n_trainable}"

    # Lightweight frozen Stage-1 KL reference snapshot (~16 MB)
    ref_adapter = FrozenReferenceAdapter(system).to(device)
    policy = AdapterRLPolicyWrapper(system=system).to(device)
    trainer = ClosedLoopRLTrainer(
        policy=policy,
        algo=args.algo,
        lr_actor=float(args.lr_actor),
        lr_critic=float(args.lr_critic),
        beta_kl=float(args.beta_kl),
        device=device,
    )

    print(
        f"[08_rl] Algo={args.algo.upper()} | Device={device} | Mock={use_mock} | "
        f"SystemTrainable={n_trainable:,} (INVARIANT PASS) | "
        f"TotalSteps={args.total_steps} | RolloutSteps={args.rollout_steps}",
        flush=True,
    )

    # 2. Build MetaDrive 3D env covering train seeds 1000..1499 and holdout val seeds 4000..4005
    #    Note: MetaDrive wraps seeds into [start_seed, start_seed + num_scenarios), so we set
    #    start_seed=1000, num_scenarios=3010 (covers 1000..4009; unseen eval 5000+ strictly outside!)
    rl_cfg = json.loads(json.dumps(cfg))
    rl_cfg.setdefault("env", {})["start_seed"] = 1000
    rl_cfg["env"]["max_episode_steps"] = int(args.max_ep_steps)
    rl_cfg["env"]["traffic_density"] = float(rl_cfg["env"].get("traffic_density", 0.12))

    base_env = MetaDriveWrapper(
        config=rl_cfg,
        use_mock=use_mock,
        start_seed=1000,
        num_scenarios=3010,
    )
    env = RealisticSimEnvWrapper(
        base_env=base_env,
        scenario="nominal_urban_cruise",
        actuator_realism=True,
        localization_mode="hd_map",
    )

    val_seeds = (4000, 4001, 4002, 4003, 4004, 4005)
    tracker = PureE2ETrajectoryTracker()
    buffer = CachedFeatureRolloutBuffer()
    history: List[Dict[str, Any]] = []

    try:
        # Baseline holdout validation before RL (Stage-1 E2E, Pure E2E + Shield OFF)
        init_val = evaluate_holdout_pure_e2e(
            policy=policy,
            env=env,
            val_seeds=val_seeds,
            device=device,
            max_steps=int(args.max_ep_steps),
            use_amp=use_amp,
        )
        best_val_score = (
            float(init_val["val_driving_score_pct"])
            + 0.4 * float(init_val["val_crash_free_pct"])
        )
        # Always write initial baseline checkpoint so output file exists even if early steps regress
        system.save_adapter_checkpoint(
            out_ckpt,
            epoch=0,
            val_metrics=init_val,
            config=cfg,
            training_stage=f"stage2_rl_{args.algo}",
        )
        print(
            f"[08_rl] Pre-RL Holdout (4000..4005, Shield OFF): "
            f"CrashFree={init_val['val_crash_free_pct']:.1f}% | "
            f"DrivingScore={init_val['val_driving_score_pct']:.1f}% | "
            f"Progress={init_val['val_progress_pct']:.1f}% | "
            f"Return={init_val['val_mean_return']:.2f}",
            flush=True,
        )
        history.append({"update": 0, "steps": 0, **init_val})

        total_steps_done = 0
        update_idx = 0
        ep_counter = 0
        group_counter = 0
        t0_all = time.perf_counter()

        while total_steps_done < int(args.total_steps):
            update_idx += 1
            upd_t0 = time.perf_counter()
            buffer.clear()
            policy.eval()

            frac_done = total_steps_done / max(1.0, float(args.total_steps))
            cur_scenarios = (
                EARLY_CURRICULUM_SCENARIOS if frac_done < 0.40 else FULL_CURRICULUM_SCENARIOS
            )

            steps_in_rollout = 0
            rollout_ep_returns: List[float] = []
            rollout_crash_free = 0
            rollout_eps = 0
            last_val_est = 0.0

            while steps_in_rollout < int(args.rollout_steps):
                # Train seeds strictly in [1000..1499]
                train_seed = int(rng.integers(1000, 1500))
                sc_name = str(cur_scenarios[int(rng.integers(0, len(cur_scenarios)))])
                repeats = int(args.grpo_group_size) if args.algo == "grpo" else 1
                group_counter += 1

                for _rep in range(repeats):
                    if steps_in_rollout >= int(args.rollout_steps) and args.algo == "ppo":
                        break
                    ep_counter += 1
                    env.set_scenario(sc_name)
                    rgb, tel, info = env.reset(seed=train_seed)
                    tracker.reset()
                    rew_fn = ClosedLoopDrivingReward(dt=env.step_dt)

                    prev_vis_pol: Optional[torch.Tensor] = None
                    ep_ret = 0.0
                    crashed = False

                    for step_i in range(int(args.max_ep_steps)):
                        img_t = _img_to_tensor(rgb, device)
                        with torch.no_grad():
                            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                                patch = system.dino(img_t)
                            act_out = policy.act(
                                patch_tokens=patch,
                                telemetry=tel,
                                prev_visual_tokens=prev_vis_pol,
                                deterministic=False,
                                use_amp=use_amp,
                            )

                        prev_vis_pol = act_out["visual_tokens"]

                        dec = _build_decision_dict(act_out)
                        (steer, tb), _gated = tracker.compute_control(
                            dec, speed_kmh=float(tel.get("speed_kmh", 0.0)), dt=env.step_dt
                        )
                        setattr(env, "_last_gated_maneuver", str(dec["maneuver"]))
                        setattr(env.base_env, "_last_gated_maneuver", str(dec["maneuver"]))

                        next_rgb, next_tel, _env_r, done, next_info = env.step((steer, tb))
                        r_step, _ = rew_fn.compute(next_tel, next_info, dt=env.step_dt)
                        ep_ret += r_step

                        if bool(next_info.get("crash", False)) or bool(
                            next_info.get("out_of_road", False)
                        ):
                            crashed = True
                            done = True

                        step_budget_hit = (
                            args.algo == "ppo"
                            and (steps_in_rollout + 1) >= int(args.rollout_steps)
                        )
                        ep_done = bool(done or (step_i + 1 >= int(args.max_ep_steps)))

                        buffer.add(
                            patch_tokens=patch,
                            is_episode_start=(step_i == 0),
                            telemetry=tel,
                            traj_act=act_out["traj_act"],
                            man_act=act_out["man_act"],
                            spd_act=act_out["spd_act"],
                            aeb_act=act_out["aeb_act"],
                            old_log_prob=float(act_out["log_prob"][0].item()),
                            value=float(act_out["value"][0].item()),
                            reward=r_step,
                            done=ep_done,
                            ref_traj_mean=act_out["traj_mean"],
                            ref_man_logits=act_out["man_logits"],
                            ref_spd_logits=act_out["spd_logits"],
                            ref_aeb_logits=act_out["aeb_logits"],
                            episode_id=ep_counter,
                            group_id=group_counter,
                        )

                        steps_in_rollout += 1
                        total_steps_done += 1
                        rgb, tel, info = next_rgb, next_tel, next_info

                        if ep_done:
                            last_val_est = 0.0
                            break
                        if step_budget_hit:
                            # Bootstrap value for truncated rollout boundary
                            with torch.no_grad():
                                next_img_t = _img_to_tensor(rgb, device)
                                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                                    next_p = system.dino(next_img_t)
                                next_act = policy.act(
                                    patch_tokens=next_p,
                                    telemetry=tel,
                                    prev_visual_tokens=prev_vis_pol,
                                    deterministic=True,
                                    use_amp=use_amp,
                                )
                                last_val_est = float(next_act["value"][0].item())
                            break

                    rollout_ep_returns.append(ep_ret)
                    rollout_eps += 1
                    if not crashed:
                        rollout_crash_free += 1

            # Compute exact Stage-1 KL reference logits in fast GPU batches (avoids per-step overhead)
            ref_adapter.populate_buffer_refs(
                buffer=buffer,
                shared_system=system,
                device=device,
                batch_size=max(32, int(args.minibatch_size)),
                use_amp=use_amp,
            )

            # Perform PPO or GRPO policy update using cached FP16 DINOv2 features
            upd_stats = trainer.update(
                buffer=buffer,
                epochs=int(args.update_epochs),
                minibatch_size=int(args.minibatch_size),
                last_value=last_val_est,
                use_amp=use_amp,
            )

            upd_sec = time.perf_counter() - upd_t0
            mean_ret = float(np.mean(rollout_ep_returns)) if rollout_ep_returns else 0.0
            cf_pct = 100.0 * rollout_crash_free / max(1, rollout_eps)
            lat_std = float(policy.get_traj_std(device)[0, :, 1].mean().item())

            log_row: Dict[str, Any] = {
                "update": update_idx,
                "steps": total_steps_done,
                "elapsed_s": round(time.perf_counter() - t0_all, 1),
                "upd_s": round(upd_sec, 1),
                "train_mean_return": round(mean_ret, 2),
                "train_crash_free_pct": round(cf_pct, 1),
                "lat_std_m": round(lat_std, 4),
                **upd_stats,
            }

            is_eval = (
                update_idx % max(1, int(args.eval_interval)) == 0
                or total_steps_done >= int(args.total_steps)
            )
            best_tag = ""
            if is_eval:
                val_m = evaluate_holdout_pure_e2e(
                    policy=policy,
                    env=env,
                    val_seeds=val_seeds,
                    device=device,
                    max_steps=int(args.max_ep_steps),
                    use_amp=use_amp,
                )
                log_row.update(val_m)
                val_score = (
                    float(val_m["val_driving_score_pct"])
                    + 0.4 * float(val_m["val_crash_free_pct"])
                )
                if val_score >= best_val_score:
                    best_val_score = val_score
                    system.save_adapter_checkpoint(
                        out_ckpt,
                        epoch=update_idx,
                        val_metrics=val_m,
                        config=cfg,
                        training_stage=f"stage2_rl_{args.algo}",
                    )
                    best_tag = " *BEST*"

            history.append(log_row)
            val_str = (
                f" | VAL_CF={log_row['val_crash_free_pct']:.1f}% VAL_DS={log_row['val_driving_score_pct']:.1f}%"
                if is_eval
                else ""
            )
            print(
                f"[Upd {update_idx:02d} | {total_steps_done:5d}/{args.total_steps}] "
                f"ret={mean_ret:6.1f} | cf={cf_pct:5.1f}% | "
                f"pi_loss={upd_stats['policy_loss']:+.4f} | clip={upd_stats['clip_frac']:.2f} | "
                f"kl={upd_stats['kl_div']:.4f} | beta={upd_stats['beta_kl']:.3f} | "
                f"std_y={lat_std:.3f}m | {upd_sec:.1f}s"
                f"{val_str}{best_tag}",
                flush=True,
            )
    finally:
        env.close()

    # Reload best RL checkpoint and run post-hoc temperature calibration
    if out_ckpt.is_file():
        system.load_adapter_checkpoint(out_ckpt, map_location=device, strict=False)
    cal_report = calibrate_e2e_rl_temperatures(
        system=system,
        data_dir=Path(args.data_dir),
        output_temp_ckpt=temp_ckpt,
        device=device,
        seed=seed,
        use_amp=use_amp,
    )

    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "algo": args.algo,
                "init_ckpt": str(init_path),
                "output_ckpt": str(out_ckpt),
                "temperature_ckpt": str(temp_ckpt),
                "trainable_parameters": n_trainable,
                "calibration": cal_report,
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"[08_rl] Saved best Stage-2 RL checkpoint -> {out_ckpt}", flush=True)
    print(f"[08_rl] Saved RL training history -> {hist_path}", flush=True)


if __name__ == "__main__":
    main()
