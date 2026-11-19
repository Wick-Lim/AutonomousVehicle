"""Stage 3 (Track 1-B): Head-to-Head 2x2 Closed-Loop, Open-Loop, Counterfactual & NAVSIM PDMS Benchmark.

Evaluates:
  - Candidates:
      1. `v12_il_baseline` (`checkpoints/projector_best.pt`, `hybrid_prior` + `residual_regularizer`)
      2. `stage1_e2e`      (`checkpoints/projector_e2e_stage1.pt`, `pure_neural` + `obstacle_aware_e2e`)
      3. `stage2_e2e_rl`   (`checkpoints/projector_e2e_rl.pt`, `pure_neural` + `obstacle_aware_e2e`)
  - Control Regimes (2x2 Matrix):
      - `pure_e2e_shield_off`: `PureE2ETrajectoryTracker` (Zero HD-map PID, Zero SafetyShield)
      - `hybrid_shield_on`:    `SafetyShieldPID` (`shield_enabled=True`)
  - Simulators (strictly sequential — never concurrent):
      - Unseen MetaDrive 3D seeds (`5000..5011`, 6 scenarios x 2 seeds)
      - Live CARLA 0.9.16 cross-town (`Town10HD_Opt`, `Town03_Opt`, `Town04_Opt`)
        + `DINOv2LayaLeaderboardAgent` (CARLA Leaderboard 2.0 / Bench2Drive) API verification
  - Metrics:
      - Open-Loop Isolated Decoder (`ADE_3s`, `FDE_3s`, `CR_traj`, `ECE`, `|temporal_gate|`, `Latency ms`)
      - Counterfactual Obstacle Lateral Shift Sensitivity (`Delta y_obs in {-1, 0, +1} m`)
      - Closed-Loop `Driving Score`, official `NAVSIM PDMS` (`NC`, `DAC`, `EP`, `TTC`, `C`),
        `Crash-Free %`, `Arrival %`, `MinTTC`, `ISO Comfort %`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.leaderboard_agent import DINOv2LayaLeaderboardAgent
from src.benchmarks.pdms_evaluator import (
    compute_pdms_from_episode_summary,
    evaluate_counterfactual_obstacle_sensitivity,
    evaluate_isolated_decoder,
)
from src.config import load_validated_config
from src.control.e2e_trajectory_tracker import PureE2ETrajectoryTracker
from src.control.shield_pid import SafetyShieldPID
from src.env.expert_labeler import IDX_TO_MANEUVER, SPEED_LEVEL_KMH
from src.env.metadrive_wrapper import MetaDriveWrapper
from src.models.laya_pilot import AutonomousDrivingSystem
from src.sim.carla_launcher import get_carla_server_manager
from src.sim.e2e_dataset import build_e2e_dataloaders
from src.sim.metrics import EpisodeSafetyComfortTracker
from src.sim.scenarios import RealisticSimEnvWrapper
from src.utils.seed import seed_everything

EVAL_SCENARIOS: Tuple[str, ...] = (
    "nominal_urban_cruise",
    "aggressive_cut_in",
    "sudden_lead_ebrake",
    "blindspot_highspeed_overtaker",
    "occluded_jaywalker_dartout",
    "sotif_heavy_rain_glare_dropout",
)

CARLA_EVAL_TOWNS: Tuple[str, ...] = (
    "Town10HD_Opt",
    "Town03_Opt",
    "Town04_Opt",
)


def _resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def _load_candidate(
    system: AutonomousDrivingSystem,
    spec: Dict[str, Any],
    device: torch.device,
) -> bool:
    """Load candidate checkpoint & modes into shared `AutonomousDrivingSystem`."""
    ckpt_path = Path(spec["ckpt"])
    if not ckpt_path.is_file():
        return False
    system.load_adapter_checkpoint(
        ckpt_path, map_location=device, strict=False, restore_modes=False
    )
    system.set_policy_modes(
        decision_mode=spec["decision_mode"],
        trajectory_mode=spec["trajectory_mode"],
    )
    # Reset temperatures to neutral 1.0 unless a matching temperature checkpoint exists
    system.laya.set_temperatures(1.0, 1.0, 1.0, 0.0)
    temp_path = Path(spec["temp_ckpt"]) if spec.get("temp_ckpt") else None
    if temp_path is not None and temp_path.is_file():
        system.load_temperature_checkpoint(temp_path, map_location=device)
    system.reset_temporal_buffer()
    system.eval()
    return True


def _load_val_sample_frames(data_dir: Path, max_frames: int = 24) -> List[np.ndarray]:
    """Load up to `max_frames` real validation RGB frames for counterfactual obstacle shift testing."""
    meta_path = data_dir / "metadata.jsonl"
    if not meta_path.is_file():
        return []
    recs: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not recs:
        return []
    # Sample from last 15% (disjoint tail episodes)
    tail = recs[int(len(recs) * 0.85) :] or recs
    step = max(1, len(tail) // max_frames)
    frames: List[np.ndarray] = []
    for rec in tail[::step][:max_frames]:
        p = data_dir / rec["image_path"]
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (196, 392):
                rgb = cv2.resize(rgb, (392, 196), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
    return frames


@torch.no_grad()
def run_closed_loop_episodes(
    system: AutonomousDrivingSystem,
    env: RealisticSimEnvWrapper,
    cfg: Dict[str, Any],
    seeds: Sequence[int],
    scenarios: Sequence[str],
    control_regime: str,
    device: torch.device,
    max_steps: int = 260,
    use_mock: bool = False,
) -> Dict[str, Any]:
    """Run closed-loop episodes on unseen seeds and compute Driving Score + NAVSIM PDMS."""
    system.eval()
    e2e_tracker = PureE2ETrajectoryTracker()
    shield_ctrl = SafetyShieldPID(config=cfg, use_mock=use_mock)
    use_pure_e2e = control_regime == "pure_e2e_shield_off"
    aux_use_prior = not (
        system.laya.decision_mode == "pure_neural"
        and system.laya.trajectory_mode == "obstacle_aware_e2e"
    )

    ep_rows: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []

    for idx, seed in enumerate(seeds):
        sc_name = scenarios[idx % len(scenarios)]
        env.set_scenario(sc_name)
        rgb, tel, info = env.reset(seed=int(seed))

        system.reset_temporal_buffer()
        e2e_tracker.reset()
        shield_ctrl.reset()
        sc_tracker = EpisodeSafetyComfortTracker(dt=env.step_dt)

        crashed = False
        oor = False
        arrived = False
        last_comp = 0.0

        for _step in range(max_steps):
            t0 = time.perf_counter()
            img_t = (
                torch.from_numpy(np.ascontiguousarray(rgb))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
                / 255.0
            )
            out = system(img_t, [tel], aux_use_prior=aux_use_prior)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            m_idx = int(out["maneuver_idx"][0].item())
            s_lvl = int(out["speed_level"][0].item())
            decision = {
                "maneuver": IDX_TO_MANEUVER[min(3, max(0, m_idx))],
                "maneuver_idx": m_idx,
                "maneuver_confidence": float(out["maneuver_confidence"][0].item()),
                "speed_level": s_lvl,
                "target_speed_kmh": SPEED_LEVEL_KMH.get(s_lvl, 40.0),
                "speed_confidence": float(out["speed_confidence"][0].item()),
                "emergency_brake": float(out["emergency_brake"][0].item()),
                "predicted_trajectory": out["predicted_trajectory"][0].detach().cpu().tolist(),
            }

            if use_pure_e2e:
                (steer, tb), gated = e2e_tracker.compute_control(
                    decision,
                    speed_kmh=float(tel.get("speed_kmh", 0.0)),
                    dt=env.step_dt,
                )
            else:
                (steer, tb), gated = shield_ctrl.compute_control(
                    decision, tel, dt=env.step_dt, shield_enabled=True
                )

            g_man = str(gated.get("maneuver", decision["maneuver"]))
            setattr(env, "_last_gated_maneuver", g_man)
            setattr(env.base_env, "_last_gated_maneuver", g_man)

            rgb, tel, _rew, done, info = env.step((steer, tb))
            sc_tracker.step(tel, (steer, tb), gated_info=gated, extra_info=info, dt=env.step_dt)

            last_comp = float(info.get("route_completion", last_comp))
            if bool(info.get("out_of_road", False)):
                oor = True
                crashed = True
                break
            if bool(info.get("crash", False)):
                crashed = True
                break
            if bool(info.get("arrive_dest", False)):
                arrived = True
                last_comp = 1.0
                break
            if done:
                break

        summary = sc_tracker.summarize(
            route_completion=last_comp, arrived=arrived, crashed=crashed
        )
        pdms = compute_pdms_from_episode_summary(
            summary=summary, crashed=crashed, out_of_road=oor
        )
        ep_rows.append(
            {
                "seed": int(seed),
                "scenario": sc_name,
                "crashed": crashed,
                "out_of_road": oor,
                "arrived": arrived,
                "route_progress_pct": summary["route_progress_pct"],
                "driving_score_pct": summary["driving_score_pct"],
                "min_ttc_s": summary["min_ttc_s"],
                "comfort_rate_pct": summary.get("iso_comfort_rate_pct", 100.0),
                **pdms,
            }
        )

    n = max(1, len(ep_rows))
    finite_ttcs = [r["min_ttc_s"] for r in ep_rows if r["min_ttc_s"] < 90.0]
    return {
        "episodes": len(ep_rows),
        "control_regime": control_regime,
        "crash_free_pct": round(100.0 * sum(1 for r in ep_rows if not r["crashed"]) / n, 2),
        "arrival_pct": round(100.0 * sum(1 for r in ep_rows if r["arrived"]) / n, 2),
        "mean_progress_pct": round(float(np.mean([r["route_progress_pct"] for r in ep_rows])), 2),
        "driving_score_pct": round(float(np.mean([r["driving_score_pct"] for r in ep_rows])), 2),
        "pdms": round(float(np.mean([r["pdms"] for r in ep_rows])), 4),
        "pdms_pct": round(float(np.mean([r["pdms_pct"] for r in ep_rows])), 2),
        "pdms_nc": round(float(np.mean([r["nc"] for r in ep_rows])), 4),
        "pdms_dac": round(float(np.mean([r["dac"] for r in ep_rows])), 4),
        "pdms_ep": round(float(np.mean([r["ep"] for r in ep_rows])), 4),
        "pdms_ttc": round(float(np.mean([r["ttc"] for r in ep_rows])), 4),
        "pdms_comfort": round(float(np.mean([r["comfort"] for r in ep_rows])), 4),
        "min_ttc_s": round(float(min(finite_ttcs)) if finite_ttcs else 99.0, 2),
        "iso_comfort_pct": round(float(np.mean([r["comfort_rate_pct"] for r in ep_rows])), 2),
        "mean_latency_ms": round(float(np.mean(latencies_ms)) if latencies_ms else 0.0, 2),
        "episode_details": ep_rows,
    }


def verify_leaderboard_agent_live(
    ckpt_path: Path,
    temp_ckpt_path: Optional[Path],
    use_mock: bool = False,
) -> Dict[str, Any]:
    """Verify CARLA Leaderboard 2.0 / Bench2Drive `DINOv2LayaLeaderboardAgent` contract end-to-end."""
    tmp_cfg = PROJECT_ROOT / "logs" / "_tmp_leaderboard_agent_cfg.json"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg.write_text(
        json.dumps(
            {
                "projector_ckpt": str(ckpt_path),
                "temperature_ckpt": str(temp_ckpt_path) if temp_ckpt_path and temp_ckpt_path.is_file() else None,
                "control_mode": "pure_e2e_trajectory",
                "decision_mode": "pure_neural",
                "trajectory_mode": "obstacle_aware_e2e",
                "use_mock": use_mock,
            }
        ),
        encoding="utf-8",
    )
    agent = DINOv2LayaLeaderboardAgent()
    try:
        agent.setup(str(tmp_cfg))
        sensors = agent.sensors()
        agent.set_global_plan(
            [({"lat": 0.0, "lon": 0.0}, "LANEFOLLOW"), ({"lat": 0.001, "lon": 0.0}, "LEFT")],
            [],
        )
        dummy_bgra = np.zeros((196, 392, 4), dtype=np.uint8)
        ctrl = agent.run_step(
            {
                "rgb_front": (0, dummy_bgra),
                "speed": (0, {"speed": 9.5}),
                "imu": (0, [0.0, 0.0, 9.81, 0.0, 0.0, 0.0, 0.0]),
                "gps": (0, [0.0, 0.0, 0.0]),
                "radar_front": (0, np.zeros((0, 4), dtype=np.float32)),
            },
            timestamp=0.05,
        )
        return {
            "status": "ok",
            "entry_point": "DINOv2LayaLeaderboardAgent",
            "num_sensors": len(sensors),
            "control_type": type(ctrl).__name__,
            "sample_control": {
                "steer": round(float(ctrl.steer), 4),
                "throttle": round(float(ctrl.throttle), 4),
                "brake": round(float(ctrl.brake), 4),
            },
        }
    finally:
        agent.destroy()
        if tmp_cfg.exists():
            tmp_cfg.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 (Track 1-B): 2x2 Closed-Loop, Open-Loop, Counterfactual & NAVSIM PDMS Benchmark."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--simulator",
        type=str,
        default="both",
        choices=("metadrive", "carla", "both"),
    )
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data_e2e"))
    parser.add_argument("--v12-ckpt", type=str, default=str(PROJECT_ROOT / "checkpoints" / "projector_best.pt"))
    parser.add_argument("--v12-temp", type=str, default=str(PROJECT_ROOT / "checkpoints" / "temperature.pt"))
    parser.add_argument("--stage1-ckpt", type=str, default=str(PROJECT_ROOT / "checkpoints" / "projector_e2e_stage1.pt"))
    parser.add_argument("--stage2-ckpt", type=str, default=str(PROJECT_ROOT / "checkpoints" / "projector_e2e_rl.pt"))
    parser.add_argument("--stage2-temp", type=str, default=str(PROJECT_ROOT / "checkpoints" / "temperature_e2e_rl.pt"))
    parser.add_argument("--metadrive-episodes", type=int, default=12)
    parser.add_argument("--carla-towns", type=str, default="Town10HD_Opt,Town03_Opt,Town04_Opt")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--auto-launch-carla", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "e2e_rl_benchmark_report.json"),
    )
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    cfg = load_validated_config(args.config).to_dict()
    seed = int(cfg.get("project", {}).get("seed", 42))
    seed_everything(seed)

    use_mock = bool(args.mock or cfg.get("project", {}).get("use_mock", False))
    device = _resolve_device(str(cfg.get("project", {}).get("device", "auto")))
    use_amp = device.type == "cuda"
    data_dir = Path(args.data_dir)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    candidates: List[Dict[str, Any]] = [
        {
            "name": "v12_il_baseline",
            "ckpt": args.v12_ckpt,
            "temp_ckpt": args.v12_temp,
            "decision_mode": "hybrid_prior",
            "trajectory_mode": "residual_regularizer",
        },
        {
            "name": "stage1_e2e",
            "ckpt": args.stage1_ckpt,
            "temp_ckpt": None,
            "decision_mode": "pure_neural",
            "trajectory_mode": "obstacle_aware_e2e",
        },
        {
            "name": "stage2_e2e_rl",
            "ckpt": args.stage2_ckpt,
            "temp_ckpt": args.stage2_temp,
            "decision_mode": "pure_neural",
            "trajectory_mode": "obstacle_aware_e2e",
        },
    ]

    system = AutonomousDrivingSystem(config=cfg, use_mock=use_mock).to(device)
    n_trainable = sum(p.numel() for p in system.parameters() if p.requires_grad)

    report: Dict[str, Any] = {
        "device": str(device),
        "use_mock": use_mock,
        "trainable_parameters": n_trainable,
        "parameter_invariant_ok": n_trainable == 4_232_991,
        "isolated_decoder": {},
        "counterfactual_sensitivity": {},
        "metadrive_unseen_5000": {},
        "carla_cross_town": {},
        "leaderboard_agent_verification": {},
    }

    # --- PART A: Isolated Decoder Open-Loop & Counterfactual Obstacle Shift ---
    meta_path = data_dir / "metadata.jsonl"
    if meta_path.is_file():
        cache_p = data_dir / "dino_patch_cache.fp16.npy"
        _, val_loader, _, _ = build_e2e_dataloaders(
            data_dir=data_dir,
            train_split=0.85,
            batch_size=32,
            num_workers=0,
            seed=seed,
            patch_cache_path=cache_p if (cache_p.is_file() and not use_mock) else None,
        )
        cf_frames = _load_val_sample_frames(data_dir, max_frames=20)
        for spec in candidates:
            if not _load_candidate(system, spec, device):
                continue
            aux_prior = spec["decision_mode"] == "hybrid_prior"
            ol_m = evaluate_isolated_decoder(
                system=system,
                val_loader=val_loader,
                device=device,
                use_amp=use_amp,
                aux_use_prior=aux_prior,
            )
            report["isolated_decoder"][spec["name"]] = ol_m
            if cf_frames:
                cf_m = evaluate_counterfactual_obstacle_sensitivity(
                    system=system,
                    base_frames=cf_frames,
                    device=device,
                    use_amp=use_amp,
                )
                report["counterfactual_sensitivity"][spec["name"]] = cf_m
            print(
                f"[Open-Loop | {spec['name']:15s}] ADE_3s={ol_m['ade_3s_m']:.3f}m | "
                f"FDE_3s={ol_m['fde_3s_m']:.3f}m | CR_traj={ol_m['cr_traj_pct']:.2f}% | "
                f"ECE={ol_m['mean_ece']:.4f} | |gate|={ol_m['abs_temporal_gate']:.4f} | "
                f"Lat={ol_m['latency_ms']:.1f}ms",
                flush=True,
            )

    # --- PART B: Closed-Loop 2x2 Matrix on Unseen MetaDrive 3D Seeds (5000+) ---
    if args.simulator in ("metadrive", "both"):
        md_seeds = [5000 + i for i in range(max(1, int(args.metadrive_episodes)))]
        md_cfg = json.loads(json.dumps(cfg))
        md_cfg.setdefault("env", {})["start_seed"] = 5000
        md_cfg["env"]["max_episode_steps"] = int(args.max_steps)

        md_base = MetaDriveWrapper(
            config=md_cfg,
            use_mock=use_mock,
            start_seed=5000,
            num_scenarios=max(20, len(md_seeds) + 4),
        )
        md_env = RealisticSimEnvWrapper(
            base_env=md_base,
            scenario="nominal_urban_cruise",
            actuator_realism=True,
            localization_mode="hd_map",
        )
        try:
            for spec in candidates:
                if not _load_candidate(system, spec, device):
                    continue
                report["metadrive_unseen_5000"][spec["name"]] = {}
                for regime in ("pure_e2e_shield_off", "hybrid_shield_on"):
                    res = run_closed_loop_episodes(
                        system=system,
                        env=md_env,
                        cfg=md_cfg,
                        seeds=md_seeds,
                        scenarios=EVAL_SCENARIOS,
                        control_regime=regime,
                        device=device,
                        max_steps=int(args.max_steps),
                        use_mock=use_mock,
                    )
                    report["metadrive_unseen_5000"][spec["name"]][regime] = res
                    print(
                        f"[MetaDrive 5000+ | {spec['name']:15s} | {regime:19s}] "
                        f"CF={res['crash_free_pct']:5.1f}% | DS={res['driving_score_pct']:5.1f}% | "
                        f"PDMS={res['pdms']:.4f} | MinTTC={res['min_ttc_s']:4.2f}s | "
                        f"ISO={res['iso_comfort_pct']:5.1f}% | Lat={res['mean_latency_ms']:.1f}ms",
                        flush=True,
                    )
        finally:
            # MUST fully close MetaDrive before launching CARLA (sequential simulator rule)
            md_env.close()

    # --- PART C: Live CARLA 0.9.16 Cross-Town Verification (Strictly Sequential) ---
    if args.simulator in ("carla", "both"):
        from src.env.carla_wrapper import CarlaWrapper

        towns = [t.strip() for t in str(args.carla_towns).split(",") if t.strip()]
        carla_mgr = get_carla_server_manager()
        launched_carla = False
        carla_env: Optional[RealisticSimEnvWrapper] = None
        try:
            carla_cfg = json.loads(json.dumps(cfg))
            carla_cfg.setdefault("env", {}).setdefault("carla", {})
            carla_cfg["env"]["carla"]["auto_launch_server"] = bool(args.auto_launch_carla and not use_mock)
            carla_cfg["env"]["carla"]["offscreen"] = True
            carla_cfg["env"]["carla"]["town"] = towns[0] if towns else "Town10HD_Opt"
            carla_cfg["env"]["max_episode_steps"] = int(args.max_steps)

            if args.auto_launch_carla and not use_mock:
                carla_mgr.ensure_running(
                    host=str(carla_cfg["env"]["carla"].get("host", "127.0.0.1")),
                    port=int(carla_cfg["env"]["carla"].get("port", 2000)),
                    offscreen=True,
                    quality_level="Low",
                    timeout_s=75.0,
                )
                launched_carla = True

            c_base = CarlaWrapper(
                config=carla_cfg,
                use_mock=use_mock,
                start_seed=5000,
                num_scenarios=12,
            )
            carla_env = RealisticSimEnvWrapper(
                base_env=c_base,
                scenario="nominal_urban_cruise",
                actuator_realism=True,
                localization_mode="hd_map",
            )
            carla_actual_mock = bool(c_base.use_mock)
            report["carla_cross_town"]["actual_mock_fallback"] = carla_actual_mock

            carla_eval_matrix = [
                ("v12_il_baseline", "hybrid_shield_on"),
                ("stage1_e2e", "pure_e2e_shield_off"),
                ("stage2_e2e_rl", "pure_e2e_shield_off"),
                ("stage2_e2e_rl", "hybrid_shield_on"),
            ]
            cand_by_name = {c["name"]: c for c in candidates}
            grouped_town_results: Dict[Tuple[str, str], Dict[str, Any]] = {}

            # Outer loop over towns so each CARLA map loads only once
            for t_idx, town in enumerate(towns):
                if not carla_actual_mock:
                    carla_env.set_town(town)
                for cand_name, regime in carla_eval_matrix:
                    spec = cand_by_name[cand_name]
                    if not _load_candidate(system, spec, device):
                        continue
                    res = run_closed_loop_episodes(
                        system=system,
                        env=carla_env,
                        cfg=carla_cfg,
                        seeds=[5100 + t_idx],
                        scenarios=("nominal_urban_cruise",),
                        control_regime=regime,
                        device=device,
                        max_steps=min(220, int(args.max_steps)),
                        use_mock=carla_actual_mock,
                    )
                    grouped_town_results.setdefault((cand_name, regime), {})[town] = res
                    print(
                        f"[CARLA {town:12s} | {cand_name:15s} | {regime:19s}] "
                        f"CF={res['crash_free_pct']:5.1f}% | DS={res['driving_score_pct']:5.1f}% | "
                        f"PDMS={res['pdms']:.4f} | Lat={res['mean_latency_ms']:.1f}ms",
                        flush=True,
                    )

            for (cand_name, regime), town_results in grouped_town_results.items():
                report["carla_cross_town"].setdefault(cand_name, {})[regime] = {
                    "mean_crash_free_pct": round(
                        float(np.mean([v["crash_free_pct"] for v in town_results.values()])), 2
                    ),
                    "mean_driving_score_pct": round(
                        float(np.mean([v["driving_score_pct"] for v in town_results.values()])), 2
                    ),
                    "mean_pdms": round(
                        float(np.mean([v["pdms"] for v in town_results.values()])), 4
                    ),
                    "towns": town_results,
                }
        finally:
            if carla_env is not None:
                carla_env.close()
            if launched_carla:
                try:
                    carla_mgr.stop()
                except Exception:
                    pass

    # --- PART D: CARLA Leaderboard 2.0 / Bench2Drive AutonomousAgent Verification ---
    best_ckpt = (
        Path(args.stage2_ckpt)
        if Path(args.stage2_ckpt).is_file()
        else (Path(args.stage1_ckpt) if Path(args.stage1_ckpt).is_file() else Path(args.v12_ckpt))
    )
    if best_ckpt.is_file():
        report["leaderboard_agent_verification"] = verify_leaderboard_agent_live(
            ckpt_path=best_ckpt,
            temp_ckpt_path=Path(args.stage2_temp),
            use_mock=use_mock,
        )

    tmp_out = out_json.with_suffix(out_json.suffix + ".tmp")
    tmp_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp_out, out_json)
    print(f"\n[09_bench] Saved full Track 1 + Track 3 benchmark report -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
