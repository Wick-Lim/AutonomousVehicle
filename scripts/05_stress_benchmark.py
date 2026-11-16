"""Step 5: Adversarial 3D Stress Benchmark & Safety/Comfort Leaderboard (MetaDrive 3D & CARLA 0.9.16)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_validated_config
from src.env import create_driving_env
from src.runtime.engine import DrivingRuntimeEngine
from src.sim.metrics import EpisodeSafetyComfortTracker, aggregate_leaderboard
from src.sim.scenarios import SCENARIO_CATALOG, RealisticSimEnvWrapper
from src.utils.seed import seed_everything


def _run_suite_for_mode(
    engine: DrivingRuntimeEngine,
    base_env: Any,
    scenarios: List[str],
    perception_mode: str,
    episodes_per_scenario: int,
    max_steps: int,
    actuator_realism: bool,
    start_seed: int,
) -> Dict[str, Any]:
    env = RealisticSimEnvWrapper(
        base_env=base_env,
        scenario=scenarios[0],
        actuator_realism=actuator_realism,
    )
    sim_dt = float(getattr(env, "step_dt", 0.05))
    tracker = EpisodeSafetyComfortTracker(dt=sim_dt)

    per_scenario: Dict[str, Dict[str, Any]] = {}
    all_summaries: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []
    seed_cursor = int(start_seed)

    for sc_name in scenarios:
        env.set_scenario(sc_name)
        sc_summaries: List[Dict[str, Any]] = []
        sc_crashes: List[bool] = []
        sc_arrivals: List[bool] = []
        sc_completions: List[float] = []
        sc_shield_count = 0

        for ep_idx in range(episodes_per_scenario):
            seed = seed_cursor
            seed_cursor += 1
            sid = f"stress_{perception_mode}_{sc_name}_{ep_idx}"
            rgb, telemetry, info = env.reset(seed=seed)
            tracker.reset()
            done = False
            step_n = 0

            while not done and step_n < max_steps:
                step_n += 1
                tel_payload = dict(telemetry)
                tel_payload["radar_targets"] = info.get("radar_targets", [])
                tel_payload["imu_telemetry"] = info.get("imu_telemetry")
                tel_payload["elapsed_steps"] = 1

                resp = engine.step(
                    session_id=sid,
                    image=rgb,
                    telemetry=tel_payload,
                    perception_mode=perception_mode,  # type: ignore[arg-type]
                    shield_enabled=True,
                    dt=sim_dt,
                    strict_input=True,
                )
                latencies_ms.append(float(resp.latency_ms))
                if resp.control.shield_triggered:
                    sc_shield_count += 1

                action = (resp.control.steering, resp.control.throttle_brake)
                setattr(env, "_last_gated_maneuver", resp.control.gated_maneuver)
                setattr(base_env, "_last_gated_maneuver", resp.control.gated_maneuver)
                rgb, telemetry, _, done_step, info = env.step(action)
                act_diag = info.get("actuator_diagnostics") if isinstance(info, dict) else None
                eff_action = (
                    (
                        float(act_diag.get("effective_steer", action[0])),
                        float(act_diag.get("effective_throttle_brake", action[1])),
                    )
                    if isinstance(act_diag, dict)
                    else action
                )
                tracker.step(
                    telemetry=telemetry,
                    control_action=eff_action,
                    gated_info={
                        "shield_triggered": resp.control.shield_triggered,
                        "shield_reason": resp.control.shield_reason,
                    },
                    extra_info=info,
                    dt=sim_dt,
                )
                done = bool(done_step)

            engine.delete_session(sid)
            arrived = bool(info.get("arrive_dest", False))
            crashed = bool(info.get("crash", False))
            comp = float(info.get("route_completion", 0.0))
            ep_summary = tracker.summarize(
                route_completion=comp,
                arrived=arrived,
                crashed=crashed,
            )
            ep_summary["scenario"] = sc_name
            ep_summary["perception_mode"] = perception_mode
            ep_summary["crashed"] = crashed
            ep_summary["arrived"] = arrived
            sc_summaries.append(ep_summary)
            all_summaries.append(ep_summary)
            sc_crashes.append(crashed)
            sc_arrivals.append(arrived)
            sc_completions.append(comp)

        agg_sc = aggregate_leaderboard(sc_summaries)
        per_scenario[sc_name] = {
            **agg_sc,
            "crash_free_rate_pct": round(float((1.0 - np.mean(sc_crashes)) * 100.0), 2),
            "mean_route_progress_pct": round(float(np.mean(sc_completions) * 100.0), 2),
            "arrival_rate_pct": round(float(np.mean(sc_arrivals) * 100.0), 2),
            "shield_interventions": sc_shield_count,
        }

    return {
        "perception_mode": perception_mode,
        "overall": {
            **aggregate_leaderboard(all_summaries),
            "crash_free_rate_pct": round(
                float((1.0 - np.mean([float(e["crashed"]) for e in all_summaries])) * 100.0),
                2,
            ),
            "mean_latency_ms": round(float(np.mean(latencies_ms)) if latencies_ms else 0.0, 2),
            "p95_latency_ms": round(
                float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0, 2
            ),
        },
        "per_scenario": per_scenario,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run adversarial 3D stress benchmark & ISO/Leaderboard safety-comfort evaluation."
    )
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--simulator", type=str, default="metadrive", choices=["metadrive", "carla"])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIO_CATALOG.keys()),
        choices=list(SCENARIO_CATALOG.keys()),
    )
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Steps per episode (defaults to a 22.0s physical horizon: 220 at 10Hz, 440 at 20Hz).",
    )
    parser.add_argument("--map-blocks", type=int, default=None)
    parser.add_argument(
        "--perception-mode",
        type=str,
        default="vision_radar_fusion",
        choices=["vision_radar_fusion", "vision_estimated", "gt"],
    )
    parser.add_argument(
        "--compare-fusion",
        action="store_true",
        help="Run both vision_estimated and vision_radar_fusion to quantify sensor fusion resilience.",
    )
    parser.add_argument(
        "--actuator-realism",
        action="store_true",
        default=True,
        help="Apply realistic 1st-order servo lag, rate limits & pedal deadzone.",
    )
    parser.add_argument("--no-actuator-realism", dest="actuator_realism", action="store_false")
    parser.add_argument("--auto-launch-carla", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use kinematic mock for fast CI runs.")
    parser.add_argument("--mock-model", action="store_true", help="Use lightweight mock DINOv2/Laya tensors.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "stress_benchmark_report.json"),
    )
    args = parser.parse_args()

    cfg = load_validated_config(args.config).to_dict()
    seed_everything(int(cfg.get("project", {}).get("seed", 42)))
    if args.auto_launch_carla:
        cfg.setdefault("env", {}).setdefault("carla", {})["auto_launch_server"] = True
    if args.map_blocks is not None:
        cfg.setdefault("env", {})["map_blocks"] = int(args.map_blocks)
    elif args.simulator == "metadrive":
        cfg.setdefault("env", {})["map_blocks"] = 1

    md_installed = importlib.util.find_spec("metadrive") is not None
    use_mock_env = bool(args.mock or (args.simulator == "metadrive" and not md_installed))
    use_mock_model = bool(args.mock or args.mock_model)

    engine = DrivingRuntimeEngine(config=cfg, use_mock=use_mock_model)
    engine.warmup(passes=1)

    eval_start_seed = int(cfg.get("env", {}).get("eval_start_seed", 5000))
    total_eps = max(1, len(args.scenarios) * max(1, int(args.episodes_per_scenario)))
    base_env = create_driving_env(
        config=cfg,
        use_mock=use_mock_env,
        start_seed=eval_start_seed,
        num_scenarios=total_eps,
        simulator=args.simulator,
    )

    sim_dt_base = float(getattr(base_env, "step_dt", 0.10))
    eff_max_steps = (
        max(10, int(args.max_steps))
        if args.max_steps is not None
        else max(220, int(round(22.0 / max(0.01, sim_dt_base))))
    )

    modes = ["vision_estimated", "vision_radar_fusion"] if args.compare_fusion else [args.perception_mode]
    results_by_mode: Dict[str, Any] = {}
    try:
        for mode in modes:
            results_by_mode[mode] = _run_suite_for_mode(
                engine=engine,
                base_env=base_env,
                scenarios=list(args.scenarios),
                perception_mode=mode,
                episodes_per_scenario=max(1, int(args.episodes_per_scenario)),
                max_steps=eff_max_steps,
                actuator_realism=bool(args.actuator_realism),
                start_seed=eval_start_seed,
            )
    finally:
        base_env.close()

    actual_real_3d = not bool(getattr(base_env, "use_mock", True))
    report = {
        "simulator": args.simulator,
        "real_3d_engine": actual_real_3d,
        "model_mode": "mock" if use_mock_model else "real_dinov2_laya",
        "actuator_realism": bool(args.actuator_realism),
        "scenarios": list(args.scenarios),
        "episodes_per_scenario": int(args.episodes_per_scenario),
        "max_steps_per_episode": eff_max_steps,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results_by_mode,
    }

    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_path, out_path)

    print("\n" + "=" * 92)
    print(
        f"Adversarial 3D Stress Leaderboard [{args.simulator.upper()} | "
        f"{'REAL 3D' if actual_real_3d else 'MOCK'} | ActuatorRealism={'ON' if args.actuator_realism else 'OFF'}]"
    )
    print("=" * 92)
    for mode, data in results_by_mode.items():
        ov = data["overall"]
        print(
            f"\nMode: {mode} | Mean DS={ov['mean_driving_score_pct']:.1f}% | "
            f"Crash-Free={ov['crash_free_rate_pct']:.1f}% | "
            f"Min TTC={ov['min_ttc_s']:.2f}s | RMS Jerk={ov['mean_rms_long_jerk_mps3']:.2f} m/s^3 | "
            f"Comfort={ov['mean_iso_comfort_rate_pct']:.1f}% | Lat={ov['mean_latency_ms']:.1f}ms"
        )
        print("-" * 92)
        print(f"  {'Scenario':<34} | {'CrashFree':>9} | {'MinTTC':>7} | {'RMSJerk':>8} | {'Comfort':>8} | {'Score':>7}")
        print("-" * 92)
        for sc_name, sc_m in data["per_scenario"].items():
            print(
                f"  {sc_name:<34} | {sc_m['crash_free_rate_pct']:>8.1f}% | "
                f"{sc_m['min_ttc_s']:>6.2f}s | {sc_m['mean_rms_long_jerk_mps3']:>7.2f} | "
                f"{sc_m['mean_iso_comfort_rate_pct']:>7.1f}% | {sc_m['mean_driving_score_pct']:>6.1f}%"
            )
    print("=" * 92)
    print(f"Report saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
