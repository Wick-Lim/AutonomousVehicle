"""Step 6: Multi-Route, Multi-Town Dynamic Re-Routing & Mapless Corridor Generalization Benchmark."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_validated_config
from src.env import create_driving_env
from src.env.carla_wrapper import AVAILABLE_CARLA_TOWNS
from src.runtime.engine import DrivingRuntimeEngine
from src.sim.metrics import EpisodeSafetyComfortTracker, aggregate_leaderboard
from src.sim.scenarios import RealisticSimEnvWrapper
from src.utils.seed import seed_everything


def _execute_episode(
    engine: DrivingRuntimeEngine,
    env: RealisticSimEnvWrapper,
    session_id: str,
    seed: int,
    max_steps: int,
    perception_mode: str = "vision_radar_fusion",
    post_reset_hook: Optional[Callable[[RealisticSimEnvWrapper, Dict[str, Any]], None]] = None,
    step_hook: Optional[Callable[[RealisticSimEnvWrapper, int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Run one closed-loop episode with optional live navigation / roadblock hooks."""
    sim_dt = float(getattr(env, "step_dt", 0.05))
    tracker = EpisodeSafetyComfortTracker(dt=sim_dt)
    rgb, telemetry, info = env.reset(seed=seed)
    if post_reset_hook is not None:
        post_reset_hook(env, info)

    tracker.reset()
    latencies_ms: List[float] = []
    source_counts: Dict[str, int] = {}
    shield_count = 0
    done = False
    step_n = 0

    while not done and step_n < max_steps:
        step_n += 1
        if step_hook is not None:
            step_hook(env, step_n, info)

        tel_payload = dict(telemetry)
        tel_payload["radar_targets"] = info.get("radar_targets", [])
        tel_payload["imu_telemetry"] = info.get("imu_telemetry")
        tel_payload["elapsed_steps"] = 1

        resp = engine.step(
            session_id=session_id,
            image=rgb,
            telemetry=tel_payload,
            perception_mode=perception_mode,  # type: ignore[arg-type]
            shield_enabled=True,
            dt=sim_dt,
            strict_input=True,
        )
        latencies_ms.append(float(resp.latency_ms))
        if resp.control.shield_triggered:
            shield_count += 1

        action = (resp.control.steering, resp.control.throttle_brake)
        setattr(env, "_last_gated_maneuver", resp.control.gated_maneuver)
        setattr(env.base_env, "_last_gated_maneuver", resp.control.gated_maneuver)
        env._last_predicted_trajectory = list(resp.decision.predicted_trajectory)

        rgb, telemetry, _, done_step, info = env.step(action)
        loc_diag = info.get("localization_diagnostics") if isinstance(info, dict) else None
        if isinstance(loc_diag, dict):
            src_key = str(loc_diag.get("localization_source", "unknown"))
            source_counts[src_key] = source_counts.get(src_key, 0) + 1

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

    engine.delete_session(session_id)
    arrived = bool(info.get("arrive_dest", False))
    crashed = bool(info.get("crash", False))
    comp = float(info.get("route_completion", 0.0))
    nav_state = info.get("nav_state") if isinstance(info, dict) else {}
    reroute_count = int((nav_state or {}).get("reroute_count", getattr(env.base_env, "_reroute_count", 0)))

    ep_summary = tracker.summarize(route_completion=comp, arrived=arrived, crashed=crashed)
    ep_summary.update(
        {
            "seed": int(seed),
            "steps_executed": int(step_n),
            "crashed": crashed,
            "arrived": arrived,
            "route_progress_pct": round(100.0 if arrived else comp * 100.0, 2),
            "reroute_count": reroute_count,
            "shield_interventions": shield_count,
            "mean_latency_ms": round(float(np.mean(latencies_ms)) if latencies_ms else 0.0, 2),
            "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0, 2),
            "localization_sources": source_counts,
            "map_name": str((nav_state or {}).get("map_name", getattr(env.base_env, "town", "procedural"))),
        }
    )
    return ep_summary


def _summarize_suite(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not episodes:
        return {}
    agg = aggregate_leaderboard(episodes)
    return {
        **agg,
        "crash_free_rate_pct": round(
            float((1.0 - np.mean([1.0 if e["crashed"] else 0.0 for e in episodes])) * 100.0), 2
        ),
        "arrival_rate_pct": round(
            float(np.mean([1.0 if e["arrived"] else 0.0 for e in episodes]) * 100.0), 2
        ),
        "mean_route_progress_pct": round(
            float(np.mean([float(e["route_progress_pct"]) for e in episodes])), 2
        ),
        "total_reroutes_executed": int(sum(int(e.get("reroute_count", 0)) for e in episodes)),
        "mean_latency_ms": round(float(np.mean([float(e["mean_latency_ms"]) for e in episodes])), 2),
    }


def _run_metadrive_suites(
    engine: DrivingRuntimeEngine,
    cfg: Dict[str, Any],
    use_mock_env: bool,
    map_blocks: int,
    num_seeds: int,
    max_steps: int,
    actuator_realism: bool,
    start_seed: int,
) -> Dict[str, Any]:
    cfg_md = dict(cfg)
    cfg_md["env"] = dict(cfg.get("env", {}))
    cfg_md["env"]["map_blocks"] = int(map_blocks)

    base_env = create_driving_env(
        config=cfg_md,
        use_mock=use_mock_env,
        start_seed=start_seed,
        num_scenarios=max(6, num_seeds + 2),
        simulator="metadrive",
    )
    actual_real_3d = not bool(getattr(base_env, "use_mock", True))
    try:
        env = RealisticSimEnvWrapper(
            base_env=base_env,
            scenario="nominal_urban_cruise",
            actuator_realism=actuator_realism,
            localization_mode="hd_map",
        )
        # Suite A: Multi-block procedural topologies + live mid-drive set_destination re-route
        multiblock_eps: List[Dict[str, Any]] = []
        for idx in range(num_seeds):
            seed = start_seed + idx

            def _mid_reroute_hook(
                w_env: RealisticSimEnvWrapper, step_i: int, _info: Dict[str, Any], _i: int = idx
            ) -> None:
                if _i == 1 and step_i == 55:
                    nodes = w_env.list_spawn_points()
                    if nodes:
                        w_env.set_destination(dest_spawn_idx=int(nodes[-1]["index"]))

            ep = _execute_episode(
                engine=engine,
                env=env,
                session_id=f"md_multiblock_{seed}",
                seed=seed,
                max_steps=max_steps,
                step_hook=_mid_reroute_hook,
            )
            ep["test_case"] = (
                f"metadrive_{map_blocks}blocks_live_reroute_seed_{seed}"
                if idx == 1
                else f"metadrive_{map_blocks}blocks_seed_{seed}"
            )
            multiblock_eps.append(ep)

        # Suite C (MetaDrive): Mapless Vision-Only & Hybrid HD-Map-Dropout Failover
        cfg_md_1b = dict(cfg)
        cfg_md_1b["env"] = dict(cfg.get("env", {}))
        cfg_md_1b["env"]["map_blocks"] = 1
    finally:
        base_env.close()

    base_env_mapless = create_driving_env(
        config=cfg_md_1b,
        use_mock=use_mock_env,
        start_seed=start_seed,
        num_scenarios=4,
        simulator="metadrive",
    )
    try:
        env_ml = RealisticSimEnvWrapper(
            base_env=base_env_mapless,
            scenario="nominal_urban_cruise",
            actuator_realism=actuator_realism,
            localization_mode="mapless_vision",
        )
        ep_mapless = _execute_episode(
            engine=engine,
            env=env_ml,
            session_id="md_mapless_vision_pure",
            seed=start_seed,
            max_steps=260,
        )
        ep_mapless["test_case"] = "metadrive_pure_mapless_vision"

        ep_mapless_curve = _execute_episode(
            engine=engine,
            env=env_ml,
            session_id="md_mapless_vision_curve",
            seed=start_seed + 2,
            max_steps=260,
        )
        ep_mapless_curve["test_case"] = "metadrive_pure_mapless_curve"

        env_ml.set_localization_mode("hybrid_fallback")
        env_ml.set_fault_mode("none")

        def _md_dropout_hook(
            w_env: RealisticSimEnvWrapper, step_i: int, _info: Dict[str, Any]
        ) -> None:
            if step_i == 35:
                w_env.set_fault_mode("hd_map_dropout")

        ep_dropout = _execute_episode(
            engine=engine,
            env=env_ml,
            session_id="md_hybrid_hd_map_dropout",
            seed=start_seed + 1,
            max_steps=260,
            step_hook=_md_dropout_hook,
        )
        ep_dropout["test_case"] = "metadrive_hybrid_hd_map_dropout"
    finally:
        base_env_mapless.close()

    return {
        "real_3d_engine": actual_real_3d,
        "map_blocks": int(map_blocks),
        "multiblock_procedural": {
            "summary": _summarize_suite(multiblock_eps),
            "episodes": multiblock_eps,
        },
        "mapless_and_dropout": {
            "summary": _summarize_suite([ep_mapless, ep_mapless_curve, ep_dropout]),
            "episodes": [ep_mapless, ep_mapless_curve, ep_dropout],
        },
    }


def _run_carla_suites(
    engine: DrivingRuntimeEngine,
    cfg: Dict[str, Any],
    use_mock_env: bool,
    towns: List[str],
    max_steps: int,
    actuator_realism: bool,
    start_seed: int,
) -> Dict[str, Any]:
    cfg_carla = dict(cfg)
    cfg_carla["env"] = dict(cfg.get("env", {}))
    cfg_carla["env"]["carla"] = dict(cfg_carla["env"].get("carla", {}))
    cfg_carla["env"]["carla"]["town"] = towns[0]

    base_env = create_driving_env(
        config=cfg_carla,
        use_mock=use_mock_env,
        start_seed=start_seed,
        num_scenarios=max(4, len(towns) + 2),
        simulator="carla",
    )
    actual_real_3d = not bool(getattr(base_env, "use_mock", True))
    multitown_eps: List[Dict[str, Any]] = []
    mapless_eps: List[Dict[str, Any]] = []

    try:
        env = RealisticSimEnvWrapper(
            base_env=base_env,
            scenario="nominal_urban_cruise",
            actuator_realism=actuator_realism,
            localization_mode="hd_map",
        )
        for idx, town in enumerate(towns):
            env.set_town(town)
            seed = start_seed + idx

            def _init_p2p(w_env: RealisticSimEnvWrapper, _info: Dict[str, Any]) -> None:
                w_env.set_destination(min_distance_m=170.0)

            def _live_event_hook(
                w_env: RealisticSimEnvWrapper,
                step_i: int,
                _info: Dict[str, Any],
                _town: str = town,
                _idx: int = idx,
            ) -> None:
                # 1) Live mid-drive destination change on first town
                if _idx == 0 and step_i == 85:
                    w_env.set_destination(min_distance_m=140.0)
                # 2) Live physical lane roadblock bypass on highway / last town
                elif (_town.startswith("Town04") or (_idx == len(towns) - 1 and _idx > 0)) and step_i == 65:
                    w_env.spawn_roadblock(distance_ahead_m=38.0)

            ep = _execute_episode(
                engine=engine,
                env=env,
                session_id=f"carla_p2p_{town}_{seed}",
                seed=seed,
                max_steps=max_steps,
                post_reset_hook=_init_p2p,
                step_hook=_live_event_hook,
            )
            ep["town"] = town
            ep["test_case"] = f"carla_{town}_p2p_dynamic_reroute"
            multitown_eps.append(ep)

        # CARLA Pure Mapless Vision + Hybrid HD-Map-Dropout Validation on Town10HD_Opt (or first town)
        primary_town = "Town10HD_Opt" if "Town10HD_Opt" in towns else towns[0]
        env.set_town(primary_town)
        env.set_localization_mode("mapless_vision")
        env.set_fault_mode("none")
        ep_c_mapless = _execute_episode(
            engine=engine,
            env=env,
            session_id=f"carla_mapless_{primary_town}",
            seed=start_seed,
            max_steps=min(max_steps, 440),
        )
        ep_c_mapless["town"] = primary_town
        ep_c_mapless["test_case"] = f"carla_{primary_town}_pure_mapless_vision"
        mapless_eps.append(ep_c_mapless)

        env.set_localization_mode("hybrid_fallback")
        env.set_fault_mode("none")

        def _carla_dropout_hook(
            w_env: RealisticSimEnvWrapper, step_i: int, _info: Dict[str, Any]
        ) -> None:
            if step_i == 35:
                w_env.set_fault_mode("hd_map_dropout")

        ep_c_dropout = _execute_episode(
            engine=engine,
            env=env,
            session_id=f"carla_dropout_{primary_town}",
            seed=start_seed + 1,
            max_steps=min(max_steps, 440),
            step_hook=_carla_dropout_hook,
        )
        ep_c_dropout["town"] = primary_town
        ep_c_dropout["test_case"] = f"carla_{primary_town}_hybrid_hd_map_dropout"
        mapless_eps.append(ep_c_dropout)
    finally:
        base_env.close()

    return {
        "real_3d_engine": actual_real_3d,
        "towns_evaluated": list(towns),
        "multitown_point_to_point": {
            "summary": _summarize_suite(multitown_eps),
            "episodes": multitown_eps,
        },
        "mapless_and_dropout": {
            "summary": _summarize_suite(mapless_eps),
            "episodes": mapless_eps,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Multi-Route, Multi-Town Dynamic A* Re-Routing & Mapless Vision Benchmark."
    )
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--simulator",
        type=str,
        default="both",
        choices=["metadrive", "carla", "both"],
    )
    parser.add_argument(
        "--towns",
        nargs="+",
        default=["Town10HD_Opt", "Town03_Opt", "Town04_Opt"],
        choices=list(AVAILABLE_CARLA_TOWNS),
    )
    parser.add_argument("--metadrive-blocks", type=int, default=3)
    parser.add_argument("--metadrive-seeds", type=int, default=4)
    parser.add_argument("--metadrive-max-steps", type=int, default=450)
    parser.add_argument("--carla-max-steps", type=int, default=720)
    parser.add_argument("--actuator-realism", action="store_true", default=True)
    parser.add_argument("--no-actuator-realism", dest="actuator_realism", action="store_false")
    parser.add_argument("--auto-launch-carla", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use fast kinematic mock for CI smoke checks.")
    parser.add_argument("--mock-model", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "multiroute_benchmark_report.json"),
    )
    args = parser.parse_args()

    cfg = load_validated_config(args.config).to_dict()
    seed_everything(int(cfg.get("project", {}).get("seed", 42)))
    if args.auto_launch_carla:
        cfg.setdefault("env", {}).setdefault("carla", {})["auto_launch_server"] = True

    md_installed = importlib.util.find_spec("metadrive") is not None
    use_mock_model = bool(args.mock or args.mock_model)
    engine = DrivingRuntimeEngine(config=cfg, use_mock=use_mock_model)
    engine.warmup(passes=1)
    start_seed = int(cfg.get("env", {}).get("eval_start_seed", 5000))

    results: Dict[str, Any] = {}
    all_episodes: List[Dict[str, Any]] = []

    if args.simulator in ("metadrive", "both"):
        md_mock = bool(args.mock or not md_installed)
        md_res = _run_metadrive_suites(
            engine=engine,
            cfg=cfg,
            use_mock_env=md_mock,
            map_blocks=max(1, int(args.metadrive_blocks)),
            num_seeds=max(1, int(args.metadrive_seeds)),
            max_steps=max(20, int(args.metadrive_max_steps)),
            actuator_realism=bool(args.actuator_realism),
            start_seed=start_seed,
        )
        results["metadrive"] = md_res
        all_episodes.extend(md_res["multiblock_procedural"]["episodes"])
        all_episodes.extend(md_res["mapless_and_dropout"]["episodes"])

    if args.simulator in ("carla", "both"):
        carla_res = _run_carla_suites(
            engine=engine,
            cfg=cfg,
            use_mock_env=bool(args.mock),
            towns=list(args.towns),
            max_steps=max(20, int(args.carla_max_steps)),
            actuator_realism=bool(args.actuator_realism),
            start_seed=start_seed,
        )
        results["carla"] = carla_res
        all_episodes.extend(carla_res["multitown_point_to_point"]["episodes"])
        all_episodes.extend(carla_res["mapless_and_dropout"]["episodes"])

    overall = _summarize_suite(all_episodes)
    report = {
        "simulator_scope": args.simulator,
        "model_mode": "mock" if use_mock_model else "real_dinov2_laya",
        "actuator_realism": bool(args.actuator_realism),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall": overall,
        "suites": results,
    }

    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_path, out_path)

    print("\n" + "=" * 98)
    print(
        f"Multi-Route / Dynamic A* Re-Routing / Mapless Benchmark "
        f"[Scope={args.simulator.upper()} | ActuatorRealism={'ON' if args.actuator_realism else 'OFF'}]"
    )
    print("=" * 98)
    if overall:
        print(
            f"Overall ({len(all_episodes)} eps): Mean DS={overall['mean_driving_score_pct']:.1f}% | "
            f"Crash-Free={overall['crash_free_rate_pct']:.1f}% | "
            f"Progress={overall['mean_route_progress_pct']:.1f}% | "
            f"Re-Routes={overall['total_reroutes_executed']} | "
            f"MinTTC={overall['min_ttc_s']:.2f}s | Lat={overall['mean_latency_ms']:.1f}ms"
        )
    print("-" * 98)
    print(
        f"  {'Test Case':<42} | {'CrashFree':>9} | {'Progress':>8} | {'ReRoutes':>8} | {'MinTTC':>7} | {'Score':>7}"
    )
    print("-" * 98)
    for ep in all_episodes:
        cf_str = "PASS" if not ep["crashed"] else "CRASH"
        print(
            f"  {ep['test_case']:<42} | {cf_str:>9} | "
            f"{ep['route_progress_pct']:>7.1f}% | {ep['reroute_count']:>8d} | "
            f"{ep['min_ttc_s']:>6.2f}s | {ep['driving_score_pct']:>6.1f}%"
        )
    print("=" * 98)
    print(f"Report saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
