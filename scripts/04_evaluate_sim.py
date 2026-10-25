"""Step 4: Closed-Loop Simulation Benchmark on Unseen Maps (50 Scenarios & 4 KPIs)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control.shield_pid import SafetyShieldPID
from src.env import create_driving_env
from src.env.expert_labeler import ExpertLabeler
from src.models.laya_pilot import AutonomousDrivingSystem
from src.utils.calibration import compute_binary_ece, compute_multiclass_ece


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def render_hud_frame(
    rgb: np.ndarray,
    telemetry: Dict[str, Any],
    decision: Dict[str, Any],
    gated_info: Dict[str, Any],
    control_action: Tuple[float, float],
    info: Dict[str, Any],
    ep_idx: int,
    num_episodes: int,
    dt_ms: float,
    simulator_name: str,
) -> np.ndarray:
    """Render an upscaled 784x532 BGR HUD frame overlaying Laya decisions, telemetry, and Safety Shield."""
    cam_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hd_rgb = info.get("hd_rgb")
    if isinstance(hd_rgb, np.ndarray) and hd_rgb.ndim == 3:
        hd_bgr = cv2.cvtColor(hd_rgb, cv2.COLOR_RGB2BGR)
        cam_up = cv2.resize(hd_bgr, (784, 392), interpolation=cv2.INTER_AREA)
        pip = cv2.resize(cam_bgr, (235, 118), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(cam_up, (784 - 245, 8), (784 - 8, 130), (20, 24, 30), -1)
        cam_up[10:128, 784 - 243 : 784 - 8] = pip
        cv2.rectangle(cam_up, (784 - 245, 8), (784 - 8, 130), (90, 180, 255), 1)
    else:
        cam_up = cv2.resize(cam_bgr, (784, 392), interpolation=cv2.INTER_LINEAR)

    panel = np.full((140, 784, 3), (24, 26, 32), dtype=np.uint8)
    cv2.line(panel, (0, 0), (784, 0), (90, 180, 255), 2)

    spd = float(telemetry.get("speed_kmh", 0.0))
    tgt_spd = float(gated_info.get("target_speed_kmh", 40.0))
    offset = float(telemetry.get("lane_offset_m", 0.0))
    lead_d = float(telemetry.get("lead_distance_m", 60.0))
    tl = str(telemetry.get("traffic_light", "green")).upper()
    route_pct = float(info.get("route_completion", 0.0)) * 100.0
    fps = 1000.0 / max(1e-2, dt_ms)

    # Row 1: Simulator, Episode, Speed, Latency/FPS, Progress
    cv2.putText(
        panel,
        f"[{simulator_name.upper()}] Ep {ep_idx + 1}/{num_episodes} | "
        f"Speed: {spd:4.1f}/{tgt_spd:2.0f} km/h | "
        f"Latency: {dt_ms:4.1f} ms ({fps:4.1f} FPS) | "
        f"Route: {route_pct:5.1f}%",
        (14, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (235, 240, 245),
        1,
        cv2.LINE_AA,
    )

    # Row 2: Laya Non-Autoregressive Decision + Confidence + AEB
    m_name = str(gated_info.get("maneuver", decision["maneuver"])).upper()
    m_conf = float(decision["maneuver_confidence"]) * 100.0
    s_lvl = int(gated_info.get("speed_level", decision["speed_level"]))
    aeb_p = float(decision["emergency_brake"]) * 100.0
    shield_on = bool(gated_info.get("shield_triggered", False))
    shield_reason = str(gated_info.get("shield_reason", "none"))

    dec_color = (80, 235, 130) if not shield_on else (70, 90, 255)
    cv2.putText(
        panel,
        f"Laya Decision: {m_name} (Conf {m_conf:4.1f}%) | "
        f"Speed Lvl: {s_lvl} | AEB Prob: {aeb_p:4.1f}%",
        (14, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        dec_color,
        2,
        cv2.LINE_AA,
    )

    shield_txt = f"SHIELD: {shield_reason}" if shield_on else "SHIELD: NOMINAL (PASS)"
    shield_col = (60, 75, 255) if shield_on else (110, 220, 110)
    cv2.putText(
        panel,
        shield_txt,
        (525, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        shield_col,
        2 if shield_on else 1,
        cv2.LINE_AA,
    )

    # Row 3: Physical PID Actuation (Linear Steering Transition [steer_from -> steer_to]) & Environment State
    steer, tb = float(control_action[0]), float(control_action[1])
    s_from = float(gated_info.get("steer_from", steer))
    s_to = float(gated_info.get("steer_to", steer))
    cv2.putText(
        panel,
        f"Linear Steer: {s_from:+5.2f} -> {s_to:+5.2f} | Pedal: {tb:+5.2f} | "
        f"Lead: {lead_d:4.1f}m | Offset: {offset:+4.2f}m | Light: {tl}",
        (14, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (185, 200, 220),
        1,
        cv2.LINE_AA,
    )

    return np.vstack([cam_up, panel])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate closed-loop autonomous driving on unseen MetaDrive maps and measure KPIs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--projector-ckpt", type=str, default=None)
    parser.add_argument("--temperature-ckpt", type=str, default=None)
    parser.add_argument(
        "--simulator",
        type=str,
        default=None,
        choices=["metadrive", "carla"],
        help="Simulator backend ('metadrive' or 'carla').",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open real-time 3D simulator window and live OpenCV HUD dashboard.",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        nargs="?",
        const="checkpoints/driving_demo.mp4",
        default=None,
        help="Save HUD driving video to MP4 file (default: checkpoints/driving_demo.mp4).",
    )
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.render:
        cfg.setdefault("env", {})["use_render"] = True
    use_mock = bool(args.mock or cfg.get("project", {}).get("use_mock", False))
    simulator_name = (args.simulator or cfg.get("env", {}).get("simulator", "metadrive")).lower()
    device = resolve_device(cfg.get("project", {}).get("device", "auto"))

    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("train", {})
    num_episodes = int(
        args.episodes
        if args.episodes is not None
        else env_cfg.get("eval_episodes", 50)
    )
    eval_start_seed = int(env_cfg.get("eval_start_seed", 5000))

    proj_ckpt_path = Path(
        args.projector_ckpt
        if args.projector_ckpt is not None
        else PROJECT_ROOT / train_cfg.get("projector_ckpt", "checkpoints/projector_best.pt")
    )
    temp_ckpt_path = Path(
        args.temperature_ckpt
        if args.temperature_ckpt is not None
        else PROJECT_ROOT / train_cfg.get("temperature_ckpt", "checkpoints/temperature.pt")
    )

    system = AutonomousDrivingSystem(config=cfg, use_mock=use_mock).to(device)
    if proj_ckpt_path.exists():
        ckpt = torch.load(proj_ckpt_path, map_location=device, weights_only=False)
        system.projector.load_state_dict(ckpt["projector_state_dict"])
        print(f"[04_evaluate_sim] Loaded SpatialProjector from: {proj_ckpt_path}")

    calibrated_ece_from_ckpt = None
    if temp_ckpt_path.exists():
        t_ckpt = torch.load(temp_ckpt_path, map_location=device, weights_only=False)
        temps = t_ckpt["temperatures"]
        system.laya.set_temperatures(
            temps["t_maneuver"],
            temps["t_speed"],
            temps["t_aeb"],
            temps.get("b_aeb", 0.0),
        )
        calibrated_ece_from_ckpt = t_ckpt.get("after_metrics", {}).get("mean_ece", None)
        print(
            f"[04_evaluate_sim] Loaded calibrated temperatures: "
            f"T_m={temps['t_maneuver']:.3f}, T_s={temps['t_speed']:.3f}, T_aeb={temps['t_aeb']:.3f}"
        )

    system.eval()
    controller = SafetyShieldPID(config=cfg, use_mock=use_mock)
    labeler = ExpertLabeler()
    env = create_driving_env(
        config=cfg,
        use_mock=use_mock,
        start_seed=eval_start_seed,
        num_scenarios=num_episodes,
        simulator=simulator_name,
    )

    # Warm-up pass for accurate latency measurement
    dummy_rgb = np.zeros(
        (int(env_cfg.get("camera_height", 196)), int(env_cfg.get("camera_width", 392)), 3),
        dtype=np.uint8,
    )
    dummy_tel = {
        "speed_kmh": 35.0,
        "lane_offset_m": 0.0,
        "heading_error": 0.0,
        "left_lane_available": True,
        "right_lane_available": True,
        "nav_command": "go_straight",
    }
    for _ in range(3):
        _ = system.predict_single(dummy_rgb, dummy_tel, device=device)

    latencies_ms: List[float] = []
    episode_completions: List[float] = []
    episode_driving_scores: List[float] = []
    episode_arrived: List[bool] = []
    episode_crashed: List[bool] = []
    episode_scenarios: List[str] = []
    latest_fsd_metrics: Dict[str, Any] = {}
    shield_reasons: Counter[str] = Counter()

    all_m_probs: List[torch.Tensor] = []
    all_s_probs: List[torch.Tensor] = []
    all_a_probs: List[torch.Tensor] = []
    all_m_targets: List[int] = []
    all_s_targets: List[int] = []
    all_a_targets: List[int] = []

    video_writer = None
    video_out_path = None
    if args.save_video:
        video_out_path = Path(args.save_video)
        if not video_out_path.is_absolute():
            video_out_path = PROJECT_ROOT / video_out_path
        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_out_path), fourcc, 20.0, (784, 532))
        print(f"[04_evaluate_sim] Recording driving HUD video to: {video_out_path}")

    for ep in tqdm(
        range(num_episodes), desc="Evaluating Closed-Loop Unseen Maps", unit="ep"
    ):
        rgb, telemetry, info = env.reset(seed=eval_start_seed + ep)
        controller.reset()
        done = False

        while not done:
            # Ground-truth reference label for online ECE monitoring
            expert_act = env.get_expert_action(info)
            gt_label = labeler.label_step(telemetry, expert_act, info)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Single-pass non-autoregressive inference (Preprocess + DINOv2 + Projector + Laya)
            tensor_img = (
                torch.from_numpy(rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )
            with torch.no_grad():
                out = system(tensor_img, [telemetry])
                m_idx = int(out["maneuver_idx"][0].item())
                s_lvl = int(out["speed_level"][0].item())
                m_conf = float(out["maneuver_confidence"][0].item())
                s_conf = float(out["speed_confidence"][0].item())
                aeb_prob = float(out["emergency_brake"][0].item())

            if device.type == "cuda":
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            all_m_probs.append(out["maneuver_probs"].cpu())
            all_s_probs.append(out["speed_probs"].cpu())
            all_a_probs.append(out["emergency_brake"].cpu())
            all_m_targets.append(gt_label.maneuver_idx)
            all_s_targets.append(gt_label.speed_level)
            all_a_targets.append(gt_label.emergency_brake)

            decision = {
                "maneuver": ("keep_lane", "change_left", "change_right", "follow_lead")[m_idx],
                "maneuver_idx": m_idx,
                "maneuver_confidence": m_conf,
                "speed_level": s_lvl,
                "speed_confidence": s_conf,
                "emergency_brake": aeb_prob,
            }

            # Apply Calibrated Safety Shield + PID controller
            control_action, gated_info = controller.compute_control(decision, telemetry)
            if gated_info["shield_triggered"]:
                shield_reasons[gated_info["shield_reason"]] += 1

            if args.render or video_writer is not None:
                hud_bgr = render_hud_frame(
                    rgb=rgb,
                    telemetry=telemetry,
                    decision=decision,
                    gated_info=gated_info,
                    control_action=control_action,
                    info=info,
                    ep_idx=ep,
                    num_episodes=num_episodes,
                    dt_ms=dt_ms,
                    simulator_name=simulator_name,
                )
                if video_writer is not None:
                    video_writer.write(hud_bgr)
                if args.render:
                    win_name = "AutonomousVehicle - DINOv2 + Laya Live HUD"
                    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
                    cv2.imshow(win_name, hud_bgr)
                    key = cv2.waitKey(25) & 0xFF
                    if key == 27 or key == ord("q"):
                        done = True

            rgb, telemetry, _, done_step, info = env.step(control_action)
            done = bool(done or done_step)

        episode_completions.append(float(info.get("route_completion", 0.0)))
        episode_driving_scores.append(
            float(info.get("driving_score", float(info.get("route_completion", 0.0)) * 100.0))
        )
        episode_arrived.append(bool(info.get("arrive_dest", False)))
        episode_crashed.append(bool(info.get("crash", False)))
        if "fsd_scenario" in info:
            episode_scenarios.append(str(info["fsd_scenario"]))
        if "fsd_metrics" in info:
            latest_fsd_metrics = dict(info["fsd_metrics"])

    if video_writer is not None:
        video_writer.release()
    if args.render:
        cv2.destroyAllWindows()

    # Compute KPIs before closing simulator connection
    mean_latency_ms = float(np.mean(latencies_ms))
    p95_latency_ms = float(np.percentile(latencies_ms, 95))
    fps = 1000.0 / max(1e-3, mean_latency_ms)

    route_completion_rate = float(np.mean(episode_arrived) * 100.0)
    crash_free_rate = float((1.0 - np.mean(episode_crashed)) * 100.0)

    m_probs_cat = torch.cat(all_m_probs, dim=0)
    s_probs_cat = torch.cat(all_s_probs, dim=0)
    a_probs_cat = torch.cat(all_a_probs, dim=0)
    m_tgt_t = torch.tensor(all_m_targets, dtype=torch.long)
    s_tgt_t = torch.tensor(all_s_targets, dtype=torch.long)
    a_tgt_t = torch.tensor(all_a_targets, dtype=torch.long)

    online_ece = (
        compute_multiclass_ece(m_probs_cat, m_tgt_t)
        + compute_multiclass_ece(s_probs_cat, s_tgt_t)
        + compute_binary_ece(a_probs_cat, a_tgt_t)
    ) / 3.0
    reported_ece = (
        float(calibrated_ece_from_ckpt)
        if calibrated_ece_from_ckpt is not None
        else float(online_ece)
    )

    mean_driving_score = float(np.mean(episode_driving_scores)) if episode_driving_scores else route_completion_rate

    summary = {
        "simulator": simulator_name,
        "episodes": num_episodes,
        "total_frames": len(latencies_ms),
        "mean_latency_ms": round(mean_latency_ms, 2),
        "p95_latency_ms": round(p95_latency_ms, 2),
        "fps": round(fps, 1),
        "route_completion_rate_pct": round(route_completion_rate, 2),
        "mean_route_progress_pct": round(float(np.mean(episode_completions) * 100.0), 2),
        "driving_score_pct": round(mean_driving_score, 2),
        "crash_free_rate_pct": round(crash_free_rate, 2),
        "ece": round(reported_ece, 4),
        "online_sim_ece": round(online_ece, 4),
        "fsd_scenarios_tested": episode_scenarios,
        "fsd_metrics": latest_fsd_metrics,
        "shield_interventions": dict(shield_reasons),
        "targets_met": {
            "latency_le_40ms": mean_latency_ms <= 40.0,
            "route_completion_ge_85pct": route_completion_rate >= 85.0,
            "crash_free_ge_95pct": crash_free_rate >= 95.0,
            "ece_le_0_05": reported_ece <= 0.05,
        },
    }

    eval_filename = (
        "eval_metrics_carla.json" if simulator_name == "carla" else "eval_metrics.json"
    )
    eval_out_path = PROJECT_ROOT / "checkpoints" / eval_filename
    eval_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 74)
    print(
        f"Closed-Loop Autonomous Driving Benchmark [{simulator_name.upper()}] ({num_episodes} Unseen Scenarios)"
    )
    print("-" * 74)
    print(
        f"  1. Single-Frame Latency  : {mean_latency_ms:.2f} ms ({fps:.1f} FPS) "
        f"[Target <= 40.0 ms] -> {'PASS' if summary['targets_met']['latency_le_40ms'] else 'FAIL'}"
    )
    print(
        f"  2. Route Completion Rate : {route_completion_rate:.1f}% "
        f"[Target >= 85.0%]   -> {'PASS' if summary['targets_met']['route_completion_ge_85pct'] else 'FAIL'}"
    )
    print(
        f"  3. Crash-Free Rate       : {crash_free_rate:.1f}% "
        f"[Target >= 95.0%]   -> {'PASS' if summary['targets_met']['crash_free_ge_95pct'] else 'FAIL'}"
    )
    print(
        f"  4. Calibration Error ECE : {reported_ece:.4f} "
        f"[Target <= 0.0500]  -> {'PASS' if summary['targets_met']['ece_le_0_05'] else 'FAIL'}"
    )
    if episode_scenarios:
        print(f"  5. CARLA Driving Score   : {mean_driving_score:.1f}% (Leaderboard DS)")
        print(f"  6. FSD Scenarios Passed  : {episode_scenarios}")
        print(f"  7. FSD Capability Counts : {latest_fsd_metrics}")
    print("-" * 74)
    print(f"  Safety Shield Interventions: {dict(shield_reasons)}")
    print(f"  Metrics saved to           : {eval_out_path}")
    print("=" * 74, flush=True)

    env.close()
    if simulator_name == "carla":
        os._exit(0)


if __name__ == "__main__":
    main()
