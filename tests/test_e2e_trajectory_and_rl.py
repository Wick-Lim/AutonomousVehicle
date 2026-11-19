"""Unit & Integration Tests for Track 1 (Obstacle-Aware E2E + PDMS + Leaderboard) & Track 3 (PPO/GRPO RL)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.benchmarks.leaderboard_agent import DINOv2LayaLeaderboardAgent, get_entry_point
from src.benchmarks.pdms_evaluator import (
    compute_episode_pdms,
    compute_pdms_from_episode_summary,
)
from src.control.e2e_trajectory_tracker import PureE2ETrajectoryTracker
from src.models.laya_pilot import AutonomousDrivingSystem
from src.models.spatial_projector import SpatialProjector
from src.rl.closed_loop_rl import (
    AdapterRLPolicyWrapper,
    CachedFeatureRolloutBuffer,
    ClosedLoopRLTrainer,
    FrozenReferenceAdapter,
)
from src.rl.rewards import ClosedLoopDrivingReward
from src.sim.e2e_dataset import (
    extract_obstacles_ego_left_frame,
    project_future_trajectory_ego_frame,
)
from src.sim.e2e_losses import (
    compute_collision_repulsion_loss,
    compute_obstacle_aware_e2e_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestTrack1AndTrack3(unittest.TestCase):
    """Automated 8-test verification suite for Track 1 & Track 3."""

    def test_temporal_gate_receives_nonzero_gradient_after_enable(self) -> None:
        """1. enable_temporal_learning() preserves Step-0 identity while unlocking temporal gradients."""
        torch.manual_seed(42)
        proj = SpatialProjector(dropout=0.0)
        p_prev = torch.randn(2, 392, 768)
        p_cur = torch.randn(2, 392, 768)

        # Verify Step-0 identity before & after enable_temporal_learning()
        with torch.no_grad():
            v_prev = proj(p_prev, use_temporal=False)
            out_before = proj(p_cur, elapsed_steps=2, prev_visual_tokens=v_prev)

        unlocked = proj.enable_temporal_learning(gain=0.08)
        self.assertTrue(unlocked)
        self.assertAlmostEqual(float(proj.temporal_gate.item()), 0.0, places=7)

        v_prev_det = proj(p_prev, use_temporal=False).detach()
        out_step0 = proj(p_cur, elapsed_steps=2, prev_visual_tokens=v_prev_det)
        max_diff = float((out_step0.detach() - out_before).abs().max().item())
        self.assertLess(max_diff, 1e-6, "Step-0 output must remain exact identity V_t + 0")

        # Step 0 backward: temporal_gate MUST immediately receive non-zero gradient!
        opt = torch.optim.Adam(proj.parameters(), lr=1e-2)
        opt.zero_grad()
        loss0 = out_step0.pow(2).mean()
        loss0.backward()
        self.assertIsNotNone(proj.temporal_gate.grad)
        self.assertGreater(float(proj.temporal_gate.grad.abs().item()), 1e-8)
        opt.step()

        # Step 1 backward: now both temporal_gate AND temporal_proj.weight receive non-zero gradients!
        opt.zero_grad()
        v_prev_det1 = proj(p_prev, use_temporal=False).detach()
        out_step1 = proj(p_cur, elapsed_steps=2, prev_visual_tokens=v_prev_det1)
        loss1 = out_step1.pow(2).mean()
        loss1.backward()
        self.assertGreater(float(proj.temporal_gate.grad.abs().item()), 1e-8)
        self.assertIsNotNone(proj.temporal_proj.weight.grad)
        self.assertGreater(float(proj.temporal_proj.weight.grad.abs().max().item()), 1e-8)

    def test_pure_neural_and_obstacle_aware_e2e_zero_param_increase(self) -> None:
        """2. Switching to pure_neural + obstacle_aware_e2e preserves EXACT 4,232,991 trainable params."""
        sys_m = AutonomousDrivingSystem(use_mock=True)
        n_default = sum(p.numel() for p in sys_m.parameters() if p.requires_grad)
        self.assertEqual(n_default, 4_232_991)

        sys_m.set_policy_modes(
            decision_mode="pure_neural",
            trajectory_mode="obstacle_aware_e2e",
        )
        sys_m.enable_temporal_learning(gain=0.08)
        n_e2e = sum(p.numel() for p in sys_m.parameters() if p.requires_grad)
        self.assertEqual(n_e2e, 4_232_991)

        # Verify pure_neural outputs match mlp_*_logits identically (zero mock_tel_readout bias)
        img = torch.rand(2, 3, 196, 392)
        tel_a = {"speed_kmh": 35.0, "lead_distance_m": 60.0, "traffic_light": "green"}
        tel_b = {"speed_kmh": 35.0, "lead_distance_m": 5.0, "traffic_light": "red"}
        out = sys_m(img, [tel_a, tel_b], aux_use_prior=False)
        self.assertTrue(torch.allclose(out["maneuver_logits"], out["mlp_maneuver_logits"], atol=1e-6))
        self.assertTrue(torch.allclose(out["speed_logits"], out["mlp_speed_logits"], atol=1e-6))
        self.assertTrue(torch.allclose(out["aeb_logits"], out["mlp_aeb_logits"], atol=1e-6))
        self.assertEqual(tuple(out["ctx_feat"].shape), (2, 1024))

    def test_obstacle_aware_e2e_monotonic_x_and_full_lateral_range(self) -> None:
        """3. obstacle_aware_e2e guarantees monotonic forward x and full +/-4.5m lateral range."""
        sys_m = AutonomousDrivingSystem(use_mock=True)
        sys_m.set_policy_modes(
            decision_mode="pure_neural",
            trajectory_mode="obstacle_aware_e2e",
        )
        # Stress extreme head biases to verify physical bounds & strict monotonicity
        with torch.no_grad():
            sys_m.laya.trajectory_head[-1].bias.zero_()
            # Force large positive lateral bias on all 6 waypoints
            for k in range(6):
                sys_m.laya.trajectory_head[-1].bias[3 * k + 1] = 4.0
        img = torch.rand(2, 3, 196, 392)
        tel = [{"speed_kmh": 40.0}, {"speed_kmh": 15.0}]
        traj_pos = sys_m(img, tel)["predicted_trajectory"]  # (2, 6, 3)

        self.assertEqual(tuple(traj_pos.shape), (2, 6, 3))
        dx = traj_pos[:, 1:, 0] - traj_pos[:, :-1, 0]
        self.assertTrue(bool((dx >= 0.0).all().item()), "x_k must be monotonically non-decreasing")
        self.assertTrue(bool((traj_pos[:, :, 0] >= 0.0).all().item()))
        self.assertGreater(float(traj_pos[:, -1, 1].mean().item()), 4.0)
        self.assertLessEqual(float(traj_pos[:, :, 1].abs().max().item()), 4.5 + 1e-5)

        # Force large negative lateral bias -> full negative bypass range
        with torch.no_grad():
            for k in range(6):
                sys_m.laya.trajectory_head[-1].bias[3 * k + 1] = -4.0
        traj_neg = sys_m(img, tel)["predicted_trajectory"]
        self.assertLess(float(traj_neg[:, -1, 1].mean().item()), -4.0)

    def test_differentiable_collision_repulsion_pushes_trajectory_away(self) -> None:
        """4. L_coll gradient pushes trajectory laterally AWAY from an overlapping obstacle."""
        # Ego trajectory near y = +0.4m, stationary obstacle at x = 15m, y = 0.0m -> push toward +y (away!)
        pred_traj = torch.tensor(
            [[[5.0, 0.4, 36.0], [10.0, 0.4, 36.0], [15.0, 0.4, 36.0],
              [20.0, 0.4, 36.0], [25.0, 0.4, 36.0], [30.0, 0.4, 36.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        # Obstacle at x=15m, y=0.0m, stationary (vx=0, vy=0)
        obs = torch.tensor(
            [[[15.0, 0.0, 0.0, 0.0, 2.35, 0.95, 0.0]]],
            dtype=torch.float32,
        )
        mask = torch.tensor([[1.0]], dtype=torch.float32)

        loss = compute_collision_repulsion_loss(pred_traj, obs, mask)
        self.assertGreater(float(loss.item()), 0.0)
        loss.backward()
        assert pred_traj.grad is not None
        # Since y=+0.4 > obs_y=0.0, minimizing loss (-grad) must INCREASE y => dL/dy < 0!
        grad_y_at_obs = float(pred_traj.grad[0, 2, 1].item())
        self.assertLess(grad_y_at_obs, -1e-4, "Collision repulsion must push y away from obstacle")

        # Verify retrospective projection & obstacle frame helper sign conventions
        poses = [(float(t) * 1.0, 0.0, 0.0) for t in range(35)]
        spds = [36.0] * 35
        trajs, valid = project_future_trajectory_ego_frame(poses, spds, step_dt=0.1)
        self.assertEqual(len(trajs), 35)
        self.assertAlmostEqual(trajs[0][0][0], 5.0, delta=0.1)
        self.assertEqual(valid[0], 6)
        obs_left = extract_obstacles_ego_left_frame(
            [{"x_m": 15.0, "y_m": 2.0, "vx_rel_mps": -5.0, "vy_rel_mps": -1.0, "category": "vehicle"}],
            ego_speed_kmh=36.0,
        )
        # y_m=+2.0 (right) -> y_left=-2.0; vx_abs = -5 + 10 = +5.0; vy_left = +1.0
        self.assertAlmostEqual(obs_left[0]["y"], -2.0, places=3)
        self.assertAlmostEqual(obs_left[0]["vx"], 5.0, places=3)
        self.assertAlmostEqual(obs_left[0]["vy"], 1.0, places=3)

    def test_pure_e2e_trajectory_tracker_zero_hd_map_dependency(self) -> None:
        """5. PureE2ETrajectoryTracker is strictly invariant to HD-map fields and follows +y=LEFT."""
        t1 = PureE2ETrajectoryTracker()
        t2 = PureE2ETrajectoryTracker()

        # Left-curving neural trajectory (+y = LEFT)
        left_dec = {
            "maneuver": "change_left",
            "speed_level": 2,
            "emergency_brake": 0.05,
            "predicted_trajectory": [
                [4.0, 0.5, 40.0],
                [8.0, 1.4, 40.0],
                [12.0, 2.2, 40.0],
                [16.0, 2.8, 40.0],
                [20.0, 3.0, 40.0],
                [24.0, 3.0, 40.0],
            ],
        }
        tel_clean = {
            "speed_kmh": 30.0,
            "lane_offset_m": 0.0,
            "heading_error": 0.0,
            "road_curvature": 0.0,
            "lead_distance_m": 60.0,
        }
        tel_corrupted_hd = {
            "speed_kmh": 30.0,
            "lane_offset_m": 2.9,
            "heading_error": -0.45,
            "road_curvature": -0.85,
            "lead_distance_m": 2.0,
            "traffic_light": "red",
        }
        (s1, tb1), info1 = t1.compute_control(left_dec, telemetry=tel_clean, dt=0.05)
        (s2, tb2), info2 = t2.compute_control(left_dec, telemetry=tel_corrupted_hd, dt=0.05)

        self.assertAlmostEqual(s1, s2, places=7)
        self.assertAlmostEqual(tb1, tb2, places=7)
        self.assertGreater(s1, 0.05, "+y waypoint must produce +steering (TURN LEFT)")
        self.assertFalse(info1["shield_triggered"])
        self.assertEqual(info2["shield_reason"], "pure_e2e_no_shield")
        # CARLA steer sign must be negated (+steer = RIGHT in CARLA)
        self.assertLess(info1["carla_control"]["steer"], 0.0)

    def test_ppo_and_grpo_rl_update_step_and_kl_anchor(self) -> None:
        """6. PPO & GRPO 1-iteration updates produce finite losses, non-zero updates, and v1.2 ckpt."""
        for algo in ("ppo", "grpo"):
            torch.manual_seed(123)
            sys_m = AutonomousDrivingSystem(use_mock=True)
            sys_m.set_policy_modes(
                decision_mode="pure_neural",
                trajectory_mode="obstacle_aware_e2e",
            )
            sys_m.enable_temporal_learning(gain=0.08)
            self.assertEqual(
                sum(p.numel() for p in sys_m.parameters() if p.requires_grad),
                4_232_991,
            )

            ref_adapter = FrozenReferenceAdapter(sys_m)
            policy = AdapterRLPolicyWrapper(sys_m)
            trainer = ClosedLoopRLTrainer(
                policy=policy,
                algo=algo,  # type: ignore[arg-type]
                lr_actor=1e-3,
                lr_critic=1e-3,
                beta_kl=0.05,
            )
            buf = CachedFeatureRolloutBuffer()
            rew_fn = ClosedLoopDrivingReward(dt=0.10)

            policy.eval()
            # Populate 2 episodes x 4 steps in same group (supports both PPO and GRPO)
            for ep in range(2):
                rew_fn.reset()
                prev_pol = None
                prev_ref = None
                for s in range(4):
                    img = torch.rand(1, 3, 196, 392)
                    patch = sys_m.dino(img)
                    tel = {
                        "speed_kmh": 32.0 + 3.0 * ep,
                        "lane_offset_m": 0.1 * (s - 1),
                        "lead_distance_m": 30.0,
                        "lead_speed_kmh": 30.0,
                        "nav_command": "go_straight",
                    }
                    act = policy.act(patch, tel, prev_visual_tokens=prev_pol, deterministic=False)
                    ref = ref_adapter.forward_ref(sys_m, patch, tel, prev_visual_tokens=prev_ref)
                    prev_pol = act["visual_tokens"]
                    prev_ref = ref["visual_tokens"]

                    r_step, _ = rew_fn.compute(
                        tel,
                        {
                            "route_completion": 0.1 * (s + 1),
                            "crash": (ep == 0 and s == 3),
                            "arrive_dest": (ep == 1 and s == 3),
                        },
                    )
                    buf.add(
                        patch_tokens=patch,
                        is_episode_start=(s == 0),
                        telemetry=tel,
                        traj_act=act["traj_act"],
                        man_act=act["man_act"],
                        spd_act=act["spd_act"],
                        aeb_act=act["aeb_act"],
                        old_log_prob=float(act["log_prob"][0].item()),
                        value=float(act["value"][0].item()),
                        reward=r_step,
                        done=(s == 3),
                        ref_traj_mean=ref["traj_mean"],
                        ref_man_logits=ref["man_logits"],
                        ref_spd_logits=ref["spd_logits"],
                        ref_aeb_logits=ref["aeb_logits"],
                        episode_id=ep,
                        group_id=1,
                    )

            w_before = sys_m.laya.trajectory_head[-1].weight.detach().clone()
            stats = trainer.update(buf, epochs=2, minibatch_size=4)
            w_after = sys_m.laya.trajectory_head[-1].weight.detach().clone()

            self.assertTrue(np.isfinite(stats["policy_loss"]))
            self.assertTrue(np.isfinite(stats["kl_div"]))
            self.assertGreaterEqual(stats["kl_div"], 0.0)
            self.assertLess(stats["kl_div"], 1.5, "Dropout must stay off so KL starts near 0")
            self.assertGreater(float((w_after - w_before).abs().max().item()), 1e-7)

            # Verify v1.2 save/load round-trip preserves exact modes & weights
            with tempfile.TemporaryDirectory() as tmpd:
                ckpt_file = Path(tmpd) / f"rl_{algo}.pt"
                sys_m.save_adapter_checkpoint(
                    ckpt_file, epoch=1, val_metrics=stats, training_stage=f"stage2_rl_{algo}"
                )
                sys_reloaded = AutonomousDrivingSystem(use_mock=True)
                meta = sys_reloaded.load_adapter_checkpoint(ckpt_file)
                self.assertEqual(meta["schema_version"], "1.2")
                self.assertEqual(sys_reloaded.laya.decision_mode, "pure_neural")
                self.assertEqual(sys_reloaded.laya.trajectory_mode, "obstacle_aware_e2e")
                self.assertEqual(
                    sum(p.numel() for p in sys_reloaded.parameters() if p.requires_grad),
                    4_232_991,
                )

    def test_pdms_formula_and_leaderboard_agent_interface(self) -> None:
        """7. NAVSIM PDMS boundary math & CARLA Leaderboard 2.0 AutonomousAgent API contract."""
        perfect = compute_episode_pdms(1.0, 1.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(perfect["pdms"], 1.0, places=6)

        crashed = compute_episode_pdms(0.0, 1.0, 0.9, 1.0, 1.0)
        self.assertAlmostEqual(crashed["pdms"], 0.0, places=6)

        # Weighted check: 1 * 1 * (5*0.8 + 5*1.0 + 2*0.5)/12 = 10/12 = 0.8333
        mid = compute_episode_pdms(1.0, 1.0, 0.8, 1.0, 0.5)
        self.assertAlmostEqual(mid["pdms"], 10.0 / 12.0, places=4)

        from_sum = compute_pdms_from_episode_summary(
            {
                "vehicle_collisions": 0,
                "pedestrian_collisions": 0,
                "off_road_events": 0,
                "lane_invasions": 0,
                "route_progress_pct": 100.0,
                "min_ttc_s": 2.4,
                "comfort_rate_pct": 96.0,
            }
        )
        self.assertGreater(from_sum["pdms"], 0.99)

        # Leaderboard 2.0 / Bench2Drive agent contract verification
        self.assertEqual(get_entry_point(), "DINOv2LayaLeaderboardAgent")
        with tempfile.TemporaryDirectory() as tmpd:
            cfg_p = Path(tmpd) / "agent_cfg.json"
            cfg_p.write_text(
                json.dumps(
                    {
                        "use_mock": True,
                        "control_mode": "pure_e2e_trajectory",
                        "decision_mode": "pure_neural",
                        "trajectory_mode": "obstacle_aware_e2e",
                    }
                ),
                encoding="utf-8",
            )
            agent = DINOv2LayaLeaderboardAgent()
            agent.setup(str(cfg_p))
            sensors = agent.sensors()
            sensor_ids = {s["id"] for s in sensors}
            self.assertIn("rgb_front", sensor_ids)
            self.assertIn("gps", sensor_ids)
            self.assertIn("imu", sensor_ids)
            self.assertIn("speed", sensor_ids)

            agent.set_global_plan(
                [({"lat": 0.0, "lon": 0.0}, "LANEFOLLOW"), ({"lat": 0.001, "lon": 0.0}, "RIGHT")],
                [],
            )
            ctrl = agent.run_step(
                {
                    "rgb_front": (0, np.zeros((196, 392, 4), dtype=np.uint8)),
                    "speed": (0, {"speed": 8.0}),
                    "imu": (0, [0.0, 0.0, 9.81, 0.0, 0.0, 0.0, 0.0]),
                    "gps": (0, [0.0, 0.0, 0.0]),
                    "radar_front": (0, np.zeros((0, 4), dtype=np.float32)),
                },
                timestamp=0.05,
            )
            self.assertTrue(-1.0 <= float(ctrl.steer) <= 1.0)
            self.assertTrue(0.0 <= float(ctrl.throttle) <= 1.0)
            self.assertTrue(0.0 <= float(ctrl.brake) <= 1.0)
            agent.destroy()

    def test_cli_e2e_smoke_and_legacy_regression(self) -> None:
        """8. CLI subcommands train-e2e, finetune-rl, e2e-bench execute cleanly in mock smoke mode."""
        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd)
            data_dir = tmp / "data_e2e_smoke"
            s1_ckpt = tmp / "s1.pt"
            s2_ckpt = tmp / "s2.pt"
            s2_temp = tmp / "s2_temp.pt"
            report_p = tmp / "bench.json"

            p1 = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.cli",
                    "train-e2e",
                    "--mock",
                    "--collect",
                    "--num-frames",
                    "40",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "8",
                    "--data-dir",
                    str(data_dir),
                    "--output-ckpt",
                    str(s1_ckpt),
                    "--history-json",
                    str(tmp / "s1_hist.json"),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(p1.returncode, 0, f"train-e2e failed:\n{p1.stdout}\n{p1.stderr}")
            self.assertTrue(s1_ckpt.is_file())

            p2 = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.cli",
                    "finetune-rl",
                    "--mock",
                    "--algo",
                    "ppo",
                    "--init-ckpt",
                    str(s1_ckpt),
                    "--output-ckpt",
                    str(s2_ckpt),
                    "--temp-ckpt",
                    str(s2_temp),
                    "--data-dir",
                    str(data_dir),
                    "--total-steps",
                    "32",
                    "--rollout-steps",
                    "16",
                    "--update-epochs",
                    "1",
                    "--minibatch-size",
                    "8",
                    "--max-ep-steps",
                    "20",
                    "--history-json",
                    str(tmp / "s2_hist.json"),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(p2.returncode, 0, f"finetune-rl failed:\n{p2.stdout}\n{p2.stderr}")
            self.assertTrue(s2_ckpt.is_file())

            p3 = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.cli",
                    "e2e-bench",
                    "--mock",
                    "--simulator",
                    "metadrive",
                    "--data-dir",
                    str(data_dir),
                    "--stage1-ckpt",
                    str(s1_ckpt),
                    "--stage2-ckpt",
                    str(s2_ckpt),
                    "--stage2-temp",
                    str(s2_temp),
                    "--metadrive-episodes",
                    "2",
                    "--max-steps",
                    "25",
                    "--output-json",
                    str(report_p),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(p3.returncode, 0, f"e2e-bench failed:\n{p3.stdout}\n{p3.stderr}")
            self.assertTrue(report_p.is_file())
            rep = json.loads(report_p.read_text(encoding="utf-8"))
            self.assertTrue(rep["parameter_invariant_ok"])
            self.assertIn("stage2_e2e_rl", rep["metadrive_unseen_5000"])


if __name__ == "__main__":
    unittest.main()
