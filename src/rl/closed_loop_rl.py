"""Closed-Loop Reinforcement Learning Engine: PPO (GAE) + GRPO + Stage-1 IL-KL Anchor.

Architecture invariants:
  - `AutonomousDrivingSystem` retains EXACTLY `4,232,991` trainable parameters.
  - Exploration parameter `traj_log_std` (`(6, 3)`) and PPO Critic MLP (`value_head`,
    `1024 -> 256 -> 1`) live strictly inside `AdapterRLPolicyWrapper` and are NEVER
    saved into or attached to `AutonomousDrivingSystem`.
  - Frozen DINOv2 `patch_tokens` `(392, 768)` are cached in FP16 during closed-loop
    rollouts so multi-epoch PPO/GRPO updates skip DINOv2 completely.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal

from src.models.laya_pilot import AutonomousDrivingSystem


def _sanitize_executed_trajectory(traj_sample: torch.Tensor) -> torch.Tensor:
    """Project sampled `(..., 6, 3)` trajectory onto physical domain for Pure-Pursuit control."""
    out = traj_sample.clone()
    x = out[..., 0].clamp(min=0.0, max=60.0)
    # Enforce monotonic forward x
    x_mono, _ = torch.cummax(x, dim=-1)
    y = out[..., 1].clamp(min=-4.5, max=4.5)
    v = out[..., 2].clamp(min=0.0, max=70.0)
    return torch.stack([x_mono, y, v], dim=-1)


class AdapterRLPolicyWrapper(nn.Module):
    """Wraps `AutonomousDrivingSystem` (4,232,991 params) with training-only RL head & log-std."""

    def __init__(
        self,
        system: AutonomousDrivingSystem,
        init_log_std_x: float = -1.35,
        init_log_std_y: float = -2.00,
        init_log_std_v: float = 0.35,
    ) -> None:
        super().__init__()
        self.system = system
        self.system.set_policy_modes(
            decision_mode="pure_neural",
            trajectory_mode="obstacle_aware_e2e",
        )

        # Training-only exploration stds (never part of AutonomousDrivingSystem)
        init_std_mat = torch.zeros((6, 3), dtype=torch.float32)
        init_std_mat[:, 0] = float(init_log_std_x)  # ~0.35 m longitudinal exploration
        init_std_mat[:, 1] = float(init_log_std_y)  # ~0.20 m lateral exploration
        init_std_mat[:, 2] = float(init_log_std_v)  # ~1.73 km/h speed exploration
        self.traj_log_std = nn.Parameter(init_std_mat)
        self.register_buffer("ref_traj_log_std", init_std_mat.detach().clone())

        # Training-only lightweight Critic MLP for PPO (1024 -> 256 -> 1)
        embed_dim = int(system.laya.embed_dim)
        self.value_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def get_traj_std(self, device: torch.device) -> torch.Tensor:
        """Return bounded `(1, 6, 3)` trajectory exploration standard deviation."""
        clamped = torch.stack(
            [
                self.traj_log_std[:, 0].clamp(-2.5, 0.2),
                self.traj_log_std[:, 1].clamp(-2.8, -0.35),
                self.traj_log_std[:, 2].clamp(-1.0, 1.6),
            ],
            dim=-1,
        )
        return torch.exp(clamped).to(device=device, dtype=torch.float32).unsqueeze(0)

    def get_ref_traj_std(self, device: torch.device) -> torch.Tensor:
        return torch.exp(self.ref_traj_log_std).to(device=device, dtype=torch.float32).unsqueeze(0)

    def actor_parameters(self) -> List[nn.Parameter]:
        """Trainable adapter parameters (4,232,991) + training-only `traj_log_std`."""
        return [p for p in self.system.parameters() if p.requires_grad] + [self.traj_log_std]

    def critic_parameters(self) -> List[nn.Parameter]:
        return list(self.value_head.parameters())

    def _build_distributions(
        self, out: Dict[str, Any]
    ) -> Tuple[Normal, Categorical, Categorical, Bernoulli]:
        traj_mean = out["predicted_trajectory"].float()  # (B, 6, 3)
        traj_std = self.get_traj_std(traj_mean.device).expand_as(traj_mean)
        traj_dist = Normal(traj_mean, traj_std)

        man_dist = Categorical(logits=out["maneuver_logits"].float())
        spd_dist = Categorical(logits=out["speed_logits"].float())
        aeb_dist = Bernoulli(logits=out["aeb_logits"].float().squeeze(-1))
        return traj_dist, man_dist, spd_dist, aeb_dist

    @torch.no_grad()
    def act(
        self,
        patch_tokens: torch.Tensor,
        telemetry: Dict[str, Any],
        prev_visual_tokens: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        use_amp: bool = False,
    ) -> Dict[str, Any]:
        """Sample (or take greedy mean of) joint trajectory + discrete decision action."""
        device = patch_tokens.device
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = self.system.forward_from_patch_tokens(
                patch_tokens=patch_tokens,
                telemetries=[telemetry],
                elapsed_steps=1,
                prev_visual_tokens=prev_visual_tokens,
                aux_use_prior=False,
            )

        traj_dist, man_dist, spd_dist, aeb_dist = self._build_distributions(out)
        if deterministic:
            traj_act = traj_dist.mean
            man_act = out["maneuver_logits"].float().argmax(dim=-1)
            spd_act = out["speed_logits"].float().argmax(dim=-1)
            aeb_act = (torch.sigmoid(out["aeb_logits"].float().squeeze(-1)) >= 0.5).float()
        else:
            traj_act = traj_dist.sample()
            man_act = man_dist.sample()
            spd_act = spd_dist.sample()
            aeb_act = aeb_dist.sample()

        # Scaled trajectory log-prob (mean over 18 coords * 6.0) keeps discrete & continuous policy gradients balanced
        lp_traj = traj_dist.log_prob(traj_act).mean(dim=(1, 2)) * 6.0
        lp_man = man_dist.log_prob(man_act)
        lp_spd = spd_dist.log_prob(spd_act)
        lp_aeb = aeb_dist.log_prob(aeb_act)
        joint_lp = lp_traj + lp_man + lp_spd + lp_aeb

        value = self.value_head(out["ctx_feat"].float()).squeeze(-1)
        exec_traj = _sanitize_executed_trajectory(traj_act)

        return {
            "traj_act": traj_act.detach(),
            "exec_traj": exec_traj.detach(),
            "man_act": man_act.detach(),
            "spd_act": spd_act.detach(),
            "aeb_act": aeb_act.detach(),
            "log_prob": joint_lp.detach(),
            "value": value.detach(),
            "visual_tokens": out["visual_tokens"].detach(),
            "traj_mean": out["predicted_trajectory"].float().detach(),
            "man_logits": out["maneuver_logits"].float().detach(),
            "spd_logits": out["speed_logits"].float().detach(),
            "aeb_logits": out["aeb_logits"].float().detach(),
            "aeb_prob": out["emergency_brake"].float().detach(),
        }

    def evaluate_actions(
        self,
        patch_tokens: torch.Tensor,
        prev_patch_tokens: torch.Tensor,
        telemetries: Sequence[Dict[str, Any]],
        traj_act: torch.Tensor,
        man_act: torch.Tensor,
        spd_act: torch.Tensor,
        aeb_act: torch.Tensor,
        ref_traj_mean: torch.Tensor,
        ref_man_logits: torch.Tensor,
        ref_spd_logits: torch.Tensor,
        ref_aeb_logits: torch.Tensor,
        use_amp: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Re-evaluate cached rollout batch for PPO/GRPO loss, Critic value, entropy & analytical KL."""
        device = patch_tokens.device
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                prev_vis = self.system.projector(prev_patch_tokens, use_temporal=False).detach()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = self.system.forward_from_patch_tokens(
                patch_tokens=patch_tokens,
                telemetries=telemetries,
                elapsed_steps=1,
                prev_visual_tokens=prev_vis,
                aux_use_prior=False,
            )

        traj_dist, man_dist, spd_dist, aeb_dist = self._build_distributions(out)
        lp_traj = traj_dist.log_prob(traj_act).mean(dim=(1, 2)) * 6.0
        lp_man = man_dist.log_prob(man_act)
        lp_spd = spd_dist.log_prob(spd_act)
        lp_aeb = aeb_dist.log_prob(aeb_act)
        joint_lp = lp_traj + lp_man + lp_spd + lp_aeb

        entropy = (
            traj_dist.entropy().mean(dim=(1, 2))
            + man_dist.entropy()
            + spd_dist.entropy()
            + aeb_dist.entropy()
        )
        values = self.value_head(out["ctx_feat"].float()).squeeze(-1)

        # Closed-form analytical KL divergence D_KL(pi_theta || pi_ref_stage1)
        ref_std = self.get_ref_traj_std(device).expand_as(ref_traj_mean)
        ref_traj_dist = Normal(ref_traj_mean, ref_std)
        kl_traj = torch.distributions.kl_divergence(traj_dist, ref_traj_dist).mean(dim=(1, 2)) * 3.0

        ref_man_dist = Categorical(logits=ref_man_logits)
        ref_spd_dist = Categorical(logits=ref_spd_logits)
        ref_aeb_dist = Bernoulli(logits=ref_aeb_logits.squeeze(-1))
        kl_man = torch.distributions.kl_divergence(man_dist, ref_man_dist)
        kl_spd = torch.distributions.kl_divergence(spd_dist, ref_spd_dist)
        kl_aeb = torch.distributions.kl_divergence(aeb_dist, ref_aeb_dist)
        kl_total = kl_traj + kl_man + kl_spd + kl_aeb

        return {
            "log_prob": joint_lp,
            "entropy": entropy,
            "values": values,
            "kl_div": kl_total,
        }


class FrozenReferenceAdapter(nn.Module):
    """Lightweight (~16 MB) frozen Stage-1 adapter snapshot reusing shared frozen Laya backbone."""

    def __init__(self, system: AutonomousDrivingSystem) -> None:
        super().__init__()
        self.projector = copy.deepcopy(system.projector).eval()
        self.maneuver_head = copy.deepcopy(system.laya.maneuver_head).eval()
        self.speed_head = copy.deepcopy(system.laya.speed_head).eval()
        self.aeb_head = copy.deepcopy(system.laya.aeb_head).eval()
        self.trajectory_head = copy.deepcopy(system.laya.trajectory_head).eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward_ref_batch(
        self,
        shared_system: AutonomousDrivingSystem,
        patch_tokens: torch.Tensor,
        prev_patch_tokens: torch.Tensor,
        telemetries: Sequence[Dict[str, Any]],
        use_amp: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Batched Stage-1 reference policy forward pass reusing shared frozen Laya encoder."""
        laya = shared_system.laya
        bsz = patch_tokens.shape[0]
        device = patch_tokens.device
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            prev_vis = self.projector(prev_patch_tokens, use_temporal=False)
            vis = self.projector(
                patch_tokens,
                elapsed_steps=1,
                prev_visual_tokens=prev_vis,
            )
            inputs_embeds, attn_mask, marker_pos = laya._prepare_multimodal_inputs_embeds(
                vis, telemetries
            )
            b_idx = torch.arange(bsz, dtype=torch.long, device=device).unsqueeze(1).expand(-1, 9)
            if laya.use_mock:
                hidden = laya.encoder(inputs_embeds)
                marker_hidden = hidden[b_idx, marker_pos]
            else:
                out_enc = laya.laya_model.encoder(
                    inputs_embeds=inputs_embeds, attention_mask=attn_mask
                )
                hidden = out_enc.last_hidden_state
                if laya.laya_model.head is not None:
                    pad = ~attn_mask.bool() if attn_mask is not None else None
                    for layer in laya.laya_model.head.layers:
                        hidden = layer(hidden, src_key_padding_mask=pad)
                marker_hidden = hidden[b_idx, marker_pos]
                qtype_ids = torch.tensor(
                    [0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=torch.long, device=device
                )
                marker_hidden = marker_hidden + laya.laya_model.type_emb(qtype_ids).unsqueeze(0)

        ctx_feat = 0.5 * marker_hidden.mean(dim=1).float() + 0.5 * vis.mean(dim=1).float()
        man_logits = self.maneuver_head(ctx_feat).float()
        spd_logits = self.speed_head(ctx_feat).float()
        aeb_logits = self.aeb_head(ctx_feat).float()

        raw = self.trajectory_head(ctx_feat).float().reshape(bsz, 6, 3)
        v_k = 70.0 * torch.sigmoid(raw[:, :, 2])
        dx_k = F.softplus((v_k / 3.6) * 0.5 + 1.5 * torch.tanh(raw[:, :, 0]))
        x_k = torch.cumsum(dx_k, dim=1)
        y_k = 4.5 * torch.tanh(raw[:, :, 1])
        traj_mean = torch.stack([x_k, y_k, v_k], dim=-1)

        return {
            "visual_tokens": vis.detach(),
            "traj_mean": traj_mean.detach(),
            "man_logits": man_logits.detach(),
            "spd_logits": spd_logits.detach(),
            "aeb_logits": aeb_logits.detach(),
        }

    @torch.no_grad()
    def populate_buffer_refs(
        self,
        buffer: "CachedFeatureRolloutBuffer",
        shared_system: AutonomousDrivingSystem,
        device: torch.device,
        batch_size: int = 64,
        use_amp: bool = False,
    ) -> None:
        """Fill `ref_*` tensors in `buffer` via fast batched GPU passes (2x faster than per-step)."""
        n = len(buffer)
        buffer.ref_traj_means = []
        buffer.ref_man_logits = []
        buffer.ref_spd_logits = []
        buffer.ref_aeb_logits = []
        for start in range(0, n, max(1, int(batch_size))):
            end = min(n, start + int(batch_size))
            cur_p = torch.stack(buffer.patch_tokens_fp16[start:end], dim=0).to(
                device=device, dtype=torch.float32
            )
            prev_p = torch.stack(
                [
                    buffer.patch_tokens_fp16[buffer.prev_indices[i]]
                    if buffer.prev_indices[i] >= 0
                    else buffer.patch_tokens_fp16[i]
                    for i in range(start, end)
                ],
                dim=0,
            ).to(device=device, dtype=torch.float32)
            tels = buffer.telemetries[start:end]
            ref = self.forward_ref_batch(
                shared_system=shared_system,
                patch_tokens=cur_p,
                prev_patch_tokens=prev_p,
                telemetries=tels,
                use_amp=use_amp,
            )
            for k in range(end - start):
                buffer.ref_traj_means.append(ref["traj_mean"][k].cpu())
                buffer.ref_man_logits.append(ref["man_logits"][k].cpu())
                buffer.ref_spd_logits.append(ref["spd_logits"][k].cpu())
                buffer.ref_aeb_logits.append(ref["aeb_logits"][k].cpu())

    @torch.no_grad()
    def forward_ref(
        self,
        shared_system: AutonomousDrivingSystem,
        patch_tokens: torch.Tensor,
        telemetry: Dict[str, Any],
        prev_visual_tokens: Optional[torch.Tensor] = None,
        use_amp: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute exact Stage-1 reference policy outputs sharing frozen `shared_system.laya` encoder."""
        laya = shared_system.laya
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            vis = self.projector(
                patch_tokens,
                elapsed_steps=1,
                prev_visual_tokens=prev_visual_tokens,
            )
            inputs_embeds, attn_mask, marker_pos = laya._prepare_multimodal_inputs_embeds(
                vis, [telemetry]
            )
            if laya.use_mock:
                hidden = laya.encoder(inputs_embeds)
                b_idx = torch.zeros((1, 9), dtype=torch.long, device=vis.device)
                marker_hidden = hidden[b_idx, marker_pos]
            else:
                out_enc = laya.laya_model.encoder(
                    inputs_embeds=inputs_embeds, attention_mask=attn_mask
                )
                hidden = out_enc.last_hidden_state
                if laya.laya_model.head is not None:
                    pad = ~attn_mask.bool() if attn_mask is not None else None
                    for layer in laya.laya_model.head.layers:
                        hidden = layer(hidden, src_key_padding_mask=pad)
                b_idx = torch.zeros((1, 9), dtype=torch.long, device=vis.device)
                marker_hidden = hidden[b_idx, marker_pos]
                qtype_ids = torch.tensor(
                    [0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=torch.long, device=vis.device
                )
                marker_hidden = marker_hidden + laya.laya_model.type_emb(qtype_ids).unsqueeze(0)

        ctx_feat = 0.5 * marker_hidden.mean(dim=1).float() + 0.5 * vis.mean(dim=1).float()
        man_logits = self.maneuver_head(ctx_feat).float()
        spd_logits = self.speed_head(ctx_feat).float()
        aeb_logits = self.aeb_head(ctx_feat).float()

        raw = self.trajectory_head(ctx_feat).float().reshape(1, 6, 3)
        v_k = 70.0 * torch.sigmoid(raw[:, :, 2])
        dx_k = F.softplus((v_k / 3.6) * 0.5 + 1.5 * torch.tanh(raw[:, :, 0]))
        x_k = torch.cumsum(dx_k, dim=1)
        y_k = 4.5 * torch.tanh(raw[:, :, 1])
        traj_mean = torch.stack([x_k, y_k, v_k], dim=-1)

        return {
            "visual_tokens": vis.detach(),
            "traj_mean": traj_mean.detach(),
            "man_logits": man_logits.detach(),
            "spd_logits": spd_logits.detach(),
            "aeb_logits": aeb_logits.detach(),
        }


@dataclass
class CachedFeatureRolloutBuffer:
    """Memory-efficient rollout buffer storing FP16 DINOv2 patch tokens & reference logits."""

    patch_tokens_fp16: List[torch.Tensor] = field(default_factory=list)
    prev_indices: List[int] = field(default_factory=list)
    telemetries: List[Dict[str, Any]] = field(default_factory=list)
    traj_acts: List[torch.Tensor] = field(default_factory=list)
    man_acts: List[torch.Tensor] = field(default_factory=list)
    spd_acts: List[torch.Tensor] = field(default_factory=list)
    aeb_acts: List[torch.Tensor] = field(default_factory=list)
    old_log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    ref_traj_means: List[torch.Tensor] = field(default_factory=list)
    ref_man_logits: List[torch.Tensor] = field(default_factory=list)
    ref_spd_logits: List[torch.Tensor] = field(default_factory=list)
    ref_aeb_logits: List[torch.Tensor] = field(default_factory=list)
    group_ids: List[int] = field(default_factory=list)
    episode_ids: List[int] = field(default_factory=list)

    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None

    def clear(self) -> None:
        self.patch_tokens_fp16.clear()
        self.prev_indices.clear()
        self.telemetries.clear()
        self.traj_acts.clear()
        self.man_acts.clear()
        self.spd_acts.clear()
        self.aeb_acts.clear()
        self.old_log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.ref_traj_means.clear()
        self.ref_man_logits.clear()
        self.ref_spd_logits.clear()
        self.ref_aeb_logits.clear()
        self.group_ids.clear()
        self.episode_ids.clear()
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        patch_tokens: torch.Tensor,
        is_episode_start: bool,
        telemetry: Dict[str, Any],
        traj_act: torch.Tensor,
        man_act: torch.Tensor,
        spd_act: torch.Tensor,
        aeb_act: torch.Tensor,
        old_log_prob: float,
        value: float,
        reward: float,
        done: bool,
        ref_traj_mean: torch.Tensor,
        ref_man_logits: torch.Tensor,
        ref_spd_logits: torch.Tensor,
        ref_aeb_logits: torch.Tensor,
        episode_id: int = 0,
        group_id: int = 0,
    ) -> None:
        cur_idx = len(self.patch_tokens_fp16)
        self.patch_tokens_fp16.append(
            patch_tokens.detach().squeeze(0).to(device="cpu", dtype=torch.float16)
        )
        self.prev_indices.append(-1 if (is_episode_start or cur_idx == 0) else (cur_idx - 1))
        self.telemetries.append(dict(telemetry))
        self.traj_acts.append(traj_act.detach().squeeze(0).cpu().float())
        self.man_acts.append(man_act.detach().squeeze(0).cpu().long())
        self.spd_acts.append(spd_act.detach().squeeze(0).cpu().long())
        self.aeb_acts.append(aeb_act.detach().squeeze(0).cpu().float())
        self.old_log_probs.append(float(old_log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.ref_traj_means.append(ref_traj_mean.detach().squeeze(0).cpu().float())
        self.ref_man_logits.append(ref_man_logits.detach().squeeze(0).cpu().float())
        self.ref_spd_logits.append(ref_spd_logits.detach().squeeze(0).cpu().float())
        self.ref_aeb_logits.append(ref_aeb_logits.detach().squeeze(0).cpu().float())
        self.episode_ids.append(int(episode_id))
        self.group_ids.append(int(group_id))

    def compute_gae(
        self,
        gamma: float = 0.98,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE) for PPO."""
        n = len(self.rewards)
        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            non_terminal = 0.0 if self.dones[t] else 1.0
            next_val = last_value if (t == n - 1) else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            adv[t] = last_gae
        vals = np.asarray(self.values, dtype=np.float32)
        self.advantages = torch.from_numpy(adv)
        self.returns = torch.from_numpy(adv + vals)

    def compute_grpo_advantages(self) -> None:
        """Compute critic-free group-relative normalized advantages for GRPO."""
        n = len(self.rewards)
        if n == 0:
            self.advantages = torch.zeros(0, dtype=torch.float32)
            self.returns = torch.zeros(0, dtype=torch.float32)
            return

        # Compute total episode return per episode_id and group by group_id (shared seed)
        ep_returns: Dict[int, float] = {}
        ep_group: Dict[int, int] = {}
        for t in range(n):
            eid = self.episode_ids[t]
            ep_returns[eid] = ep_returns.get(eid, 0.0) + self.rewards[t]
            ep_group[eid] = self.group_ids[t]

        group_to_rets: Dict[int, List[float]] = {}
        for eid, ret in ep_returns.items():
            gid = ep_group[eid]
            group_to_rets.setdefault(gid, []).append(ret)

        ep_adv: Dict[int, float] = {}
        for eid, ret in ep_returns.items():
            grp_vals = group_to_rets[ep_group[eid]]
            mean_r = float(np.mean(grp_vals))
            std_r = float(np.std(grp_vals))
            ep_adv[eid] = (ret - mean_r) / (std_r + 1e-6)

        adv = np.array([ep_adv[self.episode_ids[t]] for t in range(n)], dtype=np.float32)
        rets = np.array([ep_returns[self.episode_ids[t]] for t in range(n)], dtype=np.float32)
        self.advantages = torch.from_numpy(adv)
        self.returns = torch.from_numpy(rets)


class ClosedLoopRLTrainer:
    """Unified PPO (GAE + Clipped Surrogate + KL) and GRPO (Group Relative + KL) trainer."""

    def __init__(
        self,
        policy: AdapterRLPolicyWrapper,
        algo: Literal["ppo", "grpo"] = "ppo",
        lr_actor: float = 3e-5,
        lr_critic: float = 1e-4,
        clip_eps: float = 0.15,
        beta_kl: float = 0.04,
        target_kl: float = 0.25,
        entropy_coef: float = 0.005,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.policy = policy
        self.algo = str(algo).lower()
        if self.algo not in ("ppo", "grpo"):
            raise ValueError(f"Unsupported RL algorithm '{algo}'. Expected 'ppo' or 'grpo'.")
        self.clip_eps = float(clip_eps)
        self.beta_kl = float(beta_kl)
        self.target_kl = float(target_kl)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.device = device or next(policy.parameters()).device

        param_groups: List[Dict[str, Any]] = [
            {"params": self.policy.actor_parameters(), "lr": float(lr_actor), "weight_decay": 1e-4}
        ]
        if self.algo == "ppo":
            param_groups.append(
                {
                    "params": self.policy.critic_parameters(),
                    "lr": float(lr_critic),
                    "weight_decay": 1e-4,
                }
            )
        self.optimizer = torch.optim.AdamW(param_groups)

    def update(
        self,
        buffer: CachedFeatureRolloutBuffer,
        epochs: int = 4,
        minibatch_size: int = 64,
        gamma: float = 0.98,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
        use_amp: bool = False,
    ) -> Dict[str, float]:
        """Execute PPO or GRPO multi-epoch mini-batch updates on cached rollout features."""
        n = len(buffer)
        if n == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "kl_div": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
                "beta_kl": self.beta_kl,
            }

        if self.algo == "ppo":
            buffer.compute_gae(gamma=gamma, gae_lambda=gae_lambda, last_value=last_value)
        else:
            buffer.compute_grpo_advantages()

        assert buffer.advantages is not None and buffer.returns is not None
        adv_all = buffer.advantages.clone().float()
        if adv_all.numel() > 1 and float(adv_all.std().item()) > 1e-6:
            adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
        ret_all = buffer.returns.clone().float()
        old_lp_all = torch.tensor(buffer.old_log_probs, dtype=torch.float32)

        # Keep frozen backbones AND all Dropout layers in eval() during RL updates
        # (standard PPO/GRPO invariant: log_pi(theta_old) == old_log_prob and KL(theta_ref) == 0.0)
        self.policy.train()
        self.policy.system.dino.eval()
        if not self.policy.system.laya.use_mock:
            self.policy.system.laya.laya_model.eval()
        else:
            self.policy.system.laya.encoder.eval()
        for mod in self.policy.modules():
            if isinstance(mod, nn.Dropout):
                mod.eval()

        eff_bs = max(1, min(int(minibatch_size), n))
        all_params = (
            self.policy.actor_parameters() + self.policy.critic_parameters()
            if self.algo == "ppo"
            else self.policy.actor_parameters()
        )

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "kl_div": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
        }
        n_updates = 0

        for _ep in range(max(1, int(epochs))):
            perm = torch.randperm(n).tolist()
            for start in range(0, n, eff_bs):
                idxs = perm[start : start + eff_bs]
                if not idxs:
                    continue

                cur_patch = torch.stack(
                    [buffer.patch_tokens_fp16[i] for i in idxs], dim=0
                ).to(device=self.device, dtype=torch.float32)
                prev_patch = torch.stack(
                    [
                        buffer.patch_tokens_fp16[buffer.prev_indices[i]]
                        if buffer.prev_indices[i] >= 0
                        else buffer.patch_tokens_fp16[i]
                        for i in idxs
                    ],
                    dim=0,
                ).to(device=self.device, dtype=torch.float32)

                tels = [buffer.telemetries[i] for i in idxs]
                traj_act = torch.stack([buffer.traj_acts[i] for i in idxs], dim=0).to(self.device)
                man_act = torch.stack([buffer.man_acts[i] for i in idxs], dim=0).to(self.device)
                spd_act = torch.stack([buffer.spd_acts[i] for i in idxs], dim=0).to(self.device)
                aeb_act = torch.stack([buffer.aeb_acts[i] for i in idxs], dim=0).to(self.device)

                ref_traj = torch.stack([buffer.ref_traj_means[i] for i in idxs], dim=0).to(self.device)
                ref_man = torch.stack([buffer.ref_man_logits[i] for i in idxs], dim=0).to(self.device)
                ref_spd = torch.stack([buffer.ref_spd_logits[i] for i in idxs], dim=0).to(self.device)
                ref_aeb = torch.stack([buffer.ref_aeb_logits[i] for i in idxs], dim=0).to(self.device)

                mb_adv = adv_all[idxs].to(self.device)
                mb_ret = ret_all[idxs].to(self.device)
                mb_old_lp = old_lp_all[idxs].to(self.device)

                ev = self.policy.evaluate_actions(
                    patch_tokens=cur_patch,
                    prev_patch_tokens=prev_patch,
                    telemetries=tels,
                    traj_act=traj_act,
                    man_act=man_act,
                    spd_act=spd_act,
                    aeb_act=aeb_act,
                    ref_traj_mean=ref_traj,
                    ref_man_logits=ref_man,
                    ref_spd_logits=ref_spd,
                    ref_aeb_logits=ref_aeb,
                    use_amp=use_amp,
                )

                log_ratio = (ev["log_prob"] - mb_old_lp).clamp(-6.0, 6.0)
                ratio = torch.exp(log_ratio)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                kl_mean = ev["kl_div"].mean()
                ent_mean = ev["entropy"].mean()

                if self.algo == "ppo":
                    val_loss = F.smooth_l1_loss(ev["values"], mb_ret, beta=2.0)
                else:
                    val_loss = torch.zeros((), device=self.device)

                total_loss = (
                    policy_loss
                    + self.beta_kl * kl_mean
                    - self.entropy_coef * ent_mean
                    + (self.value_coef * val_loss if self.algo == "ppo" else 0.0)
                )

                if not torch.isfinite(total_loss):
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                    clip_frac = float(((ratio - 1.0).abs() > self.clip_eps).float().mean().item())

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(val_loss.item())
                stats["kl_div"] += float(kl_mean.item())
                stats["entropy"] += float(ent_mean.item())
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                n_updates += 1

        denom = max(1, n_updates)
        res = {k: round(v / denom, 5) for k, v in stats.items()}

        # Adaptive KL anchor scaling to prevent policy collapse while allowing improvement
        if res["kl_div"] > 1.8 * self.target_kl:
            self.beta_kl = min(0.50, self.beta_kl * 1.25)
        elif res["kl_div"] < 0.4 * self.target_kl:
            self.beta_kl = max(0.01, self.beta_kl * 0.90)
        res["beta_kl"] = round(self.beta_kl, 5)
        return res


__all__ = [
    "AdapterRLPolicyWrapper",
    "CachedFeatureRolloutBuffer",
    "ClosedLoopRLTrainer",
    "FrozenReferenceAdapter",
]
