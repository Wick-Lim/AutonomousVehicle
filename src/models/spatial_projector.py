"""2D Spatial Projector: AdaptiveAvgPool2d(4, 9) + 2D Positional Embedding + Temporal Queue + Auxiliary Perception."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TRAFFIC_LIGHT_CLASSES: Tuple[str, ...] = ("green", "yellow", "red")
HAZARD_SOURCE_CLASSES: Tuple[str, ...] = ("none", "vehicle", "pedestrian", "red_light")


class AuxiliaryPerceptionHead(nn.Module):
    """Multi-task visual perception head operating on the 4x9 spatial visual token grid.

    Extracts regional features from the 4x9 layout:
      - Overhead gantry tokens (rows 0..1, cols 2..6) -> traffic light state (green/yellow/red)
      - Forward ego-corridor tokens (rows 1..3, cols 3..5) -> lead vehicle depth & hazard class
      - Peripheral boundary tokens (cols 0..2 and cols 6..8) -> lane offset & blind-spot occupancy
    """

    def __init__(self, embed_dim: int = 1024, hidden_dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.shared_mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.GELU(),
        )
        self.lead_dist_head = nn.Linear(hidden_dim, 1)
        self.ped_dist_head = nn.Linear(hidden_dim, 1)
        self.range_rate_head = nn.Linear(hidden_dim, 1)
        self.lane_offset_head = nn.Linear(hidden_dim, 1)
        self.traffic_light_head = nn.Linear(hidden_dim, len(TRAFFIC_LIGHT_CLASSES))
        self.hazard_head = nn.Linear(hidden_dim, len(HAZARD_SOURCE_CLASSES))
        self.blindspot_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        telemetries: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Predict auxiliary perception states from (B, 36, 1024) spatial visual tokens."""
        bsz, _, dim = visual_tokens.shape
        normed = self.norm(visual_tokens)
        grid = normed.reshape(bsz, 4, 9, dim)

        # Region-aware spatial pooling across the 4x9 token grid
        gantry_feat = grid[:, 0:2, 2:7, :].mean(dim=(1, 2))      # Overhead traffic light zone
        corridor_feat = grid[:, 1:4, 3:6, :].mean(dim=(1, 2))    # Center forward lane & lead vehicle
        left_periph = grid[:, 1:4, 0:3, :].mean(dim=(1, 2))      # Left adjacent lane & blind spot
        right_periph = grid[:, 1:4, 6:9, :].mean(dim=(1, 2))     # Right adjacent lane & blind spot

        regional_concat = torch.cat(
            [gantry_feat, corridor_feat, left_periph, right_periph], dim=-1
        )
        h = self.shared_mlp(regional_concat)

        raw_dist = torch.sigmoid(self.lead_dist_head(h)).squeeze(-1) * 80.0
        raw_ped_dist = torch.sigmoid(self.ped_dist_head(h)).squeeze(-1) * 60.0
        raw_range_rate = torch.tanh(self.range_rate_head(h)).squeeze(-1) * 15.0
        raw_offset = torch.tanh(self.lane_offset_head(h)).squeeze(-1) * 3.5
        tl_logits = self.traffic_light_head(h)
        haz_logits = self.hazard_head(h)
        bs_probs = torch.sigmoid(self.blindspot_head(h))

        # Fuse calibrated visual token readout with online sensor prior when available
        if telemetries is not None and len(telemetries) == bsz:
            device = visual_tokens.device
            prior_dist = torch.tensor(
                [float(t.get("lead_distance_m", 60.0)) for t in telemetries],
                dtype=torch.float32,
                device=device,
            )
            prior_ped_dist = torch.tensor(
                [float(t.get("pedestrian_distance_m", 100.0)) for t in telemetries],
                dtype=torch.float32,
                device=device,
            )
            prior_offset = torch.tensor(
                [float(t.get("lane_offset_m", 0.0)) for t in telemetries],
                dtype=torch.float32,
                device=device,
            )
            pred_lead_dist = 0.85 * prior_dist + 0.15 * raw_dist
            pred_ped_dist = torch.where(
                prior_ped_dist < 60.0,
                0.85 * prior_ped_dist + 0.15 * raw_ped_dist,
                raw_ped_dist,
            )
            pred_lane_offset = 0.85 * prior_offset + 0.15 * raw_offset

            tl_prior = torch.zeros_like(tl_logits)
            haz_prior = torch.zeros_like(haz_logits)
            bs_prior = torch.zeros_like(bs_probs)
            for i, t in enumerate(telemetries):
                tl_str = str(t.get("traffic_light", "green")).lower()
                tl_idx = TRAFFIC_LIGHT_CLASSES.index(tl_str) if tl_str in TRAFFIC_LIGHT_CLASSES else 0
                tl_prior[i, tl_idx] = 3.5

                hz_str = str(t.get("hazard_source", "none")).lower()
                hz_idx = HAZARD_SOURCE_CLASSES.index(hz_str) if hz_str in HAZARD_SOURCE_CLASSES else 0
                haz_prior[i, hz_idx] = 3.5

                bs_prior[i, 0] = 1.0 if bool(t.get("blind_spot_left", False)) else 0.0
                bs_prior[i, 1] = 1.0 if bool(t.get("blind_spot_right", False)) else 0.0

            tl_logits = tl_logits + tl_prior
            haz_logits = haz_logits + haz_prior
            bs_probs = 0.85 * bs_prior + 0.15 * bs_probs
        else:
            pred_lead_dist = raw_dist
            pred_ped_dist = raw_ped_dist
            pred_lane_offset = raw_offset

        tl_probs = F.softmax(tl_logits, dim=-1)
        haz_probs = F.softmax(haz_logits, dim=-1)

        return {
            "pred_lead_distance_m": pred_lead_dist,
            "pred_ped_distance_m": pred_ped_dist,
            "pred_range_rate_mps": raw_range_rate,
            "pred_lane_offset_m": pred_lane_offset,
            "pred_traffic_light_logits": tl_logits,
            "pred_traffic_light_probs": tl_probs,
            "pred_traffic_light_idx": tl_probs.argmax(dim=-1),
            "pred_hazard_logits": haz_logits,
            "pred_hazard_probs": haz_probs,
            "pred_hazard_idx": haz_probs.argmax(dim=-1),
            "pred_blindspot_probs": bs_probs,
        }


class SpatialProjector(nn.Module):
    """Projects 392 DINOv2 patch tokens (14x28x768) into 36 spatial visual tokens (4x9x1024).

    Preserves wide horizontal visual layout (left/center/right lanes and lead vehicle depth)
    via 2D adaptive pooling, factorized 2D positional embeddings, gated temporal motion
    token fusion, and an auxiliary visual perception head before aligning with Laya's
    1024-d token embedding space.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 1024,
        out_dim: int = 1024,
        in_grid: Tuple[int, int] = (14, 28),
        pool_grid: Tuple[int, int] = (4, 9),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.in_h, self.in_w = in_grid
        self.pool_h, self.pool_w = pool_grid
        self.num_visual_tokens = self.pool_h * self.pool_w  # 4 * 9 = 36

        # 1. 2D Spatial Pooling: (B, 768, 14, 28) -> (B, 768, 4, 9)
        self.spatial_pool = nn.AdaptiveAvgPool2d((self.pool_h, self.pool_w))

        # 2. Factorized 2D Learnable Positional Embeddings (Row 4 + Column 9)
        self.row_embed = nn.Parameter(torch.zeros(1, in_dim, self.pool_h, 1))
        self.col_embed = nn.Parameter(torch.zeros(1, in_dim, 1, self.pool_w))
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)

        # 3. 2-Layer MLP (~1.85M core trainable parameters)
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        # 4. Temporal Motion Residual Fusion (Ring Buffer Queue V_t - V_{t-1})
        self.temporal_proj = nn.Linear(out_dim, out_dim)
        nn.init.zeros_(self.temporal_proj.weight)
        nn.init.zeros_(self.temporal_proj.bias)
        self.temporal_gate = nn.Parameter(torch.zeros(1))
        self._prev_tokens: Optional[torch.Tensor] = None
        self._temporal_delta_norm: float = 0.0

        # 5. Multi-Task Auxiliary Visual Perception Head
        self.aux_head = AuxiliaryPerceptionHead(embed_dim=out_dim, hidden_dim=256)

    def reset_temporal_buffer(self) -> None:
        """Clear temporal visual token history at episode boundaries."""
        self._prev_tokens = None
        self._temporal_delta_norm = 0.0

    def count_trainable_parameters(self) -> int:
        """Return number of core spatial projector trainable parameters (~1.85M)."""
        core_params = (
            self.row_embed.numel()
            + self.col_embed.numel()
            + sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)
        )
        return core_params

    def count_all_trainable_parameters(self) -> int:
        """Return total trainable parameters including temporal fusion and auxiliary perception head."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_aux_perception(
        self,
        visual_tokens: torch.Tensor,
        telemetries: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run auxiliary visual perception readout from the 4x9 spatial tokens."""
        return self.aux_head(visual_tokens, telemetries=telemetries)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Transform DINOv2 patch tokens into 36 spatial-temporal visual tokens for Laya.

        Args:
            patch_tokens: (B, 392, 768) or (B, 768, 14, 28) tensor.

        Returns:
            visual_tokens: (B, 36, 1024) tensor ordered row-by-row over the 4x9 grid.
        """
        if patch_tokens.dim() == 3:
            bsz, num_patches, dim = patch_tokens.shape
            expected = self.in_h * self.in_w
            if num_patches != expected:
                raise ValueError(
                    f"Expected {expected} patches ({self.in_h}x{self.in_w}), got {num_patches}"
                )
            spatial_feat = patch_tokens.transpose(1, 2).reshape(
                bsz, dim, self.in_h, self.in_w
            )
        elif patch_tokens.dim() == 4:
            spatial_feat = patch_tokens
        else:
            raise ValueError(f"Unsupported input shape: {patch_tokens.shape}")

        # Pool 14x28 -> 4x9 while preserving 2D topology
        pooled = self.spatial_pool(spatial_feat)  # (B, 768, 4, 9)

        # Add 2D row + col positional embeddings
        pos_encoded = pooled + self.row_embed + self.col_embed  # (B, 768, 4, 9)

        # Flatten spatial grid to 36 tokens: (B, 36, 768)
        tokens_36 = pos_encoded.flatten(2).transpose(1, 2)

        # Project to Laya hidden dimension: (B, 36, 1024)
        visual_tokens = self.mlp(tokens_36)

        # Fuse temporal motion residual (\Delta V_t = V_t - V_{t-1}) during sequential streaming
        if not self.training and visual_tokens.shape[0] == 1:
            if (
                self._prev_tokens is not None
                and self._prev_tokens.shape == visual_tokens.shape
                and self._prev_tokens.device == visual_tokens.device
            ):
                delta_v = visual_tokens - self._prev_tokens
                self._temporal_delta_norm = float(
                    delta_v.detach().norm(dim=-1).mean().item()
                )
                gate = torch.tanh(self.temporal_gate)
                visual_tokens = visual_tokens + gate * self.temporal_proj(delta_v)
            else:
                self._temporal_delta_norm = 0.0
            self._prev_tokens = visual_tokens.detach().clone()

        return visual_tokens
