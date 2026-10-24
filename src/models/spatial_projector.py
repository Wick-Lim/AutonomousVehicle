"""2D Spatial Projector: AdaptiveAvgPool2d(4, 9) + 2D Positional Embedding + 2-Layer MLP."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class SpatialProjector(nn.Module):
    """Projects 392 DINOv2 patch tokens (14x28x768) into 36 spatial visual tokens (4x9x1024).

    Preserves wide horizontal visual layout (left/center/right lanes and lead vehicle depth)
    via 2D adaptive pooling and factorized 2D positional embeddings before aligning with Laya's
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

        # 3. 2-Layer MLP (~1.84M trainable parameters)
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def count_trainable_parameters(self) -> int:
        """Return total number of trainable parameters (~1.85M)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Transform DINOv2 patch tokens into 36 spatial visual tokens for Laya.

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
        return visual_tokens
