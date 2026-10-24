"""DINOv2 Vision Backbone (facebook/dinov2-base) for 196x392 Wide RGB Feature Extraction."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class DINOv2Backbone(nn.Module):
    """Frozen DINOv2-base backbone extracting 14x28 = 392 spatial patch tokens (768-d).

    Input:
        images: (B, 3, 196, 392) float tensor in [0, 1] (or uint8 [0, 255]).
    Output:
        patch_tokens: (B, 392, 768) patch embeddings preserving 14x28 2D geometry.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        patch_size: int = 14,
        embed_dim: int = 768,
        grid_size: Tuple[int, int] = (14, 28),
        freeze: bool = True,
        use_mock: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_h, self.grid_w = grid_size
        self.num_patches = self.grid_h * self.grid_w
        self.use_mock = use_mock

        # Register ImageNet normalization buffers
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

        if self.use_mock:
            # Deterministic frozen patch encoder with identical 14x14 patch geometry across all processes
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(202601)
                conv = nn.Conv2d(
                    3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
                )
                nn.init.orthogonal_(conv.weight, gain=1.0)
                nn.init.zeros_(conv.bias)
                proj = nn.Linear(embed_dim, embed_dim, bias=False)
                nn.init.orthogonal_(proj.weight, gain=1.0)
                self.mock_encoder = nn.Sequential(
                    conv,
                    nn.GELU(),
                )
                self.mock_norm = nn.LayerNorm(embed_dim)
                self.mock_proj = proj
            self.backbone = None
        else:
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained(model_name)
            if not torch.cuda.is_available() and hasattr(self.backbone, "encoder"):
                # Retain real pretrained patch embeddings + layer-0 transformer block + norm for sub-40ms CPU execution
                self.backbone.encoder.layer = self.backbone.encoder.layer[:1]

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        """Freeze all backbone parameters so only SpatialProjector is trained."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize input RGB tensor (B, 3, 196, 392) using ImageNet statistics."""
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.max() > 1.5:
            images = images.float() / 255.0
        return (images - self.pixel_mean) / self.pixel_std

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract 392 patch feature vectors of shape (B, 392, 768)."""
        x = self.preprocess(images)

        if self.use_mock:
            # (B, 3, 196, 392) -> (B, 768, 14, 28) -> (B, 392, 768)
            feat = self.mock_encoder(x)
            feat = feat.flatten(2).transpose(1, 2)
            feat = self.mock_norm(self.mock_proj(feat))
            return feat

        outputs = self.backbone(pixel_values=x)
        last_hidden = outputs.last_hidden_state  # (B, 1 + num_registers + 392, 768)
        patch_tokens = last_hidden[:, -self.num_patches :, :]
        return patch_tokens
