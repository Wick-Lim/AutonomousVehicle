"""Model artifact export utilities (TorchScript and optional ONNX) for SpatialProjector."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from src.models.laya_pilot import AutonomousDrivingSystem


class _ExportableSpatialProjector(nn.Module):
    """Stateless wrapper tracing SpatialProjector patch-to-token projection."""

    def __init__(self, projector: nn.Module) -> None:
        super().__init__()
        self.projector = projector

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return self.projector(patch_tokens, elapsed_steps=1, prev_visual_tokens=None)


def export_spatial_projector(
    system: AutonomousDrivingSystem,
    output_path: str | Path,
    export_format: str = "torchscript",
    verify: bool = True,
) -> Dict[str, Any]:
    """Export `SpatialProjector` `(B, 392, 768) -> (B, 36, 1024)` to TorchScript or ONNX and verify parity."""
    fmt = export_format.strip().lower()
    if fmt not in ("torchscript", "onnx"):
        raise ValueError(f"Unsupported export format '{export_format}'; expected 'torchscript' or 'onnx'")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    system.eval()
    system.projector.eval()
    system.projector.reset_temporal_buffer()

    first_param = next(system.projector.parameters())
    device = first_param.device
    sample_patches = torch.randn(
        1,
        int(system.dino.num_patches),
        int(system.dino.embed_dim),
        dtype=torch.float32,
        device=device,
    )

    wrapper = _ExportableSpatialProjector(system.projector).eval()
    with torch.no_grad():
        eager_out = wrapper(sample_patches)
        system.projector.reset_temporal_buffer()

        max_abs_diff = 0.0
        if fmt == "torchscript":
            traced = torch.jit.trace(wrapper, (sample_patches,))
            torch.jit.save(traced, str(out_file))
            if verify:
                reloaded = torch.jit.load(str(out_file), map_location=device)
                reloaded_out = reloaded(sample_patches)
                max_abs_diff = float((eager_out - reloaded_out).abs().max().item())
                if max_abs_diff >= 1e-4:
                    raise RuntimeError(
                        f"TorchScript verification failed: max_abs_diff={max_abs_diff:.6f}"
                    )
        else:
            if importlib.util.find_spec("onnx") is None:
                raise RuntimeError(
                    "ONNX export requires the `onnx` package; install `onnx` or use `torchscript`."
                )
            torch.onnx.export(
                wrapper,
                sample_patches,
                str(out_file),
                input_names=["patch_tokens"],
                output_names=["visual_tokens"],
                dynamic_axes={"patch_tokens": {0: "batch"}, "visual_tokens": {0: "batch"}},
                opset_version=17,
            )
            if verify and importlib.util.find_spec("onnxruntime") is not None:
                import onnxruntime as ort  # type: ignore[import-not-found]

                sess = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
                ort_out = sess.run(["visual_tokens"], {"patch_tokens": sample_patches.cpu().numpy()})[0]
                max_abs_diff = float(
                    torch.from_numpy(ort_out).sub(eager_out.cpu()).abs().max().item()
                )
                if max_abs_diff >= 1e-4:
                    raise RuntimeError(
                        f"ONNX verification failed: max_abs_diff={max_abs_diff:.6f}"
                    )

    return {
        "format": fmt,
        "path": str(out_file),
        "size_bytes": int(out_file.stat().st_size),
        "max_abs_diff": max_abs_diff,
        "verified": bool(verify),
    }


__all__ = ["export_spatial_projector"]
