"""Deterministic seeding and runtime hardware introspection utilities."""

from __future__ import annotations

import os
import platform
import random
from typing import Dict

import numpy as np
import torch

from src.control.rust_backend import BACKEND_NAME, RUST_CORE_AVAILABLE


def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """Seed Python `random`, NumPy, and PyTorch CPU/CUDA RNGs."""
    seed_val = int(seed)
    random.seed(seed_val)
    os.environ["PYTHONHASHSEED"] = str(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed_val


def detect_hardware_env() -> Dict[str, str]:
    """Dynamically inspect runtime CPU, GPU, OS, PyTorch, and Rust backend state."""
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA GPU"
    else:
        gpu_name = "CPU-only"

    return {
        "cpu": platform.processor() or platform.machine() or "unknown",
        "gpu": gpu_name,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "control_backend": BACKEND_NAME,
        "rust_core_available": str(RUST_CORE_AVAILABLE),
        "precision": "FP32 (batch_size=1, warmup_passes=3)",
    }


__all__ = ["detect_hardware_env", "seed_everything"]
