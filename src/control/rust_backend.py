"""Rust PyO3 acceleration backend (`autonomous_vehicle_rust`) with automatic Python fallback."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

_CONTROL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONTROL_DIR.parents[1]

rust_core_module: Optional[Any] = None
RUST_CORE_AVAILABLE: bool = False


def _load_rust_extension() -> Optional[Any]:
    """Attempt to import `autonomous_vehicle_rust` from standard path or local build directories."""
    try:
        import autonomous_vehicle_rust as _mod  # type: ignore[import-not-found]

        return _mod
    except ImportError:
        pass

    candidate_paths = [
        _CONTROL_DIR / "autonomous_vehicle_rust.so",
        _PROJECT_ROOT / "rust_core" / "target" / "release" / "libautonomous_vehicle_rust.dylib",
        _PROJECT_ROOT / "rust_core" / "target" / "release" / "libautonomous_vehicle_rust.so",
    ]
    for cand in candidate_paths:
        if cand.is_file():
            try:
                if str(_CONTROL_DIR) not in sys.path:
                    sys.path.insert(0, str(_CONTROL_DIR))
                spec = importlib.util.spec_from_file_location(
                    "autonomous_vehicle_rust", str(cand)
                )
                if spec is not None and spec.loader is not None:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules["autonomous_vehicle_rust"] = mod
                    spec.loader.exec_module(mod)
                    return mod
            except Exception:
                continue
    return None


rust_core_module = _load_rust_extension()
RUST_CORE_AVAILABLE = rust_core_module is not None

__all__ = [
    "RUST_CORE_AVAILABLE",
    "rust_core_module",
]
