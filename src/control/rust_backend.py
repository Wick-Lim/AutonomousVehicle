"""Cross-platform PyO3 Rust acceleration backend (`autonomous_vehicle_rust`) with Python fallback."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

from src.control import python_fallback as python_fallback_module

_LOG = logging.getLogger(__name__)
_CONTROL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONTROL_DIR.parents[1]

rust_core_module: Optional[Any] = None
RUST_CORE_AVAILABLE: bool = False
BACKEND_NAME: str = "python_fallback"
_FALLBACK_WARNED: bool = False


def _candidate_extension_paths() -> List[Path]:
    cands: List[Path] = []
    # 1. Pre-staged extensions colocated inside src/control/
    for pattern in ("autonomous_vehicle_rust*.pyd", "autonomous_vehicle_rust*.so"):
        cands.extend(sorted(_CONTROL_DIR.glob(pattern)))
    cands.append(_CONTROL_DIR / "autonomous_vehicle_rust.dll")

    # 2. Local Cargo release & debug outputs across Windows / macOS / Linux
    for profile in ("release", "debug"):
        target_dir = _PROJECT_ROOT / "rust_core" / "target" / profile
        cands.extend(
            [
                target_dir / "autonomous_vehicle_rust.pyd",
                target_dir / "autonomous_vehicle_rust.dll",
                target_dir / "libautonomous_vehicle_rust.dylib",
                target_dir / "libautonomous_vehicle_rust.so",
                target_dir / "autonomous_vehicle_rust.so",
            ]
        )
    return cands


def _load_rust_extension() -> Optional[Any]:
    """Attempt to import `autonomous_vehicle_rust` from site-packages or local Cargo build outputs."""
    try:
        import autonomous_vehicle_rust as _mod  # type: ignore[import-not-found]

        return _mod
    except ImportError:
        pass

    for cand in _candidate_extension_paths():
        if not cand.is_file():
            continue
        try:
            if str(_CONTROL_DIR) not in sys.path:
                sys.path.insert(0, str(_CONTROL_DIR))
            # Explicit ExtensionFileLoader is required on Windows Python 3.12+ for `.dll` artifacts.
            loader = importlib.machinery.ExtensionFileLoader(
                "autonomous_vehicle_rust", str(cand)
            )
            spec = importlib.util.spec_from_file_location(
                "autonomous_vehicle_rust", str(cand), loader=loader
            )
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["autonomous_vehicle_rust"] = mod
                spec.loader.exec_module(mod)
                return mod
        except Exception:
            sys.modules.pop("autonomous_vehicle_rust", None)
            continue
    return None


def get_backend(require_rust: bool = False) -> Any:
    """Return native Rust module if loaded, else pure-Python fallback (unless `require_rust=True`)."""
    global _FALLBACK_WARNED
    strict = bool(require_rust or os.environ.get("AV_REQUIRE_RUST", "0") == "1")
    if rust_core_module is not None:
        return rust_core_module
    if strict:
        raise RuntimeError(
            "Rust extension `autonomous_vehicle_rust` is required. Build via `python scripts/build_rust.py`."
        )
    if not _FALLBACK_WARNED:
        _LOG.warning(
            "Native Rust extension `autonomous_vehicle_rust` not built; falling back to pure-Python backend."
        )
        _FALLBACK_WARNED = True
    return python_fallback_module


rust_core_module = _load_rust_extension()
RUST_CORE_AVAILABLE = rust_core_module is not None
BACKEND_NAME = "rust" if RUST_CORE_AVAILABLE else "python_fallback"

__all__ = [
    "BACKEND_NAME",
    "RUST_CORE_AVAILABLE",
    "get_backend",
    "python_fallback_module",
    "rust_core_module",
]
