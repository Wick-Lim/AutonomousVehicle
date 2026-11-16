"""Cross-platform build script for the `autonomous_vehicle_rust` PyO3 native extension."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest_path = PROJECT_ROOT / "rust_core" / "Cargo.toml"
    env = os.environ.copy()
    env.setdefault("PYO3_PYTHON", sys.executable)

    cmd = ["cargo", "build", "--release", "--manifest-path", str(manifest_path)]
    print(f"[build_rust] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if proc.returncode != 0:
        print(f"[build_rust] Cargo build failed with exit code {proc.returncode}", file=sys.stderr)
        return proc.returncode

    verify_code = (
        "import json; "
        "from src.control.rust_backend import RUST_CORE_AVAILABLE, BACKEND_NAME, rust_core_module; "
        "us = float(rust_core_module.benchmark_rust_control_loop_us(2000)) if RUST_CORE_AVAILABLE else -1.0; "
        "print(json.dumps({'rust_core_available': RUST_CORE_AVAILABLE, 'backend': BACKEND_NAME, 'native_us_per_step': round(us, 4)}))"
    )
    verify = subprocess.run(
        [sys.executable, "-c", verify_code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0:
        print(verify.stderr, file=sys.stderr)
        return verify.returncode

    print(f"[build_rust] Verification: {verify.stdout.strip()}")
    if '"rust_core_available": true' not in verify.stdout.lower():
        print("[build_rust] ERROR: Built artifact could not be loaded by rust_backend.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
