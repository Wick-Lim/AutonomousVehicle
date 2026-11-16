"""Smoke tests for unified production CLI (`python -m src.cli`)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestCLISmoke(unittest.TestCase):
    """Verify doctor, benchmark-rust, and export CLI subcommands succeed cleanly."""

    def test_cli_doctor_json(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "src.cli", "doctor", "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["config_valid"])
        self.assertEqual(payload["trainable_parameters"], 4_232_991)
        self.assertTrue(payload["parameter_invariant_ok"])

    def test_cli_benchmark_rust(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "src.cli", "benchmark-rust", "--iterations", "500"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertGreater(payload["python_fallback_us_per_step"], 0.0)
        if payload.get("rust_core_available"):
            self.assertLess(payload["max_control_parity_diff"], 1e-9)

    def test_cli_export_mock_torchscript(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "exported_projector.pt"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.cli",
                    "export",
                    "--mock",
                    "--format",
                    "torchscript",
                    "--output",
                    str(out_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["format"], "torchscript")
            self.assertTrue(out_path.is_file())
            self.assertTrue(payload["verified"])
            self.assertLess(payload["max_abs_diff"], 1e-4)


if __name__ == "__main__":
    unittest.main()
