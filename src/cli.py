"""Unified production CLI for AutonomousVehicle (`python -m src.cli` / `av-drive`)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.config import DEFAULT_CONFIG_PATH, load_validated_config
from src.control.rust_backend import (
    BACKEND_NAME,
    RUST_CORE_AVAILABLE,
    python_fallback_module,
    rust_core_module,
)
from src.models.laya_pilot import AutonomousDrivingSystem
from src.runtime.export import export_spatial_projector
from src.sim.carla_launcher import get_carla_server_manager
from src.utils.seed import detect_hardware_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_PIPELINE_SCRIPTS = {
    "collect": PROJECT_ROOT / "scripts" / "01_collect_data.py",
    "train": PROJECT_ROOT / "scripts" / "02_train_projector.py",
    "calibrate": PROJECT_ROOT / "scripts" / "03_calibrate.py",
    "eval": PROJECT_ROOT / "scripts" / "04_evaluate_sim.py",
    "stress-bench": PROJECT_ROOT / "scripts" / "05_stress_benchmark.py",
    "multiroute-bench": PROJECT_ROOT / "scripts" / "06_multiroute_benchmark.py",
    "train-e2e": PROJECT_ROOT / "scripts" / "07_train_e2e_trajectory.py",
    "finetune-rl": PROJECT_ROOT / "scripts" / "08_finetune_rl.py",
    "e2e-bench": PROJECT_ROOT / "scripts" / "09_e2e_benchmark.py",
}


def _cmd_doctor(config_path: Optional[str], output_json: bool) -> int:
    app_cfg = load_validated_config(config_path)
    cfg_dict = app_cfg.to_dict()
    hw = detect_hardware_env()

    system = AutonomousDrivingSystem(config=cfg_dict, use_mock=True)
    trainable_params = sum(p.numel() for p in system.parameters() if p.requires_grad)

    if RUST_CORE_AVAILABLE and rust_core_module is not None:
        native_us = float(rust_core_module.benchmark_rust_control_loop_us(1000))
    else:
        native_us = float(python_fallback_module.benchmark_rust_control_loop_us(100))

    ckpt_dir = PROJECT_ROOT / app_cfg.train.checkpoint_dir
    proj_ckpt = PROJECT_ROOT / app_cfg.train.projector_ckpt
    temp_ckpt = PROJECT_ROOT / app_cfg.train.temperature_ckpt
    md_installed = importlib.util.find_spec("metadrive") is not None
    carla_stat = get_carla_server_manager().status(
        host=str(app_cfg.env.carla.host),
        port=int(app_cfg.env.carla.port),
    )

    report: Dict[str, Any] = {
        "status": "ok",
        "config_path": str(config_path or DEFAULT_CONFIG_PATH),
        "config_valid": True,
        "hardware": hw,
        "rust_core_available": RUST_CORE_AVAILABLE,
        "control_backend": BACKEND_NAME,
        "control_loop_us_per_step": round(native_us, 4),
        "trainable_parameters": trainable_params,
        "parameter_invariant_ok": trainable_params == 4_232_991,
        "simulators": {
            "metadrive_3d_installed": md_installed,
            "carla": carla_stat,
        },
        "checkpoints": {
            "projector_best": proj_ckpt.is_file(),
            "temperature": temp_ckpt.is_file(),
            "eval_metrics": (ckpt_dir / "eval_metrics.json").is_file(),
            "eval_metrics_carla": (ckpt_dir / "eval_metrics_carla.json").is_file(),
        },
    }

    if output_json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 68)
        print("AutonomousVehicle Production Doctor Report")
        print("-" * 68)
        print(f"  Config Valid          : {report['config_valid']}")
        print(f"  Python / PyTorch      : {hw['python']} / {hw['pytorch']}")
        print(f"  Compute Device        : {hw['gpu']}")
        print(
            f"  Control Backend       : {BACKEND_NAME} "
            f"(RUST_CORE_AVAILABLE={RUST_CORE_AVAILABLE}, {native_us:.3f} us/step)"
        )
        print(
            f"  Trainable Parameters  : {trainable_params:,} "
            f"(invariant={'PASS' if report['parameter_invariant_ok'] else 'FAIL'})"
        )
        print(f"  Checkpoints           : {report['checkpoints']}")
        print("=" * 68)

    return 0 if report["parameter_invariant_ok"] else 1


def _cmd_benchmark_rust(iterations: int) -> int:
    n = max(10, int(iterations))
    py_n = min(n, 2000)

    py_us = float(python_fallback_module.benchmark_rust_control_loop_us(py_n))
    result: Dict[str, Any] = {
        "rust_core_available": RUST_CORE_AVAILABLE,
        "iterations_python": py_n,
        "python_fallback_us_per_step": round(py_us, 3),
    }

    if RUST_CORE_AVAILABLE and rust_core_module is not None:
        rust_us = float(rust_core_module.benchmark_rust_control_loop_us(n))
        # Verify single-step control parity
        dec = {
            "maneuver": "follow_lead",
            "maneuver_confidence": 0.92,
            "speed_level": 2,
            "emergency_brake": 0.04,
            "predicted_trajectory": [[3.0 * k, 0.1 * k, 25.0] for k in range(1, 7)],
        }
        tel = {
            "speed_kmh": 24.0,
            "lead_distance_m": 18.5,
            "lead_speed_kmh": 22.0,
            "pedestrian_distance_m": 100.0,
            "lane_offset_m": 0.08,
            "heading_error": -0.02,
            "road_curvature": 0.18,
            "traffic_light": "green",
            "hazard_source": "vehicle",
            "left_lane_available": True,
            "right_lane_available": True,
        }
        r_ctrl = rust_core_module.RustSafetyShieldPID()
        p_ctrl = python_fallback_module.RustSafetyShieldPID()
        (rs, rt), _ = r_ctrl.compute_control(dec, tel, 0.05, True)
        (ps, pt), _ = p_ctrl.compute_control(dec, tel, 0.05, True)
        parity_diff = max(abs(rs - ps), abs(rt - pt))

        result.update(
            {
                "iterations_rust": n,
                "native_rust_us_per_step": round(rust_us, 4),
                "speedup_vs_python": round(py_us / max(1e-6, rust_us), 1),
                "max_control_parity_diff": round(parity_diff, 12),
            }
        )

    print(json.dumps(result, indent=2))
    return 0


def _cmd_export(
    config_path: Optional[str],
    use_mock: bool,
    projector_ckpt: Optional[str],
    output_path: str,
    export_format: str,
) -> int:
    app_cfg = load_validated_config(config_path)
    cfg = app_cfg.to_dict()
    eff_mock = bool(use_mock or app_cfg.project.use_mock)
    system = AutonomousDrivingSystem(config=cfg, use_mock=eff_mock)
    ckpt_file = (
        Path(projector_ckpt)
        if projector_ckpt is not None
        else PROJECT_ROOT / app_cfg.train.projector_ckpt
    )
    if ckpt_file.is_file():
        system.load_adapter_checkpoint(ckpt_file, map_location="cpu", strict=False)

    meta = export_spatial_projector(
        system=system,
        output_path=output_path,
        export_format=export_format,
        verify=True,
    )
    print(json.dumps(meta, indent=2))
    return 0


def _cmd_serve(
    host: str,
    port: int,
    config_path: Optional[str],
    use_mock: Optional[bool],
    projector_ckpt: Optional[str],
    temperature_ckpt: Optional[str],
) -> int:
    import uvicorn

    from src.api.app import create_app

    app = create_app(
        config_path=config_path,
        use_mock=use_mock,
        projector_ckpt=projector_ckpt,
        temperature_ckpt=temperature_ckpt,
    )
    uvicorn.run(app, host=host, port=int(port), log_level="info")
    return 0


def _cmd_cockpit(
    host: str,
    port: int,
    simulator: str,
    open_browser: bool,
    config_path: Optional[str],
    use_mock: Optional[bool],
    projector_ckpt: Optional[str],
    temperature_ckpt: Optional[str],
) -> int:
    import threading
    import uvicorn

    from src.api.app import create_app

    os.environ["AV_DEFAULT_SIMULATOR"] = str(simulator)
    url = f"http://{host}:{int(port)}/cockpit"
    print("=" * 68)
    print(f"Interactive 3D Simulation Cockpit URL: {url}")
    print(f"Default Simulator Engine            : {simulator.upper()}")
    print("=" * 68, flush=True)
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app = create_app(
        config_path=config_path,
        use_mock=use_mock,
        projector_ckpt=projector_ckpt,
        temperature_ckpt=temperature_ckpt,
    )
    uvicorn.run(app, host=host, port=int(port), log_level="info")
    return 0


def _cmd_pipeline(name: str, passthrough_args: List[str]) -> int:
    script = _PIPELINE_SCRIPTS[name]
    cmd = [sys.executable, str(script), *passthrough_args]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    _ = time.perf_counter() - t0
    return proc.returncode


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="av-drive",
        description="Unified AutonomousVehicle CLI for diagnostics, export, serving, and training/evaluation pipelines.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Inspect config, Rust core, GPU/CPU, parameters, and checkpoints.")
    p_doc.add_argument("--config", type=str, default=None)
    p_doc.add_argument("--json", action="store_true", help="Print JSON output.")

    p_bench = sub.add_parser("benchmark-rust", help="Benchmark native Rust core vs pure-Python fallback.")
    p_bench.add_argument("--iterations", type=int, default=10000)

    p_exp = sub.add_parser("export", help="Export SpatialProjector to TorchScript or ONNX.")
    p_exp.add_argument("--config", type=str, default=None)
    p_exp.add_argument("--mock", action="store_true")
    p_exp.add_argument("--projector-ckpt", type=str, default=None)
    p_exp.add_argument("--output", type=str, default="checkpoints/spatial_projector_ts.pt")
    p_exp.add_argument("--format", type=str, default="torchscript", choices=["torchscript", "onnx"])

    p_srv = sub.add_parser("serve", help="Launch FastAPI closed-loop inference server.")
    p_srv.add_argument("--host", type=str, default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--config", type=str, default=None)
    p_srv.add_argument("--mock", action="store_true")
    p_srv.add_argument("--projector-ckpt", type=str, default=None)
    p_srv.add_argument("--temperature-ckpt", type=str, default=None)

    p_cockpit = sub.add_parser("cockpit", help="Launch Interactive 3D Web Simulation Cockpit (/cockpit).")
    p_cockpit.add_argument("--host", type=str, default="127.0.0.1")
    p_cockpit.add_argument("--port", type=int, default=8000)
    p_cockpit.add_argument("--simulator", type=str, default="metadrive", choices=["metadrive", "carla"])
    p_cockpit.add_argument("--open-browser", action="store_true")
    p_cockpit.add_argument("--config", type=str, default=None)
    p_cockpit.add_argument("--mock", action="store_true")
    p_cockpit.add_argument("--projector-ckpt", type=str, default=None)
    p_cockpit.add_argument("--temperature-ckpt", type=str, default=None)

    for pipe_name in (
        "collect",
        "train",
        "calibrate",
        "eval",
        "stress-bench",
        "multiroute-bench",
        "train-e2e",
        "finetune-rl",
        "e2e-bench",
    ):
        sub.add_parser(pipe_name, help=f"Run pipeline stage `{pipe_name}` (forwards remaining CLI flags).")

    args, extras = parser.parse_known_args(argv)
    if args.command == "doctor":
        return _cmd_doctor(config_path=args.config, output_json=args.json)
    if args.command == "benchmark-rust":
        return _cmd_benchmark_rust(iterations=args.iterations)
    if args.command == "export":
        return _cmd_export(
            config_path=args.config,
            use_mock=bool(args.mock),
            projector_ckpt=args.projector_ckpt,
            output_path=args.output,
            export_format=args.format,
        )
    if args.command == "serve":
        return _cmd_serve(
            host=args.host,
            port=args.port,
            config_path=args.config,
            use_mock=True if args.mock else None,
            projector_ckpt=args.projector_ckpt,
            temperature_ckpt=args.temperature_ckpt,
        )
    if args.command == "cockpit":
        return _cmd_cockpit(
            host=args.host,
            port=args.port,
            simulator=args.simulator,
            open_browser=bool(args.open_browser),
            config_path=args.config,
            use_mock=True if args.mock else None,
            projector_ckpt=args.projector_ckpt,
            temperature_ckpt=args.temperature_ckpt,
        )
    if args.command in _PIPELINE_SCRIPTS:
        return _cmd_pipeline(args.command, extras)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
