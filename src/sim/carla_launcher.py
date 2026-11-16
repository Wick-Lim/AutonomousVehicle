"""CARLA 0.9.16 server discovery, TCP readiness probe, and headless subprocess manager."""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_utils import get_logger

_LOG = get_logger("autonomous_vehicle.carla_launcher")


def is_tcp_port_open(host: str = "127.0.0.1", port: int = 2000, timeout: float = 0.5) -> bool:
    """Return True if a TCP server is accepting connections at `host:port`."""
    try:
        with socket.create_connection((str(host), int(port)), timeout=float(timeout)):
            return True
    except OSError:
        return False


def find_carla_executable(explicit_path: Optional[str | Path] = None) -> Optional[Path]:
    """Locate local CARLA 0.9.16 server executable (`CarlaUE4.exe` on Windows or `CarlaUE4.sh` on Linux)."""
    candidates: List[Path] = []
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.is_dir():
            candidates.extend([p / "CarlaUE4.exe", p / "CarlaUE4.sh"])
        else:
            candidates.append(p)

    env_exe = os.environ.get("AV_CARLA_EXE")
    if env_exe:
        candidates.append(Path(env_exe))

    env_root = os.environ.get("AV_CARLA_ROOT")
    if env_root:
        root_p = Path(env_root)
        candidates.extend([root_p / "CarlaUE4.exe", root_p / "CarlaUE4.sh"])

    # Standard local install locations
    candidates.extend(
        [
            Path(r"C:\Users\ijh69\Downloads\CARLA_0.9.16\CarlaUE4.exe"),
            Path.home() / "Downloads" / "CARLA_0.9.16" / "CarlaUE4.exe",
            Path(r"C:\CARLA_0.9.16\CarlaUE4.exe"),
            Path("/opt/carla-simulator/CarlaUE4.sh"),
            Path.home() / "CARLA_0.9.16" / "CarlaUE4.sh",
        ]
    )

    for cand in candidates:
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


class CarlaServerManager:
    """Lifecycle manager for spawning/stopping a local CARLA 0.9.16 server when needed."""

    def __init__(self, explicit_exe: Optional[str | Path] = None) -> None:
        self.explicit_exe = explicit_exe
        self._proc: Optional[subprocess.Popen[Any]] = None
        self._atexit_registered: bool = False

    def status(self, host: str = "127.0.0.1", port: int = 2000) -> Dict[str, Any]:
        """Inspect CARLA binary availability and RPC port readiness."""
        exe = find_carla_executable(self.explicit_exe)
        managed_running = self._proc is not None and self._proc.poll() is None
        return {
            "carla_exe_found": exe is not None,
            "carla_exe_path": str(exe) if exe is not None else None,
            "host": host,
            "port": int(port),
            "port_open": is_tcp_port_open(host, port, timeout=0.3),
            "managed_pid": self._proc.pid if managed_running else None,
        }

    def ensure_running(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        offscreen: bool = True,
        quality_level: str = "Low",
        timeout_s: float = 60.0,
        explicit_exe: Optional[str | Path] = None,
    ) -> bool:
        """Ensure CARLA is reachable on `host:port`, launching local `CarlaUE4` if closed."""
        if is_tcp_port_open(host, port, timeout=0.4):
            return True

        exe = find_carla_executable(explicit_exe or self.explicit_exe)
        if exe is None:
            _LOG.warning("CARLA executable not found; cannot auto-launch CARLA server.")
            return False

        if self._proc is None or self._proc.poll() is not None:
            cmd = [
                str(exe),
                f"-carla-rpc-port={int(port)}",
                f"-quality-level={quality_level}",
                "-nosound",
            ]
            if offscreen:
                cmd.append("-RenderOffScreen")
            _LOG.info(f"Launching CARLA server: {' '.join(cmd)}")
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(exe.parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as exc:
                _LOG.error(f"Failed to launch CARLA server {exe}: {exc}")
                self._proc = None
                return False

            if not self._atexit_registered:
                atexit.register(self.stop)
                self._atexit_registered = True

        deadline = time.monotonic() + max(5.0, float(timeout_s))
        while time.monotonic() < deadline:
            if is_tcp_port_open(host, port, timeout=0.4):
                # Give Unreal Streaming / World tick 1.5s extra to finish bootstrapping RPC world
                time.sleep(1.5)
                return True
            if self._proc is not None and self._proc.poll() is not None:
                _LOG.error(f"CARLA server exited prematurely with code {self._proc.returncode}")
                self._proc = None
                return False
            time.sleep(1.0)

        _LOG.warning(f"Timed out waiting {timeout_s:.1f}s for CARLA RPC at {host}:{port}.")
        return False

    def stop(self) -> None:
        """Terminate only the CARLA server subprocess spawned by this manager."""
        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
            except Exception:
                pass
            finally:
                self._proc = None


_GLOBAL_CARLA_MANAGER = CarlaServerManager()


def get_carla_server_manager() -> CarlaServerManager:
    """Return the process-wide shared CARLA server manager."""
    return _GLOBAL_CARLA_MANAGER


__all__ = [
    "CarlaServerManager",
    "find_carla_executable",
    "get_carla_server_manager",
    "is_tcp_port_open",
]
