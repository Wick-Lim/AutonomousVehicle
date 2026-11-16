"""AutonomousVehicle: Sub-40ms Non-Autoregressive Autonomous Driving powered by DINOv2 & Laya."""

from src.config import AppConfig, load_validated_config
from src.control.rust_backend import BACKEND_NAME, RUST_CORE_AVAILABLE, get_backend
from src.control.shield_pid import SafetyShieldPID
from src.models.laya_pilot import AutonomousDrivingSystem

__version__ = "1.0.0"

__all__ = [
    "AppConfig",
    "AutonomousDrivingSystem",
    "BACKEND_NAME",
    "RUST_CORE_AVAILABLE",
    "SafetyShieldPID",
    "__version__",
    "get_backend",
    "load_validated_config",
]
