"""Expert Labeler: Rust-accelerated (`PyO3`) translator from expert physical action + traffic state to Laya decision primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from src.control.rust_backend import rust_core_module

MANEUVER_OPTIONS: Tuple[str, ...] = (
    "keep_lane",
    "change_left",
    "change_right",
    "follow_lead",
)
MANEUVER_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(MANEUVER_OPTIONS)}
IDX_TO_MANEUVER: Dict[int, str] = {idx: name for idx, name in enumerate(MANEUVER_OPTIONS)}

SPEED_LEVEL_KMH: Dict[int, float] = {
    0: 0.0,   # Stop (0 km/h)
    1: 20.0,  # Slow (20 km/h - sharp curve / intersection)
    2: 40.0,  # Medium (40 km/h - normal driving / merge)
    3: 65.0,  # High (65 km/h - clear front visibility)
}


@dataclass
class ExpertLabels:
    """Ground-truth supervision labels for Laya's 3 decision primitives."""

    maneuver: str
    maneuver_idx: int
    speed_level: int
    emergency_brake: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "maneuver": self.maneuver,
            "maneuver_idx": int(self.maneuver_idx),
            "speed_level": int(self.speed_level),
            "emergency_brake": int(self.emergency_brake),
        }


class ExpertLabeler:
    """Rust-backed (`RustExpertLabeler`) translator from expert physical action + traffic state to Laya decision primitives."""

    def __init__(
        self,
        lane_change_steer_thresh: float = 0.22,
        aeb_decel_thresh: float = -0.75,
        aeb_distance_thresh_m: float = 13.5,
        follow_distance_thresh_m: float = 30.0,
        curve_curvature_thresh: float = 0.40,
    ) -> None:
        if rust_core_module is None:
            raise RuntimeError(
                "Rust extension `autonomous_vehicle_rust` is required. Build `rust_core` with `cargo build --release`."
            )
        self.lane_change_steer_thresh = float(lane_change_steer_thresh)
        self.aeb_decel_thresh = float(aeb_decel_thresh)
        self.aeb_distance_thresh_m = float(aeb_distance_thresh_m)
        self.follow_distance_thresh_m = float(follow_distance_thresh_m)
        self.curve_curvature_thresh = float(curve_curvature_thresh)
        self._rust_labeler = rust_core_module.RustExpertLabeler(
            self.lane_change_steer_thresh,
            self.aeb_decel_thresh,
            self.aeb_distance_thresh_m,
            self.follow_distance_thresh_m,
            self.curve_curvature_thresh,
        )

    def label_step(
        self,
        telemetry: Dict[str, Any],
        expert_action: Tuple[float, float],
        info: Dict[str, Any],
    ) -> ExpertLabels:
        """Derive ground-truth (maneuver, speed_level, emergency_brake) via native Rust `RustExpertLabeler`."""
        maneuver, maneuver_idx, speed_level, emergency_brake = (
            self._rust_labeler.label_step(
                telemetry,
                (float(expert_action[0]), float(expert_action[1])),
                info,
            )
        )
        return ExpertLabels(
            maneuver=str(maneuver),
            maneuver_idx=int(maneuver_idx),
            speed_level=int(speed_level),
            emergency_brake=int(emergency_brake),
        )
