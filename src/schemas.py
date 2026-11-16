"""Typed Pydantic v2 runtime & FastAPI service contracts for AutonomousVehicle."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

_VALID_NAV = {"go_straight", "turn_left", "turn_right", "merge"}
_VALID_TL = {"green", "yellow", "red"}
_VALID_HAZ = {"none", "vehicle", "pedestrian", "red_light"}

PerceptionModeLiteral = Literal["vision_estimated", "vision_radar_fusion", "gt"]


def _sanitize_finite(value: Any, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


class TelemetryInput(BaseModel):
    """Sanitized vehicle telemetry contract with NaN/Inf resilience."""

    model_config = ConfigDict(extra="allow")

    speed_kmh: float = 0.0
    lane_offset_m: float = 0.0
    heading_error: float = 0.0
    left_lane_available: bool = True
    right_lane_available: bool = True
    nav_command: str = "go_straight"
    lead_distance_m: float = 60.0
    lead_speed_kmh: Optional[float] = 40.0
    pedestrian_distance_m: float = 100.0
    road_curvature: float = 0.0
    traffic_light: str = "green"
    hazard_source: str = "none"
    blind_spot_left: bool = False
    blind_spot_right: bool = False
    wet_road: bool = False
    a_max_mps2: Optional[float] = None
    uncertainty_margin_m: Optional[float] = None
    elapsed_steps: int = Field(default=1, ge=1)
    radar_targets: List[Dict[str, Any]] = Field(default_factory=list)
    imu_telemetry: Optional[Dict[str, Any]] = None

    @field_validator("speed_kmh", mode="before")
    @classmethod
    def _val_speed(cls, v: Any) -> float:
        return max(0.0, _sanitize_finite(v, 0.0))

    @field_validator("lane_offset_m", "heading_error", "road_curvature", mode="before")
    @classmethod
    def _val_finite_zero(cls, v: Any) -> float:
        return _sanitize_finite(v, 0.0)

    @field_validator("lead_distance_m", mode="before")
    @classmethod
    def _val_lead_dist(cls, v: Any) -> float:
        return max(0.0, _sanitize_finite(v, 60.0))

    @field_validator("pedestrian_distance_m", mode="before")
    @classmethod
    def _val_ped_dist(cls, v: Any) -> float:
        return max(0.0, _sanitize_finite(v, 100.0))

    @field_validator("lead_speed_kmh", mode="before")
    @classmethod
    def _val_lead_speed(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        return max(0.0, _sanitize_finite(v, 40.0))

    @field_validator("a_max_mps2", "uncertainty_margin_m", mode="before")
    @classmethod
    def _val_optional_finite(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        f = _sanitize_finite(v, 0.0)
        return f

    @field_validator("nav_command", mode="before")
    @classmethod
    def _val_nav(cls, v: Any) -> str:
        s = str(v or "go_straight").strip().lower()
        return s if s in _VALID_NAV else "go_straight"

    @field_validator("traffic_light", mode="before")
    @classmethod
    def _val_tl(cls, v: Any) -> str:
        s = str(v or "green").strip().lower()
        return s if s in _VALID_TL else "green"

    @field_validator("hazard_source", mode="before")
    @classmethod
    def _val_hazard(cls, v: Any) -> str:
        s = str(v or "none").strip().lower()
        return s if s in _VALID_HAZ else "none"

    def to_telemetry_dict(self) -> Dict[str, Any]:
        """Convert to normalized dict expected by LayaPilot and SafetyShieldPID."""
        data = self.model_dump(mode="python")
        wet = bool(data.get("wet_road", False))
        if data.get("a_max_mps2") is None:
            data["a_max_mps2"] = 4.5 if wet else 6.0
        if data.get("uncertainty_margin_m") is None:
            data["uncertainty_margin_m"] = 1.8 if wet else 0.0
        if data.get("lead_speed_kmh") is None:
            data.pop("lead_speed_kmh", None)
        return data


class AuxPerceptionSummary(BaseModel):
    pred_lead_distance_m: float
    pred_ped_distance_m: float
    pred_range_rate_mps: float
    pred_lane_offset_m: float
    pred_traffic_light: str
    pred_hazard_source: str
    pred_blindspot_left: float
    pred_blindspot_right: float
    sensor_health_score: float = 1.0
    fusion_active: bool = False


class DecisionSummary(BaseModel):
    maneuver: str
    maneuver_idx: int
    maneuver_confidence: float
    speed_level: int
    target_speed_kmh: float
    speed_confidence: float
    emergency_brake: float
    predicted_trajectory: List[List[float]]


class CarlaControlCommand(BaseModel):
    steer: float
    throttle: float
    brake: float
    hand_brake: bool = False
    reverse: bool = False


class ControlSummary(BaseModel):
    steering: float
    throttle_brake: float
    target_speed_kmh: float
    gated_maneuver: str
    gated_speed_level: int
    shield_triggered: bool
    shield_reason: str
    mpc_feedforward: float
    steer_ramp_substeps: List[float]
    carla_control: CarlaControlCommand


class StepInferenceRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=128)
    image_base64: Optional[str] = None
    telemetry: TelemetryInput = Field(default_factory=TelemetryInput)
    perception_mode: PerceptionModeLiteral = "vision_estimated"
    shield_enabled: bool = True
    dt: float = Field(default=0.05, gt=0.0, le=1.0)


class StepInferenceResponse(BaseModel):
    session_id: str
    step_index: int
    perception_mode: PerceptionModeLiteral
    backend: str
    latency_ms: float
    failsafe_triggered: bool = False
    decision: DecisionSummary
    aux_perception: AuxPerceptionSummary
    control: ControlSummary


class SessionResetResponse(BaseModel):
    session_id: str
    status: Literal["reset", "deleted", "not_found"]


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    ready: bool
    version: str
    backend: str
    rust_core_available: bool
    device: str
    use_mock: bool


class SystemInfoResponse(BaseModel):
    version: str
    trainable_parameters: int
    backend: str
    rust_core_available: bool
    device: str
    use_mock: bool
    projector_ckpt_loaded: Optional[str] = None
    temperature_ckpt_loaded: Optional[str] = None
    temperatures: Dict[str, float]
    active_sessions: int


class MetricsSummaryResponse(BaseModel):
    total_steps: int
    failsafe_triggers: int
    active_sessions: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    fps: float
    shield_interventions: Dict[str, int]


__all__ = [
    "AuxPerceptionSummary",
    "CarlaControlCommand",
    "ControlSummary",
    "DecisionSummary",
    "HealthResponse",
    "MetricsSummaryResponse",
    "PerceptionModeLiteral",
    "SessionResetResponse",
    "StepInferenceRequest",
    "StepInferenceResponse",
    "SystemInfoResponse",
    "TelemetryInput",
]
