"""Pydantic v2 validated configuration schema and loader for AutonomousVehicle."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "AutonomousVehicle"
    seed: int = Field(default=42, ge=0)
    device: Literal["auto", "cuda", "cpu"] = "auto"
    use_mock: bool = False


class CarlaEnvConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    host: str = "127.0.0.1"
    port: int = Field(default=2000, ge=1, le=65535)
    tm_port: int = Field(default=8000, ge=1, le=65535)
    town: str = "current"
    vehicle_blueprint: str = "vehicle.tesla.model3"
    camera_fov: float = Field(default=100.0, gt=10.0, lt=180.0)
    fixed_delta_seconds: float = Field(default=0.05, gt=0.0, le=1.0)
    num_npc_vehicles: int = Field(default=25, ge=0)
    num_pedestrians: int = Field(default=15, ge=0)
    target_route_distance_m: float = Field(default=180.0, gt=0.0)


class EnvConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    simulator: Literal["metadrive", "carla"] = "metadrive"
    camera_height: int = Field(default=196, ge=14)
    camera_width: int = Field(default=392, ge=14)
    crop_top_ratio: float = Field(default=0.25, ge=0.0, lt=0.85)
    crop_bottom_ratio: float = Field(default=0.10, ge=0.0, lt=0.85)
    decision_repeat: int = Field(default=5, ge=1)
    traffic_density: float = Field(default=0.10, ge=0.0, le=1.0)
    map_blocks: int = Field(default=3, ge=1)
    start_seed: int = Field(default=1000, ge=0)
    eval_start_seed: int = Field(default=5000, ge=0)
    eval_episodes: int = Field(default=50, ge=1)
    max_episode_steps: int = Field(default=750, ge=1)
    lane_width: float = Field(default=3.5, gt=1.0, le=10.0)
    carla: CarlaEnvConfig = Field(default_factory=CarlaEnvConfig)

    @model_validator(mode="after")
    def _validate_crop_ratios(self) -> "EnvConfig":
        if (self.crop_top_ratio + self.crop_bottom_ratio) >= 0.85:
            raise ValueError(
                "env.crop_top_ratio + env.crop_bottom_ratio must be strictly < 0.85"
            )
        return self


class DinoConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name: str = "facebook/dinov2-base"
    patch_size: int = Field(default=14, ge=1)
    embed_dim: int = Field(default=768, ge=1)
    grid_h: int = Field(default=14, ge=1)
    grid_w: int = Field(default=28, ge=1)
    num_patches: int = Field(default=392, ge=1)
    freeze: bool = True

    @model_validator(mode="after")
    def _validate_patch_count(self) -> "DinoConfig":
        if self.grid_h * self.grid_w != self.num_patches:
            raise ValueError(
                f"dino.grid_h ({self.grid_h}) * dino.grid_w ({self.grid_w}) "
                f"must equal dino.num_patches ({self.num_patches})"
            )
        return self


class ProjectorConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    in_dim: int = Field(default=768, ge=1)
    out_dim: int = Field(default=1024, ge=1)
    hidden_dim: int = Field(default=1024, ge=1)
    pool_h: int = Field(default=4, ge=1)
    pool_w: int = Field(default=9, ge=1)
    num_visual_tokens: int = Field(default=36, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validate_pool_grid(self) -> "ProjectorConfig":
        if self.pool_h * self.pool_w != self.num_visual_tokens:
            raise ValueError(
                f"projector.pool_h ({self.pool_h}) * projector.pool_w ({self.pool_w}) "
                f"must equal projector.num_visual_tokens ({self.num_visual_tokens})"
            )
        return self


class LayaConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name: str = "convaiinnovations/laya"
    embed_dim: int = Field(default=1024, ge=1)
    max_seq_len: int = Field(default=256, ge=64)
    freeze: bool = True
    use_lora: bool = False
    lora_r: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)


class PrimitivesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    maneuver_options: List[str] = Field(
        default_factory=lambda: ["keep_lane", "change_left", "change_right", "follow_lead"]
    )
    speed_levels: Dict[int, float] = Field(
        default_factory=lambda: {0: 0.0, 1: 20.0, 2: 40.0, 3: 65.0}
    )


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    dino: DinoConfig = Field(default_factory=DinoConfig)
    projector: ProjectorConfig = Field(default_factory=ProjectorConfig)
    laya: LayaConfig = Field(default_factory=LayaConfig)
    primitives: PrimitivesConfig = Field(default_factory=PrimitivesConfig)

    @model_validator(mode="after")
    def _validate_dims_match(self) -> "ModelConfig":
        if self.projector.in_dim != self.dino.embed_dim:
            raise ValueError("model.projector.in_dim must match model.dino.embed_dim")
        if self.projector.out_dim != self.laya.embed_dim:
            raise ValueError("model.projector.out_dim must match model.laya.embed_dim")
        return self


class SafetyShieldConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    aeb_threshold: float = Field(default=0.70, gt=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.80, gt=0.0, le=1.0)


class LateralPIDConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    kp: float = Field(default=0.45, ge=0.0)
    ki: float = Field(default=0.01, ge=0.0)
    kd: float = Field(default=0.18, ge=0.0)
    heading_weight: float = Field(default=1.25, ge=0.0)
    max_steer_rate_per_sec: float = Field(default=3.2, gt=0.0)
    steer_smoothing_alpha: float = Field(default=0.72, gt=0.0, le=1.0)


class LongitudinalPIDConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    kp: float = Field(default=0.08, ge=0.0)
    ki: float = Field(default=0.005, ge=0.0)
    kd: float = Field(default=0.02, ge=0.0)


class ControlConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    safety_shield: SafetyShieldConfig = Field(default_factory=SafetyShieldConfig)
    lateral_pid: LateralPIDConfig = Field(default_factory=LateralPIDConfig)
    longitudinal_pid: LongitudinalPIDConfig = Field(default_factory=LongitudinalPIDConfig)


class OversamplingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    keep_lane_weight: float = Field(default=1.0, gt=0.0)
    change_left_weight: float = Field(default=4.5, gt=0.0)
    change_right_weight: float = Field(default=4.5, gt=0.0)
    follow_lead_weight: float = Field(default=2.5, gt=0.0)
    aeb_bonus_weight: float = Field(default=5.0, ge=1.0)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    data_dir: str = "data"
    train_split: float = Field(default=0.85, gt=0.0, lt=1.0)
    num_collect_frames: int = Field(default=50000, ge=1)
    oversampling: OversamplingConfig = Field(default_factory=OversamplingConfig)


class LossWeightsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    maneuver: float = Field(default=1.0, ge=0.0)
    speed_level: float = Field(default=1.0, ge=0.0)
    emergency_brake: float = Field(default=2.0, ge=0.0)
    trajectory: float = Field(default=0.5, ge=0.0)
    aux_perception: float = Field(default=0.2, ge=0.0)
    readout_aux: float = Field(default=0.1, ge=0.0)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    checkpoint_dir: str = "checkpoints"
    projector_ckpt: str = "checkpoints/projector_best.pt"
    temperature_ckpt: str = "checkpoints/temperature.pt"
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=0, ge=0)
    epochs: int = Field(default=15, ge=1)
    lr: float = Field(default=0.0003, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    loss_weights: LossWeightsConfig = Field(default_factory=LossWeightsConfig)


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    num_bins: int = Field(default=15, ge=2, le=100)
    max_iter: int = Field(default=100, ge=1)
    lr: float = Field(default=0.05, gt=0.0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    @model_validator(mode="after")
    def _validate_vision_patch_geometry(self) -> "AppConfig":
        patch = self.model.dino.patch_size
        h = self.env.camera_height
        w = self.env.camera_width
        if h % patch != 0 or w % patch != 0:
            raise ValueError(
                f"env.camera_height ({h}) and env.camera_width ({w}) "
                f"must both be divisible by model.dino.patch_size ({patch})"
            )
        if h // patch != self.model.dino.grid_h or w // patch != self.model.dino.grid_w:
            raise ValueError(
                f"env.camera dimensions ({h}x{w}) / patch_size ({patch}) "
                f"must equal model.dino grid ({self.model.dino.grid_h}x{self.model.dino.grid_w})"
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return plain Python dict representation compatible with downstream modules."""
        return self.model_dump(mode="python")


def _parse_bool_env(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def load_validated_config(
    config_path: Optional[str | Path] = None,
    apply_env_overrides: bool = True,
) -> AppConfig:
    """Load YAML config, validate invariants via Pydantic v2, and optionally apply `AV_*` env overrides."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    if apply_env_overrides:
        if "AV_DEVICE" in os.environ:
            raw.setdefault("project", {})["device"] = os.environ["AV_DEVICE"].strip().lower()
        if "AV_USE_MOCK" in os.environ:
            raw.setdefault("project", {})["use_mock"] = _parse_bool_env(
                os.environ["AV_USE_MOCK"]
            )
        if "AV_SIMULATOR" in os.environ:
            raw.setdefault("env", {})["simulator"] = os.environ["AV_SIMULATOR"].strip().lower()

    return AppConfig.model_validate(raw)


__all__ = [
    "AppConfig",
    "CalibrationConfig",
    "CarlaEnvConfig",
    "ControlConfig",
    "DEFAULT_CONFIG_PATH",
    "DataConfig",
    "DinoConfig",
    "EnvConfig",
    "LayaConfig",
    "ModelConfig",
    "ProjectConfig",
    "ProjectorConfig",
    "SafetyShieldConfig",
    "TrainConfig",
    "load_validated_config",
]
