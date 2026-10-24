"""Environment wrappers (MetaDrive & CARLA) and expert imitation labeler."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.env.carla_wrapper import CarlaWrapper
from src.env.expert_labeler import ExpertLabeler, ExpertLabels
from src.env.metadrive_wrapper import MetaDriveWrapper, TelemetryState


def create_driving_env(
    config: Optional[Dict[str, Any]] = None,
    use_mock: bool = False,
    start_seed: Optional[int] = None,
    num_scenarios: int = 50,
    simulator: Optional[str] = None,
) -> MetaDriveWrapper | CarlaWrapper:
    """Factory returning `MetaDriveWrapper` or `CarlaWrapper` based on config or CLI argument."""
    cfg = config or {}
    sim_name = (
        simulator
        if simulator is not None
        else str(cfg.get("env", {}).get("simulator", "metadrive"))
    ).lower()
    if sim_name == "carla":
        return CarlaWrapper(
            config=cfg,
            use_mock=use_mock,
            start_seed=start_seed,
            num_scenarios=num_scenarios,
        )
    return MetaDriveWrapper(
        config=cfg,
        use_mock=use_mock,
        start_seed=start_seed,
        num_scenarios=num_scenarios,
    )


__all__ = [
    "CarlaWrapper",
    "ExpertLabeler",
    "ExpertLabels",
    "MetaDriveWrapper",
    "TelemetryState",
    "create_driving_env",
]
