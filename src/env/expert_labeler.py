"""Expert Labeler: Converts IDMPolicy physical actions and state into Laya decision labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


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
    """Rule-based translator from expert physical action + traffic state to Laya decision primitives."""

    def __init__(
        self,
        lane_change_steer_thresh: float = 0.22,
        aeb_decel_thresh: float = -0.75,
        aeb_distance_thresh_m: float = 13.5,
        follow_distance_thresh_m: float = 30.0,
        curve_curvature_thresh: float = 0.40,
    ) -> None:
        self.lane_change_steer_thresh = lane_change_steer_thresh
        self.aeb_decel_thresh = aeb_decel_thresh
        self.aeb_distance_thresh_m = aeb_distance_thresh_m
        self.follow_distance_thresh_m = follow_distance_thresh_m
        self.curve_curvature_thresh = curve_curvature_thresh

    def label_step(
        self,
        telemetry: Dict[str, Any],
        expert_action: Tuple[float, float],
        info: Dict[str, Any],
    ) -> ExpertLabels:
        """Derive ground-truth (maneuver, speed_level, emergency_brake) from expert step."""
        steering, throttle_brake = float(expert_action[0]), float(expert_action[1])
        speed_kmh = float(telemetry.get("speed_kmh", 0.0))
        left_avail = bool(telemetry.get("left_lane_available", False))
        right_avail = bool(telemetry.get("right_lane_available", False))
        nav_command = str(telemetry.get("nav_command", "go_straight"))

        lead_distance_m = float(info.get("lead_distance_m", telemetry.get("lead_distance_m", 60.0)))
        road_curvature = abs(float(info.get("road_curvature", telemetry.get("road_curvature", 0.0))))
        scenario_event = str(info.get("scenario_event", ""))
        traffic_light = str(telemetry.get("traffic_light", info.get("traffic_light", "green")))
        hazard_source = str(telemetry.get("hazard_source", info.get("hazard_source", "none")))

        # 1. Determine emergency_brake (noul: 0 or 1) - includes CARLA red light & pedestrian crosswalk AEB
        is_real_sim = scenario_event in ("real_metadrive", "real_carla")
        is_aeb = (
            scenario_event == "emergency_brake"
            or traffic_light == "red"
            or (hazard_source == "pedestrian" and lead_distance_m <= 16.0)
            or (
                not is_real_sim
                and throttle_brake <= self.aeb_decel_thresh
                and lead_distance_m <= self.aeb_distance_thresh_m
            )
            or (lead_distance_m < 9.5 and speed_kmh > 15.0)
        )
        emergency_brake = 1 if is_aeb else 0

        # 2. Determine maneuver (choice: keep_lane, change_left, change_right, follow_lead)
        expert_maneuver = info.get("expert_maneuver") or telemetry.get("expert_maneuver")
        if scenario_event in MANEUVER_TO_IDX:
            maneuver = scenario_event
        elif expert_maneuver in ("change_left", "change_right"):
            maneuver = expert_maneuver
        elif is_aeb:
            maneuver = "follow_lead"
        elif (
            not is_real_sim
            and steering >= self.lane_change_steer_thresh
            and left_avail
            and road_curvature < self.curve_curvature_thresh
        ):
            maneuver = "change_left"
        elif (
            not is_real_sim
            and steering <= -self.lane_change_steer_thresh
            and right_avail
            and road_curvature < self.curve_curvature_thresh
        ):
            maneuver = "change_right"
        elif lead_distance_m <= self.follow_distance_thresh_m:
            maneuver = "follow_lead"
        else:
            maneuver = "keep_lane"

        maneuver_idx = MANEUVER_TO_IDX[maneuver]

        # 3. Determine speed_level (score: 0=0km/h, 1=20km/h, 2=40km/h, 3=65km/h)
        if emergency_brake == 1:
            speed_level = 0
        elif scenario_event == "sharp_curve" or road_curvature >= self.curve_curvature_thresh:
            speed_level = 1
        elif (
            maneuver in ("change_left", "change_right", "follow_lead")
            or nav_command in ("merge", "turn_left", "turn_right")
            or road_curvature >= 0.22
        ):
            speed_level = 2 if lead_distance_m > 16.0 else 1
        elif lead_distance_m >= 45.0 and road_curvature < 0.22:
            speed_level = 3
        else:
            speed_level = 2

        return ExpertLabels(
            maneuver=maneuver,
            maneuver_idx=maneuver_idx,
            speed_level=speed_level,
            emergency_brake=emergency_brake,
        )
