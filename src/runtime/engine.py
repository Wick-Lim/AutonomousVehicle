"""Thread-safe multi-session closed-loop driving inference engine with failsafe recovery."""

from __future__ import annotations

import base64
import binascii
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import cv2
import numpy as np
import torch

from src.config import AppConfig, load_validated_config
from src.control.rust_backend import BACKEND_NAME, RUST_CORE_AVAILABLE, get_backend
from src.control.shield_pid import SafetyShieldPID
from src.models.laya_pilot import AutonomousDrivingSystem
from src.runtime.metrics import RuntimeMetrics
from src.schemas import (
    AuxPerceptionSummary,
    CarlaControlCommand,
    ControlSummary,
    DecisionSummary,
    MetricsSummaryResponse,
    PerceptionModeLiteral,
    StepInferenceResponse,
    SystemInfoResponse,
    TelemetryInput,
)
from src.sim.sensor_fusion import (
    MultiSensorFusionTracker,
    compute_camera_health,
    estimate_monocular_perception,
)
from src.utils.logging_utils import get_logger
from src.utils.seed import seed_everything

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOG = get_logger("autonomous_vehicle.runtime")


class InvalidImageError(ValueError):
    """Raised when client-supplied base64 or encoded image bytes cannot be decoded."""


def _default_vision_tel() -> Dict[str, Any]:
    return {
        "lead_distance_m": 60.0,
        "lead_speed_kmh": 35.0,
        "pedestrian_distance_m": 100.0,
        "lane_offset_m": 0.0,
        "traffic_light": "green",
        "hazard_source": "none",
        "blind_spot_left": False,
        "blind_spot_right": False,
    }


@dataclass
class SessionContext:
    """Isolated per-vehicle state (PIDs, OLS estimator, temporal tokens, blind-spot latch)."""

    session_id: str
    controller: SafetyShieldPID
    lead_speed_estimator: Any
    fusion_tracker: MultiSensorFusionTracker = field(
        default_factory=MultiSensorFusionTracker
    )
    prev_visual_tokens: Optional[torch.Tensor] = None
    bs_left_latch: int = 0
    bs_right_latch: int = 0
    prev_vision_tel: Dict[str, Any] = field(default_factory=_default_vision_tel)
    step_count: int = 0
    last_access_ts: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)


class DrivingRuntimeEngine:
    """Production runtime orchestrating DINOv2 + SpatialProjector + Laya + SafetyShieldPID."""

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        config: Optional[AppConfig | Dict[str, Any]] = None,
        use_mock: Optional[bool] = None,
        projector_ckpt: Optional[str | Path] = None,
        temperature_ckpt: Optional[str | Path] = None,
        max_sessions: int = 256,
        require_rust: bool = False,
    ) -> None:
        if isinstance(config, AppConfig):
            self.app_config = config
        elif isinstance(config, dict):
            self.app_config = AppConfig.model_validate(config)
        else:
            self.app_config = load_validated_config(config_path)

        self.cfg = self.app_config.to_dict()
        seed_everything(int(self.app_config.project.seed))

        self.use_mock = (
            bool(use_mock)
            if use_mock is not None
            else bool(self.app_config.project.use_mock)
        )
        self.cfg.setdefault("project", {})["use_mock"] = self.use_mock
        self.require_rust = bool(require_rust)

        dev_req = str(self.app_config.project.device)
        if dev_req == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_req)

        self.camera_height = int(self.app_config.env.camera_height)
        self.camera_width = int(self.app_config.env.camera_width)
        self.crop_top_ratio = float(self.app_config.env.crop_top_ratio)
        self.crop_bottom_ratio = float(self.app_config.env.crop_bottom_ratio)

        self._model_lock = threading.Lock()
        self._sessions_lock = threading.Lock()
        self._sessions: "OrderedDict[str, SessionContext]" = OrderedDict()
        self._max_sessions = max(1, int(max_sessions))
        self.metrics = RuntimeMetrics()
        self.ready: bool = False

        self.system = AutonomousDrivingSystem(
            config=self.cfg, use_mock=self.use_mock
        ).to(self.device)
        self.system.eval()

        self.trainable_parameters = sum(
            p.numel() for p in self.system.parameters() if p.requires_grad
        )

        self.projector_ckpt_loaded: Optional[str] = None
        self.aux_head_calibrated: bool = False
        proj_path = (
            Path(projector_ckpt)
            if projector_ckpt is not None
            else PROJECT_ROOT / self.app_config.train.projector_ckpt
        )
        if proj_path.is_file():
            try:
                ckpt = self.system.load_adapter_checkpoint(
                    proj_path, map_location=self.device, strict=False
                )
                self.projector_ckpt_loaded = str(proj_path)
                self.aux_head_calibrated = bool(
                    isinstance(ckpt, dict)
                    and any(
                        str(k).startswith("aux_head.")
                        for k in (ckpt.get("projector_state_dict") or {}).keys()
                    )
                )
            except Exception as exc:
                _LOG.warning(f"Failed loading projector checkpoint {proj_path}: {exc}")

        self.temperature_ckpt_loaded: Optional[str] = None
        temp_path = (
            Path(temperature_ckpt)
            if temperature_ckpt is not None
            else PROJECT_ROOT / self.app_config.train.temperature_ckpt
        )
        if temp_path.is_file():
            try:
                self.system.load_temperature_checkpoint(
                    temp_path, map_location=self.device
                )
                self.temperature_ckpt_loaded = str(temp_path)
            except Exception as exc:
                _LOG.warning(f"Failed loading temperature checkpoint {temp_path}: {exc}")

    def warmup(self, passes: int = 2) -> None:
        """Warm up PyTorch graph & allocator for consistent low-jitter latency."""
        dummy_img = np.zeros(
            (self.camera_height, self.camera_width, 3), dtype=np.uint8
        )
        dummy_tel = TelemetryInput(speed_kmh=35.0).to_telemetry_dict()
        with self._model_lock:
            self.system.eval()
            tensor_img = (
                torch.from_numpy(dummy_img)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=self.device, dtype=torch.float32)
            )
            with torch.no_grad():
                for _ in range(max(1, int(passes))):
                    _ = self.system(tensor_img, [dummy_tel])
            self.system.reset_temporal_buffer()
        self.ready = True

    def _preprocess_raw_image(
        self, image: Optional[Union[np.ndarray, bytes, str]]
    ) -> np.ndarray:
        """Decode & preprocess RGB image/bytes/base64 into `(camera_height, camera_width, 3)` uint8 RGB."""
        if image is None:
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)

        if isinstance(image, str):
            raw_b64 = image.strip()
            if "," in raw_b64 and raw_b64.startswith("data:image"):
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                raw_bytes = base64.b64decode(raw_b64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise InvalidImageError(f"Invalid base64 image payload: {exc}") from exc
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise InvalidImageError("Unable to decode base64 bytes as image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image, (bytes, bytearray)):
            arr = np.frombuffer(image, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise InvalidImageError("Unable to decode binary bytes as image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            rgb = image
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise InvalidImageError(
                    f"Expected HxWx3 RGB array, got shape {rgb.shape}"
                )
            if rgb.dtype != np.uint8:
                if float(rgb.max()) <= 1.0:
                    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            raise InvalidImageError(f"Unsupported image input type: {type(image)}")

        h, w = rgb.shape[:2]
        if (h, w) != (self.camera_height, self.camera_width):
            top = int(h * self.crop_top_ratio)
            bottom = int(h * (1.0 - self.crop_bottom_ratio))
            if bottom > top + 16:
                rgb = rgb[top:bottom, :]
            rgb = cv2.resize(
                rgb,
                (self.camera_width, self.camera_height),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def _get_or_create_session(self, session_id: str) -> SessionContext:
        backend = get_backend(require_rust=self.require_rust)
        with self._sessions_lock:
            ctx = self._sessions.get(session_id)
            if ctx is not None:
                self._sessions.move_to_end(session_id)
                ctx.last_access_ts = time.monotonic()
                return ctx

            while len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)

            ctx = SessionContext(
                session_id=session_id,
                controller=SafetyShieldPID(
                    config=self.cfg,
                    use_mock=self.use_mock,
                    require_rust=self.require_rust,
                ),
                lead_speed_estimator=backend.RustLeadSpeedEstimator(35.0),
                fusion_tracker=MultiSensorFusionTracker(
                    require_rust=self.require_rust
                ),
            )
            self._sessions[session_id] = ctx
            self.metrics.set_active_sessions(len(self._sessions))
            return ctx

    def reset_session(self, session_id: str) -> bool:
        with self._sessions_lock:
            ctx = self._sessions.get(session_id)
            if ctx is None:
                return False
        with ctx.lock:
            ctx.controller.reset()
            ctx.lead_speed_estimator.reset(35.0)
            ctx.fusion_tracker.reset(35.0)
            ctx.prev_visual_tokens = None
            ctx.bs_left_latch = 0
            ctx.bs_right_latch = 0
            ctx.prev_vision_tel = _default_vision_tel()
            ctx.step_count = 0
            ctx.last_access_ts = time.monotonic()
        return True

    def delete_session(self, session_id: str) -> bool:
        with self._sessions_lock:
            existed = session_id in self._sessions
            if existed:
                del self._sessions[session_id]
                self.metrics.set_active_sessions(len(self._sessions))
            return existed

    def _build_failsafe_response(
        self,
        session_id: str,
        step_index: int,
        perception_mode: PerceptionModeLiteral,
        latency_ms: float,
    ) -> StepInferenceResponse:
        self.metrics.record_step(
            latency_ms=latency_ms,
            perception_mode=perception_mode,
            maneuver="keep_lane",
            shield_triggered=True,
            shield_reason="runtime_failsafe_aeb",
            failsafe_triggered=True,
        )
        return StepInferenceResponse(
            session_id=session_id,
            step_index=step_index,
            perception_mode=perception_mode,
            backend=BACKEND_NAME,
            latency_ms=round(latency_ms, 3),
            failsafe_triggered=True,
            decision=DecisionSummary(
                maneuver="keep_lane",
                maneuver_idx=0,
                maneuver_confidence=1.0,
                speed_level=0,
                target_speed_kmh=0.0,
                speed_confidence=1.0,
                emergency_brake=1.0,
                predicted_trajectory=[],
            ),
            aux_perception=AuxPerceptionSummary(
                pred_lead_distance_m=0.0,
                pred_ped_distance_m=100.0,
                pred_range_rate_mps=0.0,
                pred_lane_offset_m=0.0,
                pred_traffic_light="green",
                pred_hazard_source="none",
                pred_blindspot_left=0.0,
                pred_blindspot_right=0.0,
            ),
            control=ControlSummary(
                steering=0.0,
                throttle_brake=-1.0,
                target_speed_kmh=0.0,
                gated_maneuver="keep_lane",
                gated_speed_level=0,
                shield_triggered=True,
                shield_reason="runtime_failsafe_aeb",
                mpc_feedforward=0.0,
                steer_ramp_substeps=[0.0, 0.0, 0.0, 0.0, 0.0],
                carla_control=CarlaControlCommand(
                    steer=0.0, throttle=0.0, brake=1.0, hand_brake=False, reverse=False
                ),
            ),
        )

    def step(
        self,
        session_id: str = "default",
        image: Optional[Union[np.ndarray, bytes, str]] = None,
        telemetry: Optional[Union[TelemetryInput, Dict[str, Any]]] = None,
        perception_mode: PerceptionModeLiteral = "vision_estimated",
        shield_enabled: bool = True,
        dt: float = 0.05,
        strict_input: bool = True,
    ) -> StepInferenceResponse:
        """Execute one closed-loop inference + perception + safety shield + PID step."""
        t0 = time.perf_counter()
        tel_model = (
            telemetry
            if isinstance(telemetry, TelemetryInput)
            else TelemetryInput.model_validate(telemetry or {})
        )
        base_tel = tel_model.to_telemetry_dict()

        try:
            rgb = self._preprocess_raw_image(image)
        except InvalidImageError:
            if strict_input:
                raise
            return self._build_failsafe_response(
                session_id=session_id,
                step_index=0,
                perception_mode=perception_mode,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        ctx = self._get_or_create_session(session_id)
        with ctx.lock:
            ctx.step_count += 1
            step_idx = ctx.step_count
            try:
                prompt_tel = dict(base_tel)
                if perception_mode in ("vision_estimated", "vision_radar_fusion"):
                    prompt_tel.update(ctx.prev_vision_tel)
                    prompt_tel["lane_offset_m"] = base_tel.get("lane_offset_m", 0.0)

                tensor_img = (
                    torch.from_numpy(rgb)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device=self.device, dtype=torch.float32)
                )

                elapsed_steps = max(1, int(base_tel.get("elapsed_steps", 1)))
                with self._model_lock:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    with torch.no_grad():
                        out = self.system(
                            tensor_img,
                            [prompt_tel],
                            elapsed_steps=elapsed_steps,
                            prev_visual_tokens=ctx.prev_visual_tokens,
                        )
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    ctx.prev_visual_tokens = out["visual_tokens"].detach().clone()

                m_idx = int(out["maneuver_idx"][0].item())
                s_lvl = int(out["speed_level"][0].item())
                m_conf = float(out["maneuver_confidence"][0].item())
                s_conf = float(out["speed_confidence"][0].item())
                aeb_prob = float(out["emergency_brake"][0].item())
                pred_traj_raw = out["predicted_trajectory"][0].detach().cpu().tolist()
                pred_traj = [
                    [round(float(p[0]), 2), round(float(p[1]), 3), round(float(p[2]), 2)]
                    for p in pred_traj_raw
                ]

                aux = out["aux_perception"]
                pred_d = float(aux["pred_lead_distance_m"][0].item())
                pred_ped_d = float(aux["pred_ped_distance_m"][0].item())
                pred_rr_mps = float(aux["pred_range_rate_mps"][0].item())
                pred_off = float(aux["pred_lane_offset_m"][0].item())
                tl_idx = int(aux["pred_traffic_light_idx"][0].item())
                hz_idx = int(aux["pred_hazard_idx"][0].item())
                tl_str = ("green", "yellow", "red")[min(2, max(0, tl_idx))]
                hz_str = ("none", "vehicle", "pedestrian", "red_light")[min(3, max(0, hz_idx))]
                bs_left_prob = float(aux["pred_blindspot_probs"][0, 0].item())
                bs_right_prob = float(aux["pred_blindspot_probs"][0, 1].item())

                ctrl_tel = dict(base_tel)
                sensor_health_score = float(
                    compute_camera_health(
                        rgb,
                        ego_speed_kmh=float(base_tel.get("speed_kmh", 0.0)),
                    )["health_score"]
                )
                fusion_active = False
                if perception_mode == "vision_estimated":
                    ego_v_kmh = float(base_tel.get("speed_kmh", 0.0))
                    if self.use_mock:
                        est_v_lead = float(
                            ctx.lead_speed_estimator.update(
                                pred_d,
                                pred_rr_mps,
                                ego_v_kmh,
                                elapsed_steps,
                                float(dt),
                            )
                        )
                        ctx.bs_left_latch = (
                            30 if bs_left_prob >= 0.5 else max(0, ctx.bs_left_latch - 1)
                        )
                        ctx.bs_right_latch = (
                            30 if bs_right_prob >= 0.5 else max(0, ctx.bs_right_latch - 1)
                        )
                        ctx.prev_vision_tel = {
                            "lead_distance_m": pred_d,
                            "lead_speed_kmh": est_v_lead,
                            "pedestrian_distance_m": pred_ped_d
                            if hz_str == "pedestrian"
                            else 100.0,
                            "lane_offset_m": pred_off,
                            "traffic_light": tl_str,
                            "hazard_source": hz_str,
                            "blind_spot_left": bool(ctx.bs_left_latch > 0),
                            "blind_spot_right": bool(ctx.bs_right_latch > 0),
                        }
                        ctrl_tel.update(ctx.prev_vision_tel)
                    else:
                        mono = estimate_monocular_perception(
                            aux_vision={
                                "pred_lead_distance_m": pred_d,
                                "pred_ped_distance_m": pred_ped_d,
                                "pred_range_rate_mps": pred_rr_mps,
                                "pred_lane_offset_m": pred_off,
                                "pred_traffic_light": tl_str,
                                "pred_hazard_source": hz_str,
                                "pred_blindspot_left": bs_left_prob,
                                "pred_blindspot_right": bs_right_prob,
                            },
                            base_telemetry=base_tel,
                            lead_speed_estimator=ctx.lead_speed_estimator,
                            radar_targets=base_tel.get("radar_targets"),
                            rgb=rgb,
                            prev_rgb=ctx.fusion_tracker._prev_rgb,
                            prev_lead_d=float(
                                ctx.prev_vision_tel.get("lead_distance_m", 60.0)
                            ),
                            bs_left_latch=ctx.bs_left_latch,
                            bs_right_latch=ctx.bs_right_latch,
                            step_idx=step_idx,
                            dt=float(dt),
                            elapsed_steps=elapsed_steps,
                            aux_calibrated=self.aux_head_calibrated,
                        )
                        if rgb is not None and isinstance(rgb, np.ndarray):
                            ctx.fusion_tracker._prev_rgb = rgb.copy()
                        ctx.bs_left_latch = int(mono["bs_left_latch"])
                        ctx.bs_right_latch = int(mono["bs_right_latch"])
                        sensor_health_score = float(mono["sensor_health_score"])
                        ctx.prev_vision_tel = {
                            "lead_distance_m": mono["lead_distance_m"],
                            "lead_speed_kmh": mono["lead_speed_kmh"],
                            "pedestrian_distance_m": mono["pedestrian_distance_m"],
                            "lane_offset_m": mono["lane_offset_m"],
                            "traffic_light": mono["traffic_light"],
                            "hazard_source": mono["hazard_source"],
                            "blind_spot_left": mono["blind_spot_left"],
                            "blind_spot_right": mono["blind_spot_right"],
                        }
                        ctrl_tel.update(ctx.prev_vision_tel)
                        pred_d = float(mono["lead_distance_m"])
                        pred_ped_d = float(mono["pedestrian_distance_m"])
                        pred_off = float(mono["lane_offset_m"])
                        tl_str = str(mono["traffic_light"])
                        hz_str = str(mono["hazard_source"])
                elif perception_mode == "vision_radar_fusion":
                    fused = ctx.fusion_tracker.fuse(
                        aux_vision={
                            "pred_lead_distance_m": pred_d,
                            "pred_ped_distance_m": pred_ped_d,
                            "pred_range_rate_mps": pred_rr_mps,
                            "pred_lane_offset_m": pred_off,
                            "pred_traffic_light": tl_str,
                            "pred_hazard_source": hz_str,
                            "pred_blindspot_left": bs_left_prob,
                            "pred_blindspot_right": bs_right_prob,
                        },
                        base_telemetry=base_tel,
                        radar_targets=base_tel.get("radar_targets"),
                        imu_telemetry=base_tel.get("imu_telemetry"),
                        rgb=rgb,
                        dt=float(dt),
                        elapsed_steps=elapsed_steps,
                        aux_calibrated=self.aux_head_calibrated,
                    )
                    sensor_health_score = float(fused["sensor_health_score"])
                    fusion_active = bool(fused["fusion_active"])
                    ctx.prev_vision_tel = {
                        "lead_distance_m": fused["lead_distance_m"],
                        "lead_speed_kmh": fused["lead_speed_kmh"],
                        "pedestrian_distance_m": fused["pedestrian_distance_m"],
                        "lane_offset_m": fused["lane_offset_m"],
                        "traffic_light": fused["traffic_light"],
                        "hazard_source": fused["hazard_source"],
                        "blind_spot_left": fused["blind_spot_left"],
                        "blind_spot_right": fused["blind_spot_right"],
                        "uncertainty_margin_m": fused["uncertainty_margin_m"],
                    }
                    ctrl_tel.update(ctx.prev_vision_tel)
                    pred_d = float(fused["lead_distance_m"])
                    pred_ped_d = float(fused["pedestrian_distance_m"])
                    pred_off = float(fused["lane_offset_m"])
                    tl_str = str(fused["traffic_light"])
                    hz_str = str(fused["hazard_source"])

                maneuver_name = (
                    "keep_lane",
                    "change_left",
                    "change_right",
                    "follow_lead",
                )[min(3, max(0, m_idx))]
                speed_tiers = {0: 0.0, 1: 20.0, 2: 40.0, 3: 65.0}
                decision_dict = {
                    "maneuver": maneuver_name,
                    "maneuver_idx": m_idx,
                    "maneuver_confidence": m_conf,
                    "speed_level": s_lvl,
                    "target_speed_kmh": speed_tiers.get(s_lvl, 40.0),
                    "speed_confidence": s_conf,
                    "emergency_brake": aeb_prob,
                    "predicted_trajectory": pred_traj,
                }

                (steering, throttle_brake), gated = ctx.controller.compute_control(
                    decision_dict,
                    ctrl_tel,
                    dt=float(dt),
                    shield_enabled=bool(shield_enabled),
                )

                if not (math.isfinite(steering) and math.isfinite(throttle_brake)):
                    raise RuntimeError("Non-finite control output detected")

                dt_ms = (time.perf_counter() - t0) * 1000.0
                gated_man = str(gated.get("maneuver", maneuver_name))
                shield_triggered = bool(gated.get("shield_triggered", False))
                shield_reason = str(gated.get("shield_reason", "none"))
                self.metrics.record_step(
                    latency_ms=dt_ms,
                    perception_mode=perception_mode,
                    maneuver=gated_man,
                    shield_triggered=shield_triggered,
                    shield_reason=shield_reason,
                    failsafe_triggered=False,
                )

                carla_ctrl = gated.get(
                    "carla_control",
                    ctx.controller.to_carla_control(steering, throttle_brake),
                )
                return StepInferenceResponse(
                    session_id=session_id,
                    step_index=step_idx,
                    perception_mode=perception_mode,
                    backend=BACKEND_NAME,
                    latency_ms=round(dt_ms, 3),
                    failsafe_triggered=False,
                    decision=DecisionSummary(
                        maneuver=maneuver_name,
                        maneuver_idx=m_idx,
                        maneuver_confidence=round(m_conf, 4),
                        speed_level=s_lvl,
                        target_speed_kmh=speed_tiers.get(s_lvl, 40.0),
                        speed_confidence=round(s_conf, 4),
                        emergency_brake=round(aeb_prob, 4),
                        predicted_trajectory=pred_traj,
                    ),
                    aux_perception=AuxPerceptionSummary(
                        pred_lead_distance_m=round(pred_d, 2),
                        pred_ped_distance_m=round(pred_ped_d, 2),
                        pred_range_rate_mps=round(pred_rr_mps, 2),
                        pred_lane_offset_m=round(pred_off, 3),
                        pred_traffic_light=tl_str,
                        pred_hazard_source=hz_str,
                        pred_blindspot_left=round(bs_left_prob, 3),
                        pred_blindspot_right=round(bs_right_prob, 3),
                        sensor_health_score=round(sensor_health_score, 3),
                        fusion_active=fusion_active,
                    ),
                    control=ControlSummary(
                        steering=round(float(steering), 4),
                        throttle_brake=round(float(throttle_brake), 4),
                        target_speed_kmh=round(float(gated.get("target_speed_kmh", 0.0)), 2),
                        gated_maneuver=gated_man,
                        gated_speed_level=int(gated.get("speed_level", s_lvl)),
                        shield_triggered=shield_triggered,
                        shield_reason=shield_reason,
                        mpc_feedforward=float(gated.get("mpc_feedforward", 0.0)),
                        steer_ramp_substeps=[
                            float(x) for x in gated.get("steer_ramp_substeps", [])
                        ],
                        carla_control=CarlaControlCommand.model_validate(carla_ctrl),
                    ),
                )
            except Exception as exc:
                _LOG.error(
                    f"Runtime exception in session '{session_id}' at step {step_idx}; engaging failsafe AEB: {exc}"
                )
                return self._build_failsafe_response(
                    session_id=session_id,
                    step_index=step_idx,
                    perception_mode=perception_mode,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )

    def get_system_info(self) -> SystemInfoResponse:
        with self._sessions_lock:
            active_cnt = len(self._sessions)
        temps = {
            "t_maneuver": round(float(self.system.laya.temp_maneuver.item()), 4),
            "t_speed": round(float(self.system.laya.temp_speed.item()), 4),
            "t_aeb": round(float(self.system.laya.temp_aeb.item()), 4),
            "b_aeb": round(float(self.system.laya.bias_aeb.item()), 4),
        }
        return SystemInfoResponse(
            version="1.0.0",
            trainable_parameters=self.trainable_parameters,
            backend=BACKEND_NAME,
            rust_core_available=RUST_CORE_AVAILABLE,
            device=str(self.device),
            use_mock=self.use_mock,
            projector_ckpt_loaded=self.projector_ckpt_loaded,
            temperature_ckpt_loaded=self.temperature_ckpt_loaded,
            temperatures=temps,
            active_sessions=active_cnt,
        )

    def get_metrics_summary(self) -> MetricsSummaryResponse:
        return MetricsSummaryResponse.model_validate(self.metrics.summary())


__all__ = ["DrivingRuntimeEngine", "InvalidImageError", "SessionContext"]
