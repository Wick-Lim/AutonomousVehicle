"""FastAPI application factory with health/readiness probes, Prometheus metrics & session endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from src.api.cockpit_ui import COCKPIT_HTML
from src.control.rust_backend import BACKEND_NAME, RUST_CORE_AVAILABLE
from src.runtime.engine import DrivingRuntimeEngine, InvalidImageError
from src.schemas import (
    HealthResponse,
    MetricsSummaryResponse,
    PerceptionModeLiteral,
    SessionResetResponse,
    StepInferenceRequest,
    StepInferenceResponse,
    SystemInfoResponse,
    TelemetryInput,
)
from src.sim.studio import SimulationStudioManager


def create_app(
    config_path: Optional[str | Path] = None,
    use_mock: Optional[bool] = None,
    projector_ckpt: Optional[str | Path] = None,
    temperature_ckpt: Optional[str | Path] = None,
) -> FastAPI:
    """Create and configure the AutonomousVehicle FastAPI inference service."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        eff_mock = use_mock
        if eff_mock is None and "AV_USE_MOCK" in os.environ:
            eff_mock = os.environ["AV_USE_MOCK"].strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        engine = DrivingRuntimeEngine(
            config_path=config_path,
            use_mock=eff_mock,
            projector_ckpt=projector_ckpt,
            temperature_ckpt=temperature_ckpt,
        )
        engine.warmup(passes=2)
        app.state.engine = engine
        app.state.studio = SimulationStudioManager(
            engine=engine,
            default_simulator=os.environ.get("AV_DEFAULT_SIMULATOR", "metadrive"),
            force_mock=eff_mock,
        )
        try:
            yield
        finally:
            app.state.studio.close()

    app = FastAPI(
        title="AutonomousVehicle Closed-Loop Inference API",
        version="1.0.0",
        description="Real-time DINOv2 + SpatialProjector + Laya + Rust SafetyShieldPID service",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def attach_request_metadata(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        t0 = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    def _get_engine(request: Request) -> DrivingRuntimeEngine:
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            raise HTTPException(status_code=503, detail="Runtime engine not initialized")
        return engine

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz(request: Request) -> HealthResponse:
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            return HealthResponse(
                status="not_ready",
                ready=False,
                version="1.0.0",
                backend=BACKEND_NAME,
                rust_core_available=RUST_CORE_AVAILABLE,
                device="uninitialized",
                use_mock=bool(use_mock),
            )
        return HealthResponse(
            status="ok" if engine.ready else "not_ready",
            ready=engine.ready,
            version="1.0.0",
            backend=BACKEND_NAME,
            rust_core_available=RUST_CORE_AVAILABLE,
            device=str(engine.device),
            use_mock=engine.use_mock,
        )

    @app.get("/readyz", response_model=HealthResponse, tags=["ops"])
    async def readyz(request: Request):
        engine = getattr(request.app.state, "engine", None)
        if engine is None or not engine.ready:
            payload = HealthResponse(
                status="not_ready",
                ready=False,
                version="1.0.0",
                backend=BACKEND_NAME,
                rust_core_available=RUST_CORE_AVAILABLE,
                device="uninitialized",
                use_mock=bool(use_mock),
            )
            return JSONResponse(status_code=503, content=payload.model_dump())
        return HealthResponse(
            status="ok",
            ready=True,
            version="1.0.0",
            backend=BACKEND_NAME,
            rust_core_available=RUST_CORE_AVAILABLE,
            device=str(engine.device),
            use_mock=engine.use_mock,
        )

    @app.get("/v1/info", response_model=SystemInfoResponse, tags=["ops"])
    async def info(request: Request) -> SystemInfoResponse:
        return _get_engine(request).get_system_info()

    @app.get("/v1/metrics", response_model=MetricsSummaryResponse, tags=["ops"])
    async def metrics_json(request: Request) -> MetricsSummaryResponse:
        return _get_engine(request).get_metrics_summary()

    @app.get("/metrics", tags=["ops"])
    async def metrics_prometheus(request: Request) -> PlainTextResponse:
        body, content_type = _get_engine(request).metrics.render_prometheus()
        return PlainTextResponse(content=body.decode("utf-8"), media_type=content_type)

    @app.post("/v1/step", response_model=StepInferenceResponse, tags=["inference"])
    async def step_json(
        request: Request, payload: StepInferenceRequest
    ) -> StepInferenceResponse:
        engine = _get_engine(request)
        try:
            return engine.step(
                session_id=payload.session_id,
                image=payload.image_base64,
                telemetry=payload.telemetry,
                perception_mode=payload.perception_mode,
                shield_enabled=payload.shield_enabled,
                dt=payload.dt,
                strict_input=True,
            )
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/v1/step/multipart", response_model=StepInferenceResponse, tags=["inference"]
    )
    async def step_multipart(
        request: Request,
        image: UploadFile = File(...),
        telemetry_json: str = Form("{}"),
        session_id: str = Form("default"),
        perception_mode: PerceptionModeLiteral = Form("vision_estimated"),
        shield_enabled: bool = Form(True),
        dt: float = Form(0.05),
    ) -> StepInferenceResponse:
        engine = _get_engine(request)
        raw_bytes = await image.read()
        try:
            tel_dict = json.loads(telemetry_json)
            telemetry = TelemetryInput.model_validate(tel_dict)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid telemetry_json payload: {exc}"
            ) from exc

        try:
            return engine.step(
                session_id=session_id,
                image=raw_bytes,
                telemetry=telemetry,
                perception_mode=perception_mode,
                shield_enabled=shield_enabled,
                dt=dt,
                strict_input=True,
            )
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/v1/sessions/{session_id}/reset",
        response_model=SessionResetResponse,
        tags=["sessions"],
    )
    async def reset_session(
        request: Request, session_id: str
    ) -> SessionResetResponse:
        existed = _get_engine(request).reset_session(session_id)
        return SessionResetResponse(
            session_id=session_id, status="reset" if existed else "not_found"
        )

    @app.delete(
        "/v1/sessions/{session_id}",
        response_model=SessionResetResponse,
        tags=["sessions"],
    )
    async def delete_session(
        request: Request, session_id: str
    ) -> SessionResetResponse:
        existed = _get_engine(request).delete_session(session_id)
        return SessionResetResponse(
            session_id=session_id, status="deleted" if existed else "not_found"
        )

    def _get_studio(app_obj: FastAPI) -> SimulationStudioManager:
        studio = getattr(app_obj.state, "studio", None)
        if studio is None:
            raise HTTPException(status_code=503, detail="Simulation studio not initialized")
        return studio

    @app.get("/cockpit", response_class=HTMLResponse, tags=["cockpit"])
    async def cockpit_page() -> HTMLResponse:
        return HTMLResponse(content=COCKPIT_HTML)

    @app.get("/v1/sim/status", tags=["cockpit"])
    async def sim_status(request: Request) -> Dict[str, Any]:
        return _get_studio(request.app).get_status()

    @app.post("/v1/sim/control", tags=["cockpit"])
    async def sim_control(
        request: Request, payload: Dict[str, Any] = Body(...)
    ) -> Dict[str, Any]:
        studio = _get_studio(request.app)
        return await asyncio.to_thread(studio.apply_command, payload)

    @app.websocket("/ws/sim")
    async def sim_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        studio = _get_studio(websocket.app)
        await websocket.send_json({"type": "status", "data": studio.get_status()})
        try:
            while True:
                if studio.running:
                    try:
                        cmd = await asyncio.wait_for(websocket.receive_json(), timeout=0.04)
                        if isinstance(cmd, dict):
                            res = await asyncio.to_thread(studio.apply_command, cmd)
                            frame = res.pop("frame", None)
                            await websocket.send_json({"type": "status", "data": res})
                            if frame is not None:
                                await websocket.send_json({"type": "frame", "data": frame})
                    except asyncio.TimeoutError:
                        pass
                    if studio.running:
                        pkt = await asyncio.to_thread(studio.step_once)
                        await websocket.send_json({"type": "frame", "data": pkt})
                else:
                    cmd = await websocket.receive_json()
                    if isinstance(cmd, dict):
                        res = await asyncio.to_thread(studio.apply_command, cmd)
                        frame = res.pop("frame", None)
                        await websocket.send_json({"type": "status", "data": res})
                        if frame is not None:
                            await websocket.send_json({"type": "frame", "data": frame})
        except WebSocketDisconnect:
            return

    return app


def get_default_app() -> FastAPI:
    return create_app()


__all__ = ["create_app", "get_default_app"]
