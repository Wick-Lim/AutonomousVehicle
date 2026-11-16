# Multi-stage production Dockerfile for AutonomousVehicle inference service
FROM rust:1.82-slim AS rust-builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev && rm -rf /var/lib/apt/lists/*
COPY rust_core/ /build/rust_core/
RUN cargo build --release --manifest-path /build/rust_core/Cargo.toml

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AV_LOG_FORMAT=json \
    AV_REQUIRE_RUST=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      torch torchvision transformers accelerate safetensors \
      numpy opencv-python-headless Pillow PyYAML tqdm einops \
      pydantic fastapi "uvicorn[standard]" prometheus-client python-multipart

COPY configs/ /app/configs/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY checkpoints/ /app/checkpoints/
COPY --from=rust-builder /build/rust_core/target/release/libautonomous_vehicle_rust.so /app/src/control/autonomous_vehicle_rust.so

RUN useradd --create-home --shell /bin/bash avuser && chown -R avuser:avuser /app
USER avuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["python", "-m", "src.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
