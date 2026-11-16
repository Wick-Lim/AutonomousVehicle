"""Prometheus and rolling latency telemetry metrics for the runtime engine."""

from __future__ import annotations

import threading
from collections import Counter as PyCounter
from collections import deque
from typing import Deque, Dict

import numpy as np
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class RuntimeMetrics:
    """Thread-safe metrics registry using an isolated Prometheus `CollectorRegistry`."""

    def __init__(self, max_latency_samples: int = 2000) -> None:
        self._lock = threading.Lock()
        self.registry = CollectorRegistry(auto_describe=True)
        self._latencies_ms: Deque[float] = deque(maxlen=max(10, int(max_latency_samples)))
        self._total_steps: int = 0
        self._failsafe_triggers: int = 0
        self._shield_counts: PyCounter[str] = PyCounter()
        self._active_sessions_count: int = 0

        self.step_latency_hist = Histogram(
            "av_step_latency_ms",
            "End-to-end single-frame inference + control step latency in milliseconds",
            buckets=(1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 250.0),
            registry=self.registry,
        )
        self.steps_total = Counter(
            "av_steps_total",
            "Total closed-loop control steps executed",
            labelnames=("perception_mode", "maneuver"),
            registry=self.registry,
        )
        self.shield_interventions_total = Counter(
            "av_shield_interventions_total",
            "Total Safety Shield interventions by reason",
            labelnames=("reason",),
            registry=self.registry,
        )
        self.failsafe_triggers_total = Counter(
            "av_failsafe_triggers_total",
            "Total runtime failsafe AEB recoveries triggered",
            registry=self.registry,
        )
        self.active_sessions = Gauge(
            "av_active_sessions",
            "Current number of active vehicle control sessions",
            registry=self.registry,
        )

    def set_active_sessions(self, count: int) -> None:
        with self._lock:
            self._active_sessions_count = max(0, int(count))
            self.active_sessions.set(self._active_sessions_count)

    def record_step(
        self,
        latency_ms: float,
        perception_mode: str,
        maneuver: str,
        shield_triggered: bool,
        shield_reason: str,
        failsafe_triggered: bool = False,
    ) -> None:
        lat = max(0.0, float(latency_ms))
        with self._lock:
            self._total_steps += 1
            self._latencies_ms.append(lat)
            self.step_latency_hist.observe(lat)
            self.steps_total.labels(
                perception_mode=perception_mode, maneuver=maneuver
            ).inc()
            if shield_triggered:
                reason_key = str(shield_reason or "unknown")
                self._shield_counts[reason_key] += 1
                self.shield_interventions_total.labels(reason=reason_key).inc()
            if failsafe_triggered:
                self._failsafe_triggers += 1
                self.failsafe_triggers_total.inc()

    def summary(self) -> Dict[str, object]:
        with self._lock:
            lats = list(self._latencies_ms)
            if lats:
                arr = np.asarray(lats, dtype=np.float64)
                mean_ms = float(np.mean(arr))
                p50_ms = float(np.percentile(arr, 50))
                p95_ms = float(np.percentile(arr, 95))
                p99_ms = float(np.percentile(arr, 99))
                fps = 1000.0 / max(1e-3, mean_ms)
            else:
                mean_ms = p50_ms = p95_ms = p99_ms = fps = 0.0

            return {
                "total_steps": self._total_steps,
                "failsafe_triggers": self._failsafe_triggers,
                "active_sessions": self._active_sessions_count,
                "mean_latency_ms": round(mean_ms, 3),
                "p50_latency_ms": round(p50_ms, 3),
                "p95_latency_ms": round(p95_ms, 3),
                "p99_latency_ms": round(p99_ms, 3),
                "fps": round(fps, 2),
                "shield_interventions": dict(self._shield_counts),
            }

    def render_prometheus(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


__all__ = ["RuntimeMetrics"]
