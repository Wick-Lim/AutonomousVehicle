"""Structured logging utility supporting text and JSON (`AV_LOG_FORMAT=json`) modes."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "autonomous_vehicle") -> logging.Logger:
    """Configure and return a module logger with text or JSON formatting."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_str = os.environ.get("AV_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    log_format = os.environ.get("AV_LOG_FORMAT", "text").strip().lower()
    if log_format == "json":
        handler.setFormatter(_JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


__all__ = ["get_logger"]
