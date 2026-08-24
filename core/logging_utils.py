from __future__ import annotations

import logging as pylogging
from typing import Any

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def configure_logging(level: str = "INFO") -> None:
    if pylogging.getLogger().handlers:
        pylogging.getLogger().setLevel(level.upper())
        return
    pylogging.basicConfig(level=level.upper(), format=_DEFAULT_FORMAT)


def get_logger(name: str, *, level: str | None = None) -> pylogging.Logger:
    logger = pylogging.getLogger(name)
    if level is not None:
        logger.setLevel(level.upper())
    return logger


def log_kv(logger: pylogging.Logger, message: str, **values: Any) -> None:
    suffix = " ".join(f"{key}={value!r}" for key, value in values.items())
    logger.info("%s%s", message, f" | {suffix}" if suffix else "")
