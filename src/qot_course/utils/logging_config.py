"""Console logging setup for the course. No output is ever silenced."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger with a single console handler.

    Parameters
    ----------
    name : str
        Logger name (usually ``__name__``).
    level : str
        Logging level, e.g. ``"INFO"`` or ``"DEBUG"``.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:  # idempotent: never add duplicate handlers
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    return logger
