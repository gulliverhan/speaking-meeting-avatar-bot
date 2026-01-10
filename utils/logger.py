"""Logging configuration for the meeting bot."""

import logging
import sys
from typing import Optional


def configure_logger(
    name: str = "recall-elevenlabs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: The name for the logger.
        level: The logging level.

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


# Default logger instance
logger = configure_logger()
