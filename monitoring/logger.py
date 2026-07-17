"""Logging setup for VisionGPT monitoring module.

Directs monitoring and experiment logging messages to both the console
and `logs/monitoring.log` with standardized timestamp formatting.
"""

import logging
import os
from typing import Optional


def setup_monitoring_logger(
    log_dir: str = "logs", log_level: int = logging.INFO
) -> logging.Logger:
    """Configure and return the monitoring logger instance.

    Creates a logger that sends output to standard output and writes
    log messages to the file `logs/monitoring.log`.

    Args:
        log_dir: Directory where the log file is stored.
        log_level: Severity level for the logger (default: logging.INFO).

    Returns:
        The configured logging.Logger instance.
    """
    logger = logging.getLogger("visiongpt.monitoring")
    logger.setLevel(log_level)

    # Prevent duplicating handlers if already configured
    if logger.handlers:
        return logger

    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "monitoring.log")

    # Define standardized formatting
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Console stream handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # 2. File handler
    try:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback if file logging fails (e.g. permission error)
        print(f"Warning: Failed to setup monitoring log file handler: {e}")

    logger.propagate = False
    return logger
