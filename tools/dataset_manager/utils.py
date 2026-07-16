"""
Utility functions for logging, size formatting, and SHA256 checksum computation.
"""

import hashlib
import logging
from pathlib import Path
from typing import Union


def setup_logger(log_dir: Path) -> logging.Logger:
    """
    Initializes a structured logger that outputs to both a file and the console.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "dataset_manager.log"

    logger = logging.getLogger("DatasetManager")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if re-initialized
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "[%(levelname)s] %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def format_size(size_bytes: int) -> str:
    """
    Formats byte size to human-readable format.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def compute_sha256(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """
    Computes the SHA256 hash of a file block-by-block to avoid memory exhaustion.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()
