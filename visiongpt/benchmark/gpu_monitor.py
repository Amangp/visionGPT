"""GPU monitoring utilities for VisionGPT.

This module retrieves detailed physical GPU metrics (model name, total/used/free VRAM)
by executing nvidi-smi and falls back to TensorFlow configuration APIs
for cross-platform and hardware compatibility.
"""

import logging
import subprocess
from typing import Any, Dict, List

import tensorflow as tf

logger = logging.getLogger(__name__)


def get_gpu_details() -> List[Dict[str, Any]]:
    """Retrieve detailed GPU metrics (name, total, used, free memory) using nvidia-smi.

    Returns:
        A list of dictionaries containing metrics for each visible NVIDIA GPU.
    """
    gpus_info: List[Dict[str, Any]] = []
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
        # Set a short timeout to avoid hanging if nvidia-smi is frozen
        output = subprocess.check_output(cmd, timeout=3.0).decode().strip()
        if not output:
            return gpus_info

        for line in output.split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus_info.append(
                    {
                        "name": parts[0],
                        "total_vram_mb": float(parts[1]),
                        "used_vram_mb": float(parts[2]),
                        "free_vram_mb": float(parts[3]),
                    }
                )
    except Exception as e:
        logger.debug(
            "nvidia-smi query failed (or not available on this platform): %s", e
        )

    return gpus_info


def get_gpu_status() -> Dict[str, Any]:
    """Retrieve consolidated GPU status details.

    Detects both TensorFlow visible devices and hardware details via nvidia-smi.

    Returns:
        A status dictionary indicating GPU model, availability, and memory stats.
    """
    physical_gpus = tf.config.list_physical_devices("GPU")
    details = get_gpu_details()

    gpu_available = len(physical_gpus) > 0
    status = {
        "gpu_available": gpu_available,
        "visible_gpu_count": len(physical_gpus),
        "details": details,
    }

    if details:
        # Aggregate info from nvidia-smi query
        status["gpu_name"] = details[0]["name"]
        status["total_vram_mb"] = sum(g["total_vram_mb"] for g in details)
        status["used_vram_mb"] = sum(g["used_vram_mb"] for g in details)
        status["free_vram_mb"] = sum(g["free_vram_mb"] for g in details)
    elif gpu_available:
        # Fallback to tf.config query if nvidia-smi is unavailable
        try:
            device_details = tf.config.experimental.get_device_details(
                physical_gpus[0]
            )
            status["gpu_name"] = device_details.get(
                "device_name", "Visible GPU"
            )
        except Exception:
            status["gpu_name"] = "Visible GPU"
        status["total_vram_mb"] = 0.0
        status["used_vram_mb"] = 0.0
        status["free_vram_mb"] = 0.0
    else:
        status["gpu_name"] = "N/A"
        status["total_vram_mb"] = 0.0
        status["used_vram_mb"] = 0.0
        status["free_vram_mb"] = 0.0

    return status
