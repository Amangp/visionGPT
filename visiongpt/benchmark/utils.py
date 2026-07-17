"""Utility functions for the VisionGPT benchmark suite.

This module provides hardware resource detection, Keras model parameter analysis,
and statistic computing helpers.
"""

import logging
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def get_cpu_info() -> str:
    """Detect the CPU model name in a cross-platform manner.

    Returns:
        The CPU model name or brand string.
    """
    try:
        sys_platform = platform.system()
        if sys_platform == "Windows":
            cmd = ["wmic", "cpu", "get", "name"]
            output = subprocess.check_output(cmd).decode().strip()
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            if len(lines) > 1:
                return lines[1]
        elif sys_platform == "Linux":
            cmd = "cat /proc/cpuinfo | grep 'model name' | uniq"
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            if ":" in output:
                return output.split(":", 1)[1].strip()
        elif sys_platform == "Darwin":
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.check_output(cmd).decode().strip()
    except Exception as e:
        logger.debug("Failed to retrieve detailed CPU brand: %s", e)

    return platform.processor() or "Unknown CPU"


def get_total_ram() -> float:
    """Get the total system RAM in gigabytes (GB).

    Returns:
        The total system RAM size.
    """
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        logger.debug("psutil is not installed. Falling back to system calls.")

    try:
        sys_platform = platform.system()
        if sys_platform == "Windows":
            cmd = ["wmic", "computersystem", "get", "totalphysicalmemory"]
            output = subprocess.check_output(cmd).decode().strip()
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            if len(lines) > 1:
                return float(lines[1]) / (1024**3)
        elif sys_platform == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        parts = line.split()
                        return float(parts[1]) / (1024**2)  # kB to GB
    except Exception as e:
        logger.debug("Failed to retrieve system physical memory: %s", e)

    return 0.0


def get_cuda_version() -> str:
    """Retrieve the installed CUDA version.

    Returns:
        The CUDA version string or 'N/A' if CUDA is unavailable.
    """
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode().strip()
        for line in output.split("\n"):
            if "release" in line:
                return line.split("release")[-1].strip()
    except Exception:
        pass

    # Check TensorFlow build info as fallback
    if hasattr(tf, "sysconfig") and callable(tf.sysconfig.get_build_info):
        try:
            build_info = tf.sysconfig.get_build_info()
            if "cuda_version" in build_info:
                return str(build_info["cuda_version"])
        except Exception:
            pass

    return "N/A"


def get_hardware_info() -> Dict[str, Any]:
    """Retrieve complete hardware and system diagnostics.

    Returns:
        A dictionary mapping hardware components to their specs.
    """
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names: List[str] = []
    for gpu in gpus:
        # Try to query the GPU device name if visible
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            if "device_name" in gpu_details:
                gpu_names.append(gpu_details["device_name"])
            else:
                gpu_names.append(str(gpu))
        except Exception:
            gpu_names.append(str(gpu))

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
        "cpu": get_cpu_info(),
        "ram_gb": round(get_total_ram(), 2),
        "cuda_version": get_cuda_version(),
        "cuda_available": tf.test.is_built_with_cuda(),
        "gpus_available_count": len(gpus),
        "gpu_devices": gpu_names,
    }


def count_params(layer: Any) -> Tuple[int, int, int]:
    """Compute (total, trainable, non-trainable) parameters for a layer/model.

    Args:
        layer: A tf.keras.layers.Layer or tf.keras.Model.

    Returns:
        A tuple of (total_params, trainable_params, non_trainable_params).
    """
    if layer is None or not hasattr(layer, "variables"):
        return 0, 0, 0

    total = sum(int(np.prod(v.shape)) for v in layer.variables)
    trainable = sum(int(np.prod(v.shape)) for v in layer.trainable_variables)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def get_model_size_mb(model: tf.keras.Model) -> float:
    """Calculate the memory footprint of model weights in MB.

    Args:
        model: The tf.keras.Model instance.

    Returns:
        The model size in Megabytes.
    """
    if model is None or not hasattr(model, "variables"):
        return 0.0

    total_bytes = 0
    for v in model.variables:
        try:
            # Try to get eager representation bytes
            total_bytes += v.numpy().nbytes
        except Exception:
            # Fallback estimation based on parameter shape and float32 size
            total_bytes += int(np.prod(v.shape)) * 4

    return total_bytes / (1024 * 1024)


def get_layer_count(layer: Any) -> int:
    """Recursively count the number of layers in a Keras Model or Layer.

    Args:
        layer: The layer or model instance.

    Returns:
        The recursive layer count.
    """
    if layer is None:
        return 0
    if not hasattr(layer, "layers") or not layer.layers:
        return 1
    return 1 + sum(get_layer_count(l) for l in layer.layers)


def compute_statistics(latencies: List[float]) -> Dict[str, float]:
    """Compute metric statistics (mean, min, max, P50, P90, P95, P99) for a list.

    Args:
        latencies: List of latency measurements (typically in milliseconds).

    Returns:
        A dictionary mapping statistic names to values.
    """
    if not latencies:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    arr = np.array(latencies)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
