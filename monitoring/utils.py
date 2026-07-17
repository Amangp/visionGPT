"""Utility functions for VisionGPT training monitoring.

Handles directory creation, Git revision tracking, and hardware configuration queries.
"""

import logging
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List
import tensorflow as tf

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Retrieve the current Git commit hash.

    Returns:
        The commit hash string, or "N/A" if git is not available or not in a repo.
    """
    try:
        # Run git rev-parse HEAD safely
        output = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return output
    except Exception:
        logger.debug("Failed to retrieve Git commit hash.")
        return "N/A"


def get_cpu_info() -> str:
    """Detect the CPU model name.

    Returns:
        The CPU model name.
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
        logger.debug("Failed to retrieve CPU model name: %s", e)

    return platform.processor() or "Unknown CPU"


def get_system_ram_gb() -> float:
    """Retrieve system physical RAM in GB.

    Returns:
        System RAM size in GB.
    """
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

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
                        return float(parts[1]) / (1024**2)
    except Exception:
        pass

    return 0.0


def get_hardware_diagnostics() -> Dict[str, Any]:
    """Retrieve hardware configuration data.

    Returns:
        A dictionary mapping hardware components to their attributes.
    """
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names: List[str] = []
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            gpu_names.append(details.get("device_name", "GPU Device"))
        except Exception:
            gpu_names.append(str(gpu))

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
        "cpu": get_cpu_info(),
        "ram_gb": round(get_system_ram_gb(), 2),
        "cuda_available": tf.test.is_built_with_cuda(),
        "gpus_available_count": len(gpus),
        "gpu_devices": gpu_names,
    }


def ensure_output_directories(base_dir: str = ".") -> Dict[str, str]:
    """Ensure all required monitoring output directories exist.

    Creates directories for results, plots, logs, tensorboard, reports, and history.

    Args:
        base_dir: Root directory under which folders are created.

    Returns:
        A dictionary of the absolute paths of the created folders.
    """
    folders = {
        "results": "results",
        "plots": "plots",
        "logs": "logs",
        "tensorboard": "tensorboard",
        "reports": "reports",
        "history": "history",
    }

    resolved_paths = {}
    for name, relative_path in folders.items():
        path = os.path.abspath(os.path.join(base_dir, relative_path))
        os.makedirs(path, exist_ok=True)
        resolved_paths[name] = path

    return resolved_paths
