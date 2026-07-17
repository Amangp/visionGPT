"""Memory profiling and tracking utilities for VisionGPT.

This module monitors RAM and GPU VRAM allocation, tracking current and peak
memory usage. It provides a robust, cross-platform context manager for memory auditing.
"""

import logging
import os
import platform
from typing import Dict

import tensorflow as tf

logger = logging.getLogger(__name__)


def get_current_ram() -> float:
    """Get the current process RAM usage in Megabytes (MB).

    Returns:
        The resident memory size of the process in MB.
    """
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        logger.debug("psutil not available for process memory extraction.")

    # Platform-specific fallback
    try:
        if platform.system() == "Linux":
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if "VmRSS" in line:
                        parts = line.split()
                        return float(parts[1]) / 1024.0  # kB to MB
    except Exception as e:
        logger.debug("Failed to extract VmRSS fallback memory: %s", e)

    return 0.0


def get_peak_ram() -> float:
    """Get the peak working set or RSS memory of the process in MB.

    Returns:
        The peak memory usage of the current process in MB.
    """
    try:
        if platform.system() == "Windows":
            import psutil

            mem_info = psutil.Process(os.getpid()).memory_info()
            if hasattr(mem_info, "peak_wset"):
                return mem_info.peak_wset / (1024 * 1024)
        else:
            import resource

            rusage = resource.getrusage(resource.RUSAGE_SELF)
            # maxrss is in KB on Linux, bytes on macOS
            factor = 1024.0 if platform.system() == "Linux" else (1024.0 * 1024.0)
            return rusage.ru_maxrss / factor
    except Exception as e:
        logger.debug("Failed to compute peak RAM: %s", e)

    return get_current_ram()


def get_tf_gpu_memory_info(device: str = "GPU:0") -> Dict[str, float]:
    """Retrieve the current and peak GPU memory allocated by TensorFlow.

    Args:
        device: The TensorFlow device string (e.g., 'GPU:0').

    Returns:
        A dictionary mapping 'current' and 'peak' memory in MB.
    """
    try:
        info = tf.config.experimental.get_memory_info(device)
        return {
            "current": info["current"] / (1024 * 1024),
            "peak": info["peak"] / (1024 * 1024),
        }
    except Exception:
        # Silently return zeros if GPU is not present or initialized
        return {"current": 0.0, "peak": 0.0}


class MemoryTracker:
    """Context manager to measure memory allocations (RAM & VRAM) during code blocks."""

    def __init__(self, device: str = "GPU:0"):
        """Initialize the tracker.

        Args:
            device: TensorFlow GPU device string to monitor.
        """
        self.device = device
        self.start_ram = 0.0
        self.end_ram = 0.0
        self.peak_ram = 0.0
        self.start_gpu = {"current": 0.0, "peak": 0.0}
        self.end_gpu = {"current": 0.0, "peak": 0.0}
        self.peak_gpu = 0.0

    def __enter__(self):
        """Reset stats and capture baseline memory usage."""
        try:
            tf.config.experimental.reset_memory_stats(self.device)
        except Exception:
            pass

        self.start_ram = get_current_ram()
        self.start_gpu = get_tf_gpu_memory_info(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Capture post-execution memory usage and peak metrics."""
        self.end_ram = get_current_ram()
        self.end_gpu = get_tf_gpu_memory_info(self.device)

        self.peak_ram = get_peak_ram()
        self.peak_gpu = self.end_gpu["peak"]

        # Log memory summary
        logger.info(
            "Memory usage summary - RAM Peak: %.2f MB, VRAM Peak: %.2f MB",
            self.peak_ram,
            self.peak_gpu,
        )
