"""VisionGPT Benchmark Suite package.

This package provides a comprehensive, production-ready benchmarking suite for VisionGPT v3.
It allows profiling model parameters, hardware resources, inference speed/latency,
memory allocations (RAM and GPU VRAM), and training step metrics.
"""

from visiongpt.benchmark.benchmark import BenchmarkSuite

__all__ = ["BenchmarkSuite"]
