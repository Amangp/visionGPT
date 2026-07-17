"""Inference speed benchmarking for VisionGPT.

This module measures inference latency, percentiles, throughput (images/sec),
and warm-up times across different batch sizes using high-precision timers.
"""

import logging
import time
from typing import Any, Dict, List
import numpy as np
import tensorflow as tf

from visiongpt.benchmark.utils import compute_statistics

logger = logging.getLogger(__name__)


def generate_synthetic_inputs(
    batch_size: int, seq_len: int = 5, vocab_size: int = 10000
) -> tuple[tf.Tensor, tf.Tensor]:
    """Generate synthetic image and text tokens for benchmarking.

    Args:
        batch_size: Size of the input batch.
        seq_len: Sequence length for the text tokens.
        vocab_size: Model vocabulary size limit.

    Returns:
        A tuple of (images_tensor, text_tensor).
    """
    images = tf.random.normal((batch_size, 224, 224, 3))
    # Tokens: Start token (3), random tokens, End token (4)
    words = np.random.randint(10, vocab_size - 1, size=(batch_size, seq_len - 2))
    starts = np.full((batch_size, 1), 3)
    ends = np.full((batch_size, 1), 4)
    text_data = np.concatenate([starts, words, ends], axis=1)
    text = tf.constant(text_data, dtype=tf.int64)
    return images, text


def benchmark_inference_speed(
    model: tf.keras.Model,
    batch_size: int = 1,
    seq_len: int = 5,
    num_warmup: int = 5,
    num_steps: int = 50,
) -> Dict[str, Any]:
    """Measure inference latency and throughput for a specific batch size.

    Forces synchronization by converting the output to a NumPy array in eager mode.

    Args:
        model: The tf.keras.Model to benchmark.
        batch_size: Batch size for inputs.
        seq_len: Sequence length for text inputs.
        num_warmup: Number of warm-up iterations.
        num_steps: Number of measured iterations.

    Returns:
        A dictionary containing speed, latency, and throughput statistics.
    """
    logger.info(
        "Benchmarking inference (batch_size=%d, steps=%d)...",
        batch_size,
        num_steps,
    )

    vocab_size = getattr(model, "vocab_size", 10000)
    images, text = generate_synthetic_inputs(batch_size, seq_len, vocab_size=vocab_size)
    inputs = (images, text)

    # Warm-up phase
    logger.debug("Starting warm-up phase of %d runs.", num_warmup)
    warmup_start = time.perf_counter()
    for _ in range(num_warmup):
        output = model(inputs, training=False)
        # Force evaluation in eager execution
        if hasattr(output, "numpy"):
            _ = output.numpy()
    warmup_duration = time.perf_counter() - warmup_start
    logger.debug("Warm-up completed in %.4f seconds.", warmup_duration)

    # Benchmark loop
    latencies: List[float] = []
    for step in range(num_steps):
        step_start = time.perf_counter()
        output = model(inputs, training=False)
        if hasattr(output, "numpy"):
            _ = output.numpy()
        step_end = time.perf_counter()
        latency_ms = (step_end - step_start) * 1000.0
        latencies.append(latency_ms)

    # Compute latency and throughput statistics
    stats = compute_statistics(latencies)
    mean_latency_sec = stats["mean"] / 1000.0

    images_per_sec = batch_size / mean_latency_sec if mean_latency_sec > 0 else 0.0
    questions_per_sec = batch_size / mean_latency_sec if mean_latency_sec > 0 else 0.0

    return {
        "batch_size": batch_size,
        "warmup_time_sec": warmup_duration,
        "latencies_ms": latencies,
        "mean_latency_ms": stats["mean"],
        "min_latency_ms": stats["min"],
        "max_latency_ms": stats["max"],
        "p50_latency_ms": stats["p50"],
        "p90_latency_ms": stats["p90"],
        "p95_latency_ms": stats["p95"],
        "p99_latency_ms": stats["p99"],
        "throughput_images_per_sec": images_per_sec,
        "throughput_questions_per_sec": questions_per_sec,
    }


def benchmark_batch_sizes(
    model: tf.keras.Model,
    batch_sizes: List[int],
    num_warmup: int = 5,
    num_steps: int = 50,
) -> Dict[int, Dict[str, Any]]:
    """Run inference benchmarking across a list of batch sizes.

    Args:
        model: The tf.keras.Model to benchmark.
        batch_sizes: List of batch sizes to evaluate.
        num_warmup: Number of warm-up iterations per configuration.
        num_steps: Number of measured iterations per configuration.

    Returns:
        A dictionary mapping batch size to its benchmarking results.
    """
    results = {}
    for bs in batch_sizes:
        try:
            results[bs] = benchmark_inference_speed(
                model=model,
                batch_size=bs,
                num_warmup=num_warmup,
                num_steps=num_steps,
            )
        except Exception as e:
            logger.error(
                "Failed to benchmark inference for batch size %d: %s", bs, e
            )
    return results
