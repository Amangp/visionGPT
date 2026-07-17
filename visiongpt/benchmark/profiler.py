"""Training performance profiler for VisionGPT.

This module profiles training step durations, forward and backward pass times,
gradient updates, and samples/sec throughput. It also integrates TF Profiler trace logging.
"""

import logging
import os
import time
from typing import Any, Dict, List
import tensorflow as tf

from visiongpt.benchmark.speed import generate_synthetic_inputs
from visiongpt.benchmark.utils import compute_statistics

logger = logging.getLogger(__name__)


def benchmark_training_speed(
    model: tf.keras.Model,
    batch_size: int = 2,
    num_steps: int = 5,
    seq_len: int = 5,
    enable_profiler: bool = False,
    profiler_logdir: str = "./logs/profiler",
    steps_per_epoch: int = 100,
) -> Dict[str, Any]:
    """Profile model training performance.

    Measures forward, backward, and optimization step times, and calculates
    throughput. Optionally runs the TensorFlow Profiler.

    Args:
        model: The tf.keras.Model to profile.
        batch_size: Batch size for inputs.
        num_steps: Number of training steps to measure.
        seq_len: Sequence length for text inputs.
        enable_profiler: Whether to activate the TF experimental profiler.
        profiler_logdir: Directory to save profiler trace files.
        steps_per_epoch: Simulated steps per epoch to estimate epoch time.

    Returns:
        A dictionary containing training metrics and profiles.
    """
    logger.info("Starting training benchmark (batch_size=%d, steps=%d)...", batch_size, num_steps)

    vocab_size = getattr(model, "vocab_size", 10000)
    images, text = generate_synthetic_inputs(batch_size, seq_len, vocab_size=vocab_size)
    inputs = (images, text)

    # Initialize model weights by running a dummy forward pass if not already built
    if not model.built:
        logger.debug("Building model weights with a dummy pass.")
        _ = model(inputs, training=False)

    trainable_vars = model.trainable_variables
    if not trainable_vars:
        # Re-try capturing variables
        _ = model(inputs, training=True)
        trainable_vars = model.trainable_variables

    if not trainable_vars:
        raise ValueError("Model does not have any trainable variables.")

    # Configure optimizer
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        logger.debug("No optimizer found on model. Using default Adam optimizer.")
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Activate TensorFlow Profiler if requested
    profiler_active = False
    if enable_profiler:
        try:
            os.makedirs(profiler_logdir, exist_ok=True)
            options = tf.profiler.experimental.ProfilerOptions(
                host_tracer_level=2, device_tracer_level=1, python_tracer_level=1
            )
            tf.profiler.experimental.start(profiler_logdir, options=options)
            profiler_active = True
            logger.info("TensorFlow Profiler started. Logs directed to %s", profiler_logdir)
        except Exception as e:
            logger.warning("Failed to start TensorFlow Profiler: %s", e)

    # Timers lists
    forward_times: List[float] = []
    backward_times: List[float] = []
    update_times: List[float] = []
    step_times: List[float] = []

    try:
        # Warmup step
        logger.debug("Running warmup training step.")
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = tf.reduce_mean(tf.square(predictions))
        gradients = tape.gradient(loss, trainable_vars)
        grads_and_vars = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
        optimizer.apply_gradients(grads_and_vars)

        # Main profiling loop
        for step in range(num_steps):
            step_start = time.perf_counter()

            # 1. Forward Pass
            fwd_start = time.perf_counter()
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = tf.reduce_mean(tf.square(predictions))
            # Force eager synchronization
            if hasattr(loss, "numpy"):
                _ = loss.numpy()
            fwd_end = time.perf_counter()
            forward_times.append((fwd_end - fwd_start) * 1000.0)

            # 2. Backward Pass
            bwd_start = time.perf_counter()
            gradients = tape.gradient(loss, trainable_vars)
            # Force eager synchronization on first gradient
            valid_grads = [g for g in gradients if g is not None]
            if valid_grads and hasattr(valid_grads[0], "numpy"):
                _ = valid_grads[0].numpy()
            bwd_end = time.perf_counter()
            backward_times.append((bwd_end - bwd_start) * 1000.0)

            # 3. Gradient Update
            upd_start = time.perf_counter()
            grads_and_vars = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
            optimizer.apply_gradients(grads_and_vars)
            # Force eager synchronization on first trainable variable
            if grads_and_vars and hasattr(grads_and_vars[0][1], "numpy"):
                _ = grads_and_vars[0][1].numpy()
            upd_end = time.perf_counter()
            update_times.append((upd_end - upd_start) * 1000.0)

            step_end = time.perf_counter()
            step_times.append((step_end - step_start) * 1000.0)

    finally:
        # Deactivate TensorFlow Profiler if running
        if profiler_active:
            try:
                tf.profiler.experimental.stop()
                logger.info("TensorFlow Profiler stopped successfully.")
            except Exception as e:
                logger.warning("Failed to stop TensorFlow Profiler: %s", e)

    # Compute statistics
    fwd_stats = compute_statistics(forward_times)
    bwd_stats = compute_statistics(backward_times)
    upd_stats = compute_statistics(update_times)
    step_stats = compute_statistics(step_times)

    mean_step_sec = step_stats["mean"] / 1000.0
    samples_per_sec = batch_size / mean_step_sec if mean_step_sec > 0 else 0.0
    epoch_time_sec = mean_step_sec * steps_per_epoch

    return {
        "batch_size": batch_size,
        "mean_forward_time_ms": fwd_stats["mean"],
        "mean_backward_time_ms": bwd_stats["mean"],
        "mean_gradient_update_time_ms": upd_stats["mean"],
        "mean_step_time_ms": step_stats["mean"],
        "epoch_time_sec": epoch_time_sec,
        "samples_per_sec": samples_per_sec,
        "step_times_ms": step_times,
    }
