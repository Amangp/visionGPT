"""Main BenchmarkSuite orchestrator for VisionGPT.

Provides the public API to benchmark hardware configurations, model parameters,
inference speed, memory footprint (VRAM/RAM), and training pass steps.
"""

import logging
import os
from typing import Any, Dict, List, Optional
import tensorflow as tf

from visiongpt.benchmark.gpu_monitor import get_gpu_status
from visiongpt.benchmark.memory import MemoryTracker
from visiongpt.benchmark.profiler import benchmark_training_speed
from visiongpt.benchmark.report import (
    export_reports,
    generate_plots,
    print_console_report,
)
from visiongpt.benchmark.speed import (
    benchmark_batch_sizes,
    generate_synthetic_inputs,
)
from visiongpt.benchmark.utils import (
    count_params,
    get_hardware_info,
    get_layer_count,
    get_model_size_mb,
)

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Production-ready benchmarking suite for VisionGPT v3.

    Integrates speed, memory, training profilers, hardware diagnostics,
    and model analysis into a single unified workspace tool.
    """

    def __init__(self, output_dir: str = "./benchmark_reports"):
        """Initialize the benchmark suite.

        Args:
            output_dir: Default destination directory for reports and plots.
        """
        self.output_dir = output_dir
        self.results: Dict[str, Any] = {
            "hardware": {},
            "model": {},
            "memory": {},
            "inference": {},
            "training": {},
        }

    def benchmark_hardware(self) -> Dict[str, Any]:
        """Perform system hardware diagnostics.

        Checks CPU, System RAM, operating system details, CUDA support,
        and visible GPU details.

        Returns:
            A dictionary containing system hardware configuration details.
        """
        logger.info("Executing hardware diagnostic benchmark...")
        hw_info = get_hardware_info()
        gpu_info = get_gpu_status()

        # Merge system and GPU details
        hw_info.update(
            {
                "gpu_available": gpu_info["gpu_available"],
                "gpu_name": gpu_info["gpu_name"],
                "total_vram_mb": gpu_info["total_vram_mb"],
            }
        )

        self.results["hardware"] = hw_info
        return hw_info

    def benchmark_model(self, model: tf.keras.Model) -> Dict[str, Any]:
        """Analyze model layers, weights, parameter counts, and sub-modules.

        Extracts parameters for the Vision Encoder, Fusion Layer, and Answer Decoder.

        Args:
            model: The tf.keras.Model or VisionGPT model to analyze.

        Returns:
            A dictionary containing parameter statistics of the model.
        """
        logger.info("Analyzing model architecture and parameters...")

        # Ensure model variables are initialized
        if not model.built:
            logger.debug("Building model weights for parameter analysis...")
            vocab_size = getattr(model, "vocab_size", 10000)
            images, text = generate_synthetic_inputs(batch_size=1, vocab_size=vocab_size)
            _ = model((images, text), training=False)

        total, trainable, non_trainable = count_params(model)
        size_mb = get_model_size_mb(model)
        layer_count = get_layer_count(model)

        # Retrieve specific sub-module parameter sizes
        encoder = getattr(model, "vision_encoder", None)
        decoder = getattr(model, "answer_decoder", None)
        fusion = getattr(model, "fusion_layer", None)

        encoder_params, _, _ = count_params(encoder)
        decoder_params, _, _ = count_params(decoder)
        fusion_params, _, _ = count_params(fusion)

        model_info = {
            "total_params": total,
            "trainable_params": trainable,
            "non_trainable_params": non_trainable,
            "size_mb": size_mb,
            "layer_count": layer_count,
            "encoder_params": encoder_params,
            "decoder_params": decoder_params,
            "fusion_params": fusion_params,
        }

        self.results["model"] = model_info
        return model_info

    def benchmark_inference(
        self,
        model: tf.keras.Model,
        batch_sizes: List[int] = [1, 2, 4, 8],
        num_warmup: int = 5,
        num_steps: int = 50,
    ) -> Dict[str, Any]:
        """Measure inference speed, latency, percentiles, and throughput.

        Runs benchmarks across multiple batch sizes.

        Args:
            model: Keras model to evaluate.
            batch_sizes: List of batch sizes to benchmark.
            num_warmup: Warm-up runs before timing steps.
            num_steps: Main iterations to record timings.

        Returns:
            A dictionary mapping batch sizes to inference speed metrics.
        """
        logger.info("Starting inference latency and throughput benchmark...")
        # Ensure model weights are built
        if not model.built:
            vocab_size = getattr(model, "vocab_size", 10000)
            images, text = generate_synthetic_inputs(batch_size=1, vocab_size=vocab_size)
            _ = model((images, text), training=False)

        inference_results = benchmark_batch_sizes(
            model=model,
            batch_sizes=batch_sizes,
            num_warmup=num_warmup,
            num_steps=num_steps,
        )

        # Convert keys to strings for JSON serialization
        serialized_results = {str(k): v for k, v in inference_results.items()}
        self.results["inference"] = serialized_results
        return serialized_results

    def benchmark_training(
        self,
        model: tf.keras.Model,
        batch_size: int = 2,
        steps: int = 5,
        enable_profiler: bool = False,
        steps_per_epoch: int = 100,
    ) -> Dict[str, Any]:
        """Benchmark training steps, isolating forward/backward/update times.

        Optionally logs trace events using the Keras Profiler.

        Args:
            model: Keras model to benchmark.
            batch_size: Input batch size.
            steps: Number of measured training steps.
            enable_profiler: True to dump TF Profiler details.
            steps_per_epoch: Number of steps to simulate a training epoch.

        Returns:
            A dictionary containing step duration statistics and throughput.
        """
        logger.info("Starting training step profiling...")
        profiler_logdir = os.path.join(self.output_dir, "profiler_logs")

        training_results = benchmark_training_speed(
            model=model,
            batch_size=batch_size,
            num_steps=steps,
            enable_profiler=enable_profiler,
            profiler_logdir=profiler_logdir,
            steps_per_epoch=steps_per_epoch,
        )

        self.results["training"] = training_results
        return training_results

    def benchmark_memory(
        self, model: tf.keras.Model, batch_size: int = 4
    ) -> Dict[str, Any]:
        """Perform a memory consumption audit (Peak RAM & VRAM Vitals) under load.

        Args:
            model: Keras model to trace.
            batch_size: Load batch size.

        Returns:
            A dictionary containing RAM and VRAM start/peak memory metrics.
        """
        logger.info("Starting memory allocation benchmark under load...")

        # Ensure model is built
        vocab_size = getattr(model, "vocab_size", 10000)
        if not model.built:
            images, text = generate_synthetic_inputs(batch_size=1, vocab_size=vocab_size)
            _ = model((images, text), training=False)

        images, text = generate_synthetic_inputs(batch_size, vocab_size=vocab_size)
        inputs = (images, text)

        # Track memory allocations during a dummy execution step
        with MemoryTracker() as tracker:
            # Execute training pass call to trigger maximum allocation context
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = tf.reduce_mean(tf.square(predictions))
            gradients = tape.gradient(loss, model.trainable_variables)
            # Force eager synchronization
            if hasattr(loss, "numpy"):
                _ = loss.numpy()
            valid_grads = [g for g in gradients if g is not None]
            if valid_grads and hasattr(valid_grads[0], "numpy"):
                _ = valid_grads[0].numpy()

        mem_results = {
            "start_ram_mb": tracker.start_ram,
            "end_ram_mb": tracker.end_ram,
            "peak_ram_mb": tracker.peak_ram,
            "start_vram_mb": tracker.start_gpu["current"],
            "end_vram_mb": tracker.end_gpu["current"],
            "peak_vram_mb": tracker.peak_gpu,
        }

        self.results["memory"] = mem_results
        return mem_results

    def benchmark_model_and_system(
        self,
        model: tf.keras.Model,
        batch_sizes: List[int] = [1, 2, 4, 8],
        training_batch_size: int = 2,
        training_steps: int = 5,
        enable_profiler: bool = False,
    ) -> Dict[str, Any]:
        """Convenience method to execute the full suite of benchmarks.

        Runs hardware, model, speed, training, and memory tests sequentially.

        Args:
            model: Keras model to evaluate.
            batch_sizes: Batch sizes for inference benchmarking.
            training_batch_size: Batch size for training benchmarking.
            training_steps: Steps for training benchmarking.
            enable_profiler: True to dump TF Profiler details.

        Returns:
            The consolidated results dictionary.
        """
        self.benchmark_hardware()
        self.benchmark_model(model)
        self.benchmark_memory(model, batch_size=max(batch_sizes))
        self.benchmark_inference(
            model, batch_sizes=batch_sizes, num_steps=50
        )
        self.benchmark_training(
            model,
            batch_size=training_batch_size,
            steps=training_steps,
            enable_profiler=enable_profiler,
        )
        return self.results

    def generate_report(self, output_dir: Optional[str] = None) -> None:
        """Serialize benchmark results and generate visualizations.

        Args:
            output_dir: Target output folder path. Defaults to self.output_dir.
        """
        target_dir = output_dir or self.output_dir
        logger.info("Generating benchmark reports and plots at %s", target_dir)

        # Write data files
        export_reports(self.results, target_dir)

        # Generate plots
        generate_plots(self.results, target_dir)

        # Display dashboard summary to stdout
        print_console_report(self.results)

    def save_results(self, filepath: str) -> None:
        """Directly save the raw JSON results dictionary to a filepath.

        Args:
            filepath: Destination file path.
        """
        logger.info("Saving raw results dict to %s", filepath)
        try:
            parent_dir = os.path.dirname(filepath)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(self.results, f, indent=4)
        except Exception as e:
            logger.error("Failed to save results dict: %s", e)
