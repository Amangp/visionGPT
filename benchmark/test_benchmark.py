"""Unit tests and verification script for the VisionGPT Benchmark Suite.

Instantiates a VisionGPT model and runs the full benchmarking suite,
checking report file outputs, statistics calculations, and profiling runs.
"""

import os
import shutil
import unittest
import numpy as np
import tensorflow as tf

# Import benchmark modules
from visiongpt.benchmark.benchmark import BenchmarkSuite
from models.visiongpt import VisionGPT


class TestBenchmarkSuite(unittest.TestCase):
    """Test case suite for the VisionGPT Benchmark Module."""

    @classmethod
    def setUpClass(cls):
        """Set up testing directories and dummy model."""
        cls.test_dir = "./test_benchmark_reports"
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Instantiate actual VisionGPT model from workspace
        cls.model = VisionGPT(
            vocab_size=1000,
            embed_dim=64,  # Smaller embedding dimension for fast test execution
            context_dropout_rate=0.10,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up generated files after testing."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_hardware_benchmark(self):
        """Test system hardware details query."""
        suite = BenchmarkSuite(output_dir=self.test_dir)
        hw_info = suite.benchmark_hardware()
        
        self.assertIn("os", hw_info)
        self.assertIn("cpu", hw_info)
        self.assertIn("ram_gb", hw_info)
        self.assertIn("tensorflow_version", hw_info)
        self.assertIn("gpu_available", hw_info)

    def test_model_benchmark(self):
        """Test model parameter statistics counting."""
        suite = BenchmarkSuite(output_dir=self.test_dir)
        model_info = suite.benchmark_model(self.model)
        
        self.assertIn("total_params", model_info)
        self.assertIn("trainable_params", model_info)
        self.assertIn("size_mb", model_info)
        self.assertIn("layer_count", model_info)
        self.assertIn("encoder_params", model_info)
        self.assertIn("decoder_params", model_info)
        self.assertIn("fusion_params", model_info)
        
        # Values should be non-zero for initialized model
        self.assertGreater(model_info["total_params"], 0)
        self.assertGreater(model_info["encoder_params"], 0)
        self.assertGreater(model_info["decoder_params"], 0)

    def test_inference_benchmark(self):
        """Test speed and latency benchmarking on inference paths."""
        suite = BenchmarkSuite(output_dir=self.test_dir)
        inf_results = suite.benchmark_inference(
            model=self.model,
            batch_sizes=[1, 2],
            num_warmup=2,
            num_steps=5
        )
        
        self.assertIn("1", inf_results)
        self.assertIn("2", inf_results)
        self.assertEqual(inf_results["1"]["batch_size"], 1)
        self.assertIn("mean_latency_ms", inf_results["1"])
        self.assertIn("p50_latency_ms", inf_results["1"])
        self.assertIn("throughput_images_per_sec", inf_results["1"])

    def test_training_benchmark(self):
        """Test training step time profiling."""
        suite = BenchmarkSuite(output_dir=self.test_dir)
        train_results = suite.benchmark_training(
            model=self.model,
            batch_size=2,
            steps=3,
            enable_profiler=False,
            steps_per_epoch=10
        )
        
        self.assertEqual(train_results["batch_size"], 2)
        self.assertIn("mean_forward_time_ms", train_results)
        self.assertIn("mean_backward_time_ms", train_results)
        self.assertIn("mean_gradient_update_time_ms", train_results)
        self.assertIn("mean_step_time_ms", train_results)
        self.assertIn("samples_per_sec", train_results)
        self.assertEqual(len(train_results["step_times_ms"]), 3)

    def test_memory_benchmark(self):
        """Test memory audit allocations."""
        suite = BenchmarkSuite(output_dir=self.test_dir)
        mem_results = suite.benchmark_memory(self.model, batch_size=2)
        
        self.assertIn("peak_ram_mb", mem_results)
        self.assertIn("start_ram_mb", mem_results)
        self.assertIn("peak_vram_mb", mem_results)
        self.assertGreaterEqual(mem_results["peak_ram_mb"], 0.0)

    def test_report_generation(self):
        """Test reports and visualizations generation."""
        suite = BenchmarkSuite(output_dir=self.test_dir)
        
        # Populate results
        suite.benchmark_hardware()
        suite.benchmark_model(self.model)
        suite.benchmark_memory(self.model, batch_size=2)
        suite.benchmark_inference(self.model, batch_sizes=[1, 2], num_warmup=1, num_steps=2)
        suite.benchmark_training(self.model, batch_size=2, steps=2)
        
        # Generate files
        suite.generate_report()
        
        # Verify file existence
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "benchmark.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "benchmark.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "benchmark.txt")))
        
        # Plots (if matplotlib is available, they will be saved)
        try:
            import matplotlib
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, "latency.png")))
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, "memory.png")))
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, "throughput.png")))
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
