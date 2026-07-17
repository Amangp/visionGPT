# Benchmarking & Performance Profiling

This document explains the VisionGPT v3 Benchmarking module, detailing performance metrics, profiling, and report outputs.

---

## 1. Benchmarking Suite Overview

The `benchmark` package is designed to profile the execution performance of VisionGPT models under load. It runs hardware detection, counts layer parameters, checks CPU/GPU memory footprint, and profiles inference latencies and training loops.

---

## 2. Monitored Performance Metrics

### A. Hardware & System
* **CPU Model Name:** Cross-platform processor brand string query.
* **System RAM Capacity:** Virtual memory size in GB.
* **CUDA Support:** Checks CUDA build integration and visible GPU models.

### B. Keras Model Statistics
* **Total Parameters:** Total weight variable sizes.
* **Trainable Parameters:** Variables that accumulate gradients.
* **Frozen Parameters:** Non-trainable features (e.g. pre-trained backbones).
* **Model Size (MB):** Space consumed by model float32 variables on memory.
* **Layers Count:** Recursive depth count of network sub-layers.

### C. Speed & Throughput
* **Average Latency (ms):** Mean time taken for single-batch model calls.
* **Inference Percentiles (ms):** P50, P90, P95, and P99 latency distribution markers.
* **Warm-up duration (sec):** Compilation/compilation overhead.
* **Throughput (Images/sec):** Rate of visual features processed under load.
* **Throughput (Questions/sec):** Rate of VQA generation.

### D. Training Profiles
* **Forward Pass Time (ms):** Time inside `tf.GradientTape()` context.
* **Backward Pass Time (ms):** Gradient calculation time.
* **Gradient Update Time (ms):** Optimizer apply gradients time.
* **Total Step Time (ms):** Sum duration of a complete training step.

### E. Memory Footprint
* **Process Peak RAM (MB):** Maximum memory footprint of the host process.
* **GPU VRAM Allocation (MB):** Baseline and peak VRAM allocated by TensorFlow.

---

## 3. How to Run the Benchmark

### Run script via CLI:
```bash
python -m benchmark.test_benchmark
```

### Run programmatically:
```python
from benchmark.benchmark import BenchmarkSuite
from models.visiongpt import VisionGPT

# Setup model
model = VisionGPT()

# Instantiate Benchmark Suite
suite = BenchmarkSuite(output_dir="./benchmark_reports")

# Run full evaluation
results = suite.benchmark_model_and_system(
    model,
    batch_sizes=[1, 2, 4, 8],
    training_batch_size=2,
    training_steps=10
)

# Export CSV/JSON and render Matplotlib graphs
suite.generate_report()
```

---

## 4. Visualizations & Plots

The suite generates three Matplotlib plots:
1. `latency.png`: Plots the inference latency progression over steps for each batch size.
2. `memory.png`: A bar chart comparing process Peak RAM and peak GPU VRAM.
3. `throughput.png`: A bar chart showing throughput queries/sec for each batch size.
