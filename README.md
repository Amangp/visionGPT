# VisionGPT v3: Vision Language Model & ML Infrastructure Suite

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](docs/LICENSE.md)

VisionGPT v3 is a state-of-the-art production-grade Vision-Language Model (VLM) built in TensorFlow 2.x and Python 3.10+. It implements an end-to-end deep learning model alongside a robust set of infrastructure modules for dataset management, automated evaluation, hardware benchmarking, real-time training dashboards, and publication-ready architecture diagram generation.

---

## 1. Project Architecture Overview

The VisionGPT model coordinates a visual encoder, a projections fusion layer, and an autoregressive text decoder to decode text tokens from fused visual features:

```
                  ┌───────────────┐
                  │  Input Image  │
                  └───────┬───────┘
                          │ (224x224x3 RGB)
                          ▼
            ┌───────────────────────────┐
            │  EfficientNet-B0 Encoder  │  <-- Frozen Feature Extractor
            └─────────────┬─────────────┘
                          │ (7x7x1280 Spatial Grid)
                          ▼
            ┌───────────────────────────┐
            │   Feature Fusion Layer    │  <-- Linear Dimension Projection
            └─────────────┬─────────────┘
                          │ (49x256 Token Sequence)
                          ▼
            ┌───────────────────────────┐
            │    Transformer Decoder    │  <-- Autoregressive Text Generator
            └─────────────┬─────────────┘
                          │ (Multi-Head Cross-Attention)
                          ▼
            ┌───────────────────────────┐
            │     Output Softmax        │  <-- Probability Vocabulary Classifier
            └─────────────┬─────────────┘
                          │
                          ▼
                  [Caption / Answer]
```

For a detailed analysis of our model layers and shapes, see the [Architecture Document](docs/ARCHITECTURE.md).

---

## 2. Directory Structure

```
visiongpt/
├── evaluation/         # Metrics (BLEU, METEOR, ROUGE-L, CIDEr, Exact Match)
├── benchmark/          # Speed, latency, and memory profiling suite
├── monitoring/         # Real-time training Keras callbacks & experiment dashboards
├── visualization/      # Matplotlib architecture layout and flowcharts generators
├── models/             # Core layers (VisionEncoder, FusionLayer, AnswerDecoder)
├── dataset_tools/      # Dataset verification and reader utilities
├── tools/
│   └── dataset_manager/# Concurrent dataset downloader CLI
└── docs/               # Detailed documentation files
```

---

## 3. Key Features

1. **Vision-Language Task Core:** Capable of Image Captioning, Visual Question Answering (VQA), and OCR-aware QA.
2. **Dataset Manager CLI:** Downloads, extracts, and checksum-verifies COCO 2017, TextVQA, GQA, and Visual Genome.
3. **Automated Evaluation:** Fully implementations for BLEU (1-4), ROUGE-L, METEOR, CIDEr, and VQA Accuracy (Exact Match, Token F1, Word Levenshtein).
4. **Hardware & Model Benchmarking:** Profiles throughput (images/sec), latency percentiles (P50, P90, P99), RAM peak RSS, GPU VRAM peak, and isolations for forward/backward/update loops.
5. **Real-time Callback Dashboards:** Standardized Keras training callback pushing scalars and weight histograms to TensorBoard, exporting CSV/JSON histories, and updating a responsive HTML dashboard.
6. **Architecture Diagram Generator:** Outputs publication-quality vertical layers flowcharts, horizontal pipelines, and parameter pie charts in PNG, PDF, and SVG formats across light, dark, and monochrome themes.

---

## 4. Documentation Index

To get started, explore the detailed guides below:

* **[Installation Guide](docs/INSTALLATION.md)**: Set up Python, TensorFlow, and GPU CUDA/cuDNN variables.
* **[Dataset Reference](docs/DATASETS.md)**: Explore manager commands for COCO, TextVQA, GQA, and Visual Genome.
* **[Training Pipeline](docs/TRAINING.md)**: Mixed precision, callbacks, checkpointing, and resume instructions.
* **[Inference execution](docs/INFERENCE.md)**: Interactive and batch CLI calls and Python API usage.
* **[Architecture Specifications](docs/ARCHITECTURE.md)**: Sub-module parameters, activations, and tensor flows.
* **[API Reference](docs/API.md)**: Public interfaces for Evaluator, BenchmarkSuite, and ExperimentLogger.
* **[Benchmark Guide](docs/BENCHMARKS.md)**: Running performance profiles on CPU/GPU hardware.
* **[Evaluation Module](docs/EVALUATION.md)**: Math details of BLEU, METEOR, ROUGE, and CIDEr.
* **[Troubleshooting FAQ](docs/FAQ.md)**: Out-of-memory errors, CUDA warnings, and dataset extraction failures.
* **[Contributing Guidelines](docs/CONTRIBUTING.md)**: PEP8 style compliance, pull requests, and test runs.
* **[Changelog](docs/CHANGELOG.md)**: Release histories and updates.
* **[Apache 2.0 License](docs/LICENSE.md)**: Open-source terms.

---

## 5. Quick Start Examples

### A. Run Hardware and Speed Benchmarks
```python
import models.visiongpt as vgpt
import benchmark.benchmark as bmark

# Instantiate and build model
model = vgpt.VisionGPT()
suite = bmark.BenchmarkSuite(output_dir="./reports")

# Run full speed and VRAM peak benchmarks
results = suite.benchmark_model_and_system(model, batch_sizes=[1, 2, 4])
suite.generate_report()
```

### B. Evaluate Text Generation Metrics
```python
import evaluation.evaluator as eval_mod

evaluator = eval_mod.Evaluator()
predictions = ["a brown cat sitting on a table", "a red bus on the street"]
references = ["a cat sitting on the table", "a bus on a street"]

scores = evaluator.evaluate(predictions, references)
print("BLEU-4:", scores["BLEU-4"])
print("CIDEr:", scores["CIDEr"])
```

### C. Train with Live Dashboards & Callbacks
```python
from monitoring.experiment import ExperimentLogger
from monitoring.callbacks import VisionGPTTrainingCallback

logger = ExperimentLogger(experiment_name="coco_efficientnet_dec", epochs=10)
callback = VisionGPTTrainingCallback(experiment_logger=logger)

model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[callback]
)
```

### D. Generate Publication Diagrams
```python
from visualization.architecture import ArchitectureVisualizer

visualizer = ArchitectureVisualizer(model=model, output_dir=".")
visualizer.generate_architecture(theme="monochrome") # monochrome paper layout
visualizer.export_png("architecture.png")
visualizer.export_pdf("architecture.pdf") # vector format
```

---

## 6. Citation

If you use VisionGPT v3 in your research, please cite:

```bibtex
@software{visiongpt_v3_2026,
  author = {VisionGPT Team},
  title = {VisionGPT v3: An Infrastructure and Deep Learning Suite for Vision-Language Models},
  year = {2026},
  url = {https://github.com/Amangp/visionGPT}
}
```

---

## 7. License

Distributed under the Apache 2.0 License. See the [LICENSE](docs/LICENSE.md) file for complete details.
