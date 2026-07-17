# VisionGPT v3 Technical Documentation Index

Welcome to the technical documentation library for **VisionGPT v3**. This folder contains comprehensive, production-ready documentation covering our deep learning model alongside the dataset, evaluation, benchmark, monitoring, and visualization packages.

---

## 1. Documentation Index

Explore our detailed markdown manuals below:

* **[Installation Manual (INSTALLATION.md)](INSTALLATION.md)**  
  Step-by-step setup guide for conda virtual environments, CPU/GPU TensorFlow configurations, CUDA Toolkit and cuDNN libraries, and system sanity tests.

* **[Datasets & Preprocessing (DATASETS.md)](DATASETS.md)**  
  Data structures and folder mappings for MS COCO 2017, GQA, TextVQA, and Visual Genome. Outlines the dataset downloader CLI tool and SHA256 integrity validation checks.

* **[Training Pipeline (TRAINING.md)](TRAINING.md)**  
  Details training configurations including mixed-precision, Custom Keras Metrics callbacks, checkpointers, resumptions, and TensorBoard logging.

* **[Inference execution (INFERENCE.md)](INFERENCE.md)**  
  Interactive CLI scripts, Python API calls, batch inputs, and token decoding patterns.

* **[Architecture Specifications (ARCHITECTURE.md)](ARCHITECTURE.md)**  
  Comprehensive study of model parameters, shapes, and layer connectivity: EfficientNet-B0 Visual Encoder, projections Fusion layer, and masked self-attention Answer Decoder.

* **[API Reference (API.md)](API.md)**  
  Full API manual detailing methods, constructors, parameters, and code examples for all core classes (`VisionGPT`, `Evaluator`, `BenchmarkSuite`, `ExperimentLogger`).

* **[Benchmarking Guide (BENCHMARKS.md)](BENCHMARKS.md)**  
  Detailed instructions for running speed throughput, latency percentiles, VRAM peak allocations, and step profiles on CPU/GPU systems.

* **[Evaluation Framework (EVALUATION.md)](EVALUATION.md)**  
  Deep dive into metrics: unigram/n-gram precisions, BLEU-1 to BLEU-4, METEOR word alignments, ROUGE-L LCS dynamic programming, and CIDEr TF-IDF calculations.

* **[Troubleshooting FAQ (FAQ.md)](FAQ.md)**  
  Solutions to common GPU Out-Of-Memory (OOM) errors, CUDA dll missing errors, download timeout errors, and checkpoint loading warnings.

* **[Contributing Guidelines (CONTRIBUTING.md)](CONTRIBUTING.md)**  
  Coding standard rules (PEP8 style, type hint, unit test), branching strategies, and pull requests.

* **[Changelog (CHANGELOG.md)](CHANGELOG.md)**  
  Tracks version releases and updates from base models to v3.0.0.

* **[Open Source License (LICENSE.md)](LICENSE.md)**  
  The complete Apache 2.0 open-source license.

---

## 2. Directory Layout & Imports Map

When building custom components for VisionGPT, imports are mapped relative to the workspace root:

```python
# Import core model layers
from models.visiongpt import VisionGPT

# Import metrics evaluator
from evaluation.evaluator import Evaluator

# Import system hardware and latency benchmarks
from benchmark.benchmark import BenchmarkSuite

# Import real-time training callback loggers
from monitoring.callbacks import VisionGPTTrainingCallback

# Import layout diagram visualization visualizer
from visualization.architecture import ArchitectureVisualizer
```
