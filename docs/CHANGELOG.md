# Changelog

This document tracks changes and releases for VisionGPT.

---

## [3.0.0] - 2026-07-17

### Added
* **Automated Evaluation Module:** Added implementations for BLEU (1-4), METEOR, ROUGE-L, CIDEr, and VQA accuracies.
* **Hardware Benchmarking Suite:** Added hardware detection, system speed calculations, latency percentiles, step profiling, and reports.
* **Real-time Experiment Logger & Callbacks:** Added Keras training callbacks, TensorBoard scalars, weight histograms logging, CSV/JSON log exports, and a responsive glassmorphism HTML dashboard.
* **Architecture Diagram Generator:** Added custom Matplotlib layout generators to export publication-quality flowcharts, horizontal pipelines, and parameter pie charts in PNG, PDF, and SVG formats across light, dark, monochrome, and presentation themes.

### Fixed
* **Decoder Bug Fix:** Fixed an `AttributeError` inside `models/answer_decoder.py` by initializing the missing `self.embedding_dropout` layer.
* **Monochrome Style Correction:** Corrected monochrome theme parameters inside `styles.py` to prevent initialization errors.

---

## [2.0.0] - 2026-03-10

### Added
* **Multimodal Fusion Layer:** Added linear feature projection to align EfficientNet visual features with the Transformer decoder space.
* **Masked Answer Decoder:** Implemented a multi-head causal self-attention Transformer decoder for answer generation.
* **Multi-Dataset Support:** Added GQA and TextVQA readers to the dataset pipeline.

---

## [1.0.0] - 2025-11-05

### Added
* **Base Model Architecture:** Created the initial model skeleton with the EfficientNet-B0 visual backbone.
* **Dataset Downloader Tool:** Implemented the concurrent dataset downloader CLI with ThreadPool support.
* **Vocabulary Generator:** Added custom tokenizers and vocabulary configuration mappings.
