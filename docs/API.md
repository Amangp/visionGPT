# API Reference Manual

This document provides detailed API specifications for the public classes of VisionGPT v3, including constructor parameters, method arguments, return types, and code snippets.

---

## 1. Class: `VisionGPT`
* **Module Path:** `models.visiongpt`
* **Inherits From:** `tf.keras.Model`

The main model class coordinating visual feature extraction, fusion, and text decoding.

### Constructor Parameters:
* `vocab_size` (int, default=10000): Vocabulary size.
* `embed_dim` (int, default=256): Transformer hidden state dimension.
* `context_dropout_rate` (float, default=0.10): Text token dropout rate.
* `mask_token_id` (int, default=1): Mask token identifier.
* `start_token_id` (int, default=3): Start-of-sentence token identifier.
* `end_token_id` (int, default=4): End-of-sentence token identifier.

### Usage Example:
```python
from models.visiongpt import VisionGPT
import tensorflow as tf

model = VisionGPT(vocab_size=5000, embed_dim=256)
images = tf.random.normal((2, 224, 224, 3))
text = tf.constant([[3, 40, 50, 4]] * 2, dtype=tf.int64)

# Call signature: call(inputs, training=False)
output_logits = model((images, text), training=False)
print(output_logits.shape)  # Output: (2, 4, 5000)
```

---

## 2. Class: `Evaluator`
* **Module Path:** `evaluation.evaluator`

A unified metric evaluation class to evaluate text generation and question answering tasks.

### Methods:
#### `evaluate(predictions: List[Any], references: List[Any]) -> Dict[str, float]`
* **Arguments:**
  * `predictions`: Predictions (list of strings, bytes, numpy arrays, or eager TF Tensors).
  * `references`: References (list of strings, bytes, numpy arrays, or eager TF Tensors).
* **Returns:** A dictionary containing "BLEU-1" through "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "Exact Match", "Token Accuracy", and "Word Accuracy".
* **Raises:** `LengthMismatchError`, `EmptyInputError`.

### Usage Example:
```python
from evaluation.evaluator import Evaluator

evaluator = Evaluator()
preds = ["a brown cat sat on a table", "a blue car"]
refs = ["a cat sitting on the table", "a blue car"]

results = evaluator.evaluate(preds, refs)
print("BLEU-4:", results["BLEU-4"])
print("Exact Match:", results["Exact Match"])
```

---

## 3. Class: `BenchmarkSuite`
* **Module Path:** `benchmark.benchmark`

Performs hardware diagnostics, model statistics profiling, memory checks under load, and speed evaluations.

### Methods:
#### `benchmark_model_and_system(model: tf.keras.Model, batch_sizes: list[int] = [1, 2, 4, 8]) -> Dict[str, Any]`
* **Arguments:**
  * `model`: The compiled or unbuilt Keras model.
  * `batch_sizes`: Batch sizes to run for speed latency checks.
* **Returns:** A dictionary containing CPU, GPU, RAM parameters, layer sizes, and latency metrics.

#### `generate_report(output_dir: Optional[str] = None) -> None`
* Writes `benchmark.json`, `benchmark.csv`, and `benchmark.txt` reports and updates Matplotlib figures.

### Usage Example:
```python
from benchmark.benchmark import BenchmarkSuite

suite = BenchmarkSuite(output_dir="./reports")
suite.benchmark_model_and_system(model, batch_sizes=[1, 2])
suite.generate_report()
```

---

## 4. Class: `ExperimentLogger`
* **Module Path:** `monitoring.experiment`

Monitors experiment details, registers epochs, and updates reports/dashboards.

### Methods:
#### `log_epoch(metrics: EpochMetrics) -> None`
* **Arguments:**
  * `metrics`: An instance of `EpochMetrics` containing metric scores for the current epoch.

#### `save() -> None`
* Forces updates of JSON/CSV metrics logs, plots, reports, and the HTML dashboard page.

### Usage Example:
```python
from monitoring.experiment import ExperimentLogger
from monitoring.metrics import EpochMetrics

logger = ExperimentLogger(experiment_name="run_1", epochs=2)
logger.log_epoch(EpochMetrics(epoch=0, loss=0.45, learning_rate=1e-3))
logger.save()
```

---

## 5. Class: `VisionGPTTrainingCallback`
* **Module Path:** `monitoring.callbacks`
* **Inherits From:** `tf.keras.callbacks.Callback`

A training callback that plugs directly into Keras `.fit()` loops to automate logging.

### Constructor Parameters:
* `experiment_logger` (ExperimentLogger): The active ExperimentLogger instance.
* `tensorboard_logdir` (str, default='tensorboard'): TensorBoard log folder.
* `validation_data` (Optional): Optional validation dataset.
* `eval_func` (Optional): Optional function evaluating model validation scores.

### Usage Example:
```python
from monitoring.callbacks import VisionGPTTrainingCallback

callback = VisionGPTTrainingCallback(experiment_logger=logger)
model.fit(train_data, epochs=10, callbacks=[callback])
```

---

## 6. Class: `ArchitectureVisualizer`
* **Module Path:** `visualization.architecture`

Generates structural layout and dataflow diagrams for models.

### Methods:
#### `generate_architecture(theme: str = 'light') -> plt.Figure`
* Renders a vertical layout diagram under themes: `light`, `dark`, `monochrome`, or `presentation`.

#### `generate_summary() -> str`
* Compiles the Markdown table and plain-text model summary files.

### Usage Example:
```python
from visualization.architecture import ArchitectureVisualizer

viz = ArchitectureVisualizer(model=model, output_dir=".")
viz.generate_architecture(theme="monochrome")
viz.export_png("monochrome_architecture.png")
viz.generate_summary()
```
