# Training Pipeline & Configuration

This document outlines the training pipeline of VisionGPT v3. It covers dataset tokenization, mixed precision configuration, custom monitoring callbacks, checkpointing, and model fine-tuning or checkpoint resumption.

---

## 1. Training Pipeline Overview

VisionGPT v3 employs a step-by-step training flow:
1. **Data Prep:** Raw images are resized to `(224, 224, 3)` and normalized. Text answers are tokenized, padded, and mapped to token IDs.
2. **Vision Encoding:** Images pass through the EfficientNet-B0 backbone (frozen by default to retain visual priors) to produce a `(7, 7, 1280)` feature grid.
3. **Feature Projection:** Spatial grid features are projected through the linear `FusionLayer` to map to the `embed_dim` dimension of the Transformer, yielding a `(49, 256)` token sequence.
4. **Decoder Cross-Attention:** Text tokens pass through an embedding and causal self-attention, and attend to the fused image tokens via cross-attention.
5. **Loss Maximization:** Categorical cross-entropy loss is computed over the output tokens and backpropagated through the Answer Decoder and Fusion Layer.

---

## 2. Mixed Precision Acceleration

To speed up training on modern NVIDIA GPUs (Tensor Core architectures like Turing, Ampere, and Ada Lovelace), configure mixed-precision training:

```python
import tensorflow as tf

# Set global policy to mixed_float16
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

print("Compute dtype:", policy.compute_dtype)
print("Variable dtype:", policy.variable_dtype)
```

By enabling mixed precision:
* Weights and activations are represented in 16-bit floats (`float16`) to accelerate matrix calculations.
* Master weights are maintained in 32-bit floats (`float32`) to retain numeric precision.
* TensorFlow automatically applies **loss scaling** to prevent underflow.

---

## 3. Training Callback Registry

We provide a dedicated callback module in `monitoring/callbacks.py` to monitor metrics and resource utilization during training.

### Setting Up the Callback:
```python
from monitoring.experiment import ExperimentLogger
from monitoring.callbacks import VisionGPTTrainingCallback

# 1. Initialize the experiment metadata manager
logger = ExperimentLogger(
    experiment_name="efficientnet_transformer_coco",
    dataset="COCO 2017",
    batch_size=32,
    epochs=10,
    optimizer="Adam",
    learning_rate=1e-4,
    mixed_precision=True
)

# 2. Instantiate the callback
callback = VisionGPTTrainingCallback(
    experiment_logger=logger,
    tensorboard_logdir="./tensorboard/run_1"
)
```

The callback automatically:
* Measures step duration and overall epoch execution times.
* Monitors host system RAM RSS memory usage.
* Monitors physical GPU VRAM peak allocation.
* Records weights and gradients histograms.
* Exports metric tables (`history/training_history.csv` / `.json`).
* Generates DPI-150 progression plots inside `plots/`.
* Compiles Markdown summaries in `reports/` and updates a responsive HTML dashboard.

---

## 4. Checkpointing & Resume Training

### A. Saving Checkpoints
We configure standard Keras `ModelCheckpoint` callbacks to save model weights:

```python
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/visiongpt_epoch_{epoch:02d}.keras",
    save_weights_only=False,
    monitor="val_loss",
    save_best_only=True
)
```

### B. Resuming Training
When resuming training from a saved checkpoint:
1. Reload the model weights using Keras.
2. Initialize `ExperimentLogger` and call `load()` to read prior training history records.
3. Call `prune_for_resume(initial_epoch)` on the history manager to overwrite any metrics recorded after the resumed epoch. This prevents duplication.

```python
# Reload model
model = tf.keras.models.load_model("checkpoints/visiongpt_epoch_05.keras")

# Initialize logger and load prior logs
logger = ExperimentLogger(experiment_name="efficientnet_transformer_coco")
logger.load()

# Prune history starting at epoch 6 (index 6)
logger.history.prune_for_resume(initial_epoch=6)

# Continue training passing the callback
model.fit(
    train_dataset,
    epochs=10,
    initial_epoch=6,
    callbacks=[callback, checkpoint_callback]
)
```

---

## 5. TensorBoard Monitoring

To monitor metrics live during training:
1. Open a terminal and run the TensorBoard server pointing to your logdir:
   ```bash
   tensorboard --logdir=tensorboard/
   ```
2. Navigate to `http://localhost:6006` in your browser.
3. Access:
   * **Scalars Tab:** View loss, validation loss, learning rate, and validation evaluation metrics over epochs.
   * **Histograms Tab:** Monitor weight distribution changes and gradient values of model layers over epochs.
   * **Graphs Tab:** Visualize the model's computational graph and connections.
