# Frequently Asked Questions (FAQ)

This document addresses common issues and troubleshooting tips for VisionGPT v3.

---

## 1. Installation Issues

### Q: I get `ModuleNotFoundError: No module named 'tensorflow'` even after installing it.
* **A:** This happens if Python is running in a different environment than the one where you ran the installation. Check your active conda environment:
  ```bash
  conda info --envs
  ```
  Ensure the active environment has `*` next to it. Activate it using:
  ```bash
  conda activate visiongpt
  ```

---

## 2. CUDA & GPU Issues

### Q: TensorFlow runs on CPU instead of GPU.
* **A:** This occurs when CUDA libraries cannot be resolved by the system loader. Verify that TensorFlow sees the GPU:
  ```python
  import tensorflow as tf
  print("Visible Devices:", tf.config.list_physical_devices('GPU'))
  ```
  If this returns an empty list `[]`, check:
  1. That you have an NVIDIA driver installed. Run `nvidia-smi` to verify.
  2. That your CUDA Toolkit and cuDNN versions match your TensorFlow version. Check our [Installation Guide](INSTALLATION.md) for compatibility details.
  3. That the CUDA path is added to your environment `PATH`.

---

## 3. Dataset Download Issues

### Q: The Dataset Manager gets stuck or reports a network timeout.
* **A:** Large dataset downloads can experience timeouts depending on your connection. The downloader supports resumption:
  1. Interrupt the downloader: `Ctrl + C`.
  2. Re-run the command (e.g. `python tools/dataset_manager/main.py --gqa`).
  3. The downloader will read the existing bytes and resume downloading via HTTP `Range` headers.

---

## 4. Training & Out-of-Memory (OOM) Issues

### Q: I get `ResourceExhaustedError: OOM when allocating tensor...` during training.
* **A:** This means the model batch size is too large for your GPU memory.
  1. Reduce your batch size (e.g. from 32 to 16, or 8, or 4).
  2. Enable mixed-precision training:
     ```python
     tf.keras.mixed_precision.set_global_policy('mixed_float16')
     ```
  3. Decrease the input sequence length or use gradient accumulation steps.

### Q: Validation metrics are not logged or display as `N/A` in reports.
* **A:** Ensure the validation logs passed to the callback contain the expected keys. The logger supports common key variations (e.g. `val_loss`, `val_bleu_4`, `val_cider`). If you run custom evaluations, pass your validation dataset and evaluation function directly to the callback:
  ```python
  callback = VisionGPTTrainingCallback(
      experiment_logger=logger,
      validation_data=val_dataset,
      eval_func=my_custom_eval_fn
  )
  ```

---

## 5. Checkpoint & History Resumption

### Q: How do I reload my experiment history after resuming training?
* **A:** Instantiate the `ExperimentLogger` using the same `experiment_name`. Call `load()` to read the saved logs from disk, and then call `prune_for_resume(initial_epoch)` to clear any duplicate history records:
  ```python
  logger = ExperimentLogger(experiment_name="my_run")
  logger.load()
  logger.history.prune_for_resume(initial_epoch=6)
  ```
