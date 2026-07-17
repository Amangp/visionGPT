"""Keras training callback implementation for VisionGPT.

Implements `VisionGPTTrainingCallback` which integrates with TensorFlow fit loops,
performs resource audits, updates metrics logs/plots, and writes TensorBoard summaries.
"""

import logging
import time
from typing import Any, Dict, Optional
import tensorflow as tf

from visiongpt.monitoring.utils import get_hardware_diagnostics, get_system_ram_gb
from visiongpt.monitoring.metrics import EpochMetrics

logger = logging.getLogger(__name__)

# Try to import VRAM query from benchmark if available, otherwise fallback
try:
    from benchmark.memory import get_tf_gpu_memory_info
except ImportError:
    def get_tf_gpu_memory_info(device: str = "GPU:0") -> Dict[str, float]:
        try:
            info = tf.config.experimental.get_memory_info(device)
            return {
                "current": info["current"] / (1024 * 1024),
                "peak": info["peak"] / (1024 * 1024),
            }
        except Exception:
            return {"current": 0.0, "peak": 0.0}

# Try to import current process RAM from benchmark or fallback
try:
    from benchmark.memory import get_current_ram
except ImportError:
    import os
    import platform
    def get_current_ram() -> float:
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0


class VisionGPTTrainingCallback(tf.keras.callbacks.Callback):
    """Keras Callback to monitor, log, and visualize training progress in real-time.

    Logs training loss, validation metrics, learning rates, RAM/VRAM resource profiles,
    and updates TensorBoard summaries, CSV/JSON logs, plots, and markdown reports.
    """

    def __init__(
        self,
        experiment_logger: Any,
        tensorboard_logdir: str = "tensorboard",
        validation_data: Optional[Any] = None,
        eval_func: Optional[Any] = None,
    ):
        """Initialize the callback.

        Args:
            experiment_logger: Reference to the ExperimentLogger instance.
            tensorboard_logdir: Folder path for TensorBoard event files.
            validation_data: Optional validation dataset/generator.
            eval_func: Optional evaluation function taking (model, validation_data)
                and returning a dict of evaluation scores.
        """
        super().__init__()
        self.experiment_logger = experiment_logger
        self.tensorboard_logdir = tensorboard_logdir
        self.validation_data = validation_data
        self.eval_func = eval_func
        self.writer: Optional[tf.summary.SummaryWriter] = None
        self.epoch_start_time = 0.0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the TensorBoard SummaryWriter and prune history if resuming.

        Args:
            logs: Callback logs dictionary.
        """
        logger.info("Initializing training monitor callback.")
        try:
            self.writer = tf.summary.create_file_writer(self.tensorboard_logdir)
        except Exception as e:
            logger.error("Failed to create TensorBoard SummaryWriter: %s", e)

        # Check if we are resuming from an initial epoch > 0
        initial_epoch = 0
        if self.model and hasattr(self.model, "history") and self.model.history:
            # Try to get initial epoch from keras fit context if available
            if hasattr(self.model.history, "epoch") and self.model.history.epoch:
                initial_epoch = self.model.history.epoch[-1] + 1
        
        # If the optimizer has iterations, we can estimate starting epoch
        # (Though prune_for_resume will be called manually by user via load(),
        # doing it here provides an extra safety net)
        if initial_epoch > 0:
            self.experiment_logger.history.prune_for_resume(initial_epoch)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Start the epoch timer.

        Args:
            epoch: Index of the current epoch.
            logs: Callback logs dictionary.
        """
        self.epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Profile learning rate, resource usage, and update all outputs.

        Args:
            epoch: Index of the epoch.
            logs: Metrics dictionary from Keras.
        """
        epoch_time = time.perf_counter() - self.epoch_start_time
        logs = logs or {}

        # 1. Fetch current Learning Rate
        lr = None
        if self.model and hasattr(self.model, "optimizer") and self.model.optimizer is not None:
            optimizer = self.model.optimizer
            if hasattr(optimizer, "learning_rate"):
                lr_var = optimizer.learning_rate
                if hasattr(lr_var, "numpy") or isinstance(lr_var, tf.Variable):
                    lr = float(lr_var.numpy())
                elif callable(lr_var):
                    # Decay schedule
                    try:
                        step = int(optimizer.iterations.numpy())
                        lr = float(lr_var(step).numpy())
                    except Exception:
                        pass
                else:
                    try:
                        lr = float(lr_var)
                    except Exception:
                        pass

        # 2. Run custom evaluation if provided
        if self.eval_func and self.validation_data is not None:
            try:
                logger.info("Running custom validation evaluation at epoch %d...", epoch)
                eval_metrics = self.eval_func(self.model, self.validation_data)
                logs.update(eval_metrics)
            except Exception as e:
                logger.error("Failed custom evaluation at epoch end: %s", e)

        # 3. Retrieve memory consumption
        cpu_mem = get_current_ram()
        gpu_mem = get_tf_gpu_memory_info("GPU:0")["peak"]

        # 4. Construct metrics mappings (support diverse key variations)
        def get_log_metric(*keys: str) -> Optional[float]:
            for k in keys:
                for log_k in [k, k.lower(), k.upper(), f"val_{k}", f"val_{k.lower()}"]:
                    if log_k in logs:
                        try:
                            return float(logs[log_k])
                        except (ValueError, TypeError):
                            pass
            return None

        # Build EpochMetrics instance
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            loss=float(logs.get("loss", 0.0)),
            validation_loss=get_log_metric("validation_loss", "val_loss", "valLoss"),
            learning_rate=lr if lr is not None else get_log_metric("learning_rate", "lr"),
            bleu_1=get_log_metric("bleu_1", "bleu1", "BLEU-1"),
            bleu_2=get_log_metric("bleu_2", "bleu2", "BLEU-2"),
            bleu_3=get_log_metric("bleu_3", "bleu3", "BLEU-3"),
            bleu_4=get_log_metric("bleu_4", "bleu4", "bleu", "val_bleu", "BLEU-4"),
            meteor=get_log_metric("meteor", "METEOR"),
            rouge_l=get_log_metric("rouge_l", "rouge", "rougeL", "ROUGE-L"),
            cider=get_log_metric("cider", "CIDEr"),
            exact_match=get_log_metric("exact_match", "exactMatch", "em", "Exact Match"),
            token_accuracy=get_log_metric("token_accuracy", "tokenAccuracy", "token_acc", "Token Accuracy"),
            time_per_epoch=epoch_time,
            gpu_memory=gpu_mem if gpu_mem > 0 else None,
            cpu_memory=cpu_mem if cpu_mem > 0 else None,
        )

        # 5. Log inside ExperimentLogger
        try:
            self.experiment_logger.log_epoch(epoch_metrics)
        except Exception as e:
            logger.error("Failed to append metrics inside ExperimentLogger: %s", e)

        # 6. TensorBoard Logging
        if self.writer is not None:
            try:
                metrics_dict = epoch_metrics.to_dict()
                with self.writer.as_default():
                    # Log metrics scalars
                    for name, value in metrics_dict.items():
                        if name != "epoch":
                            tf.summary.scalar(f"epoch/{name}", value, step=epoch)

                    # Log weights histograms for model parameter profiling
                    if self.model and hasattr(self.model, "trainable_variables"):
                        for var in self.model.trainable_variables:
                            tf.summary.histogram(f"weights/{var.name}", var, step=epoch)
                logger.debug("Successfully pushed epoch %d metrics to TensorBoard.", epoch)
            except Exception as e:
                logger.error("TensorBoard metric logging failed: %s", e)

        # 7. Serialize reports and trigger plots updates
        try:
            self.experiment_logger.save()
        except Exception as e:
            logger.error("Failed to trigger automated serialization and plots update: %s", e)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Close handlers on training completion.

        Args:
            logs: Callback logs dictionary.
        """
        logger.info("Training runs completed. Finalizing logs.")
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as e:
                logger.error("Failed to close TensorBoard SummaryWriter: %s", e)
