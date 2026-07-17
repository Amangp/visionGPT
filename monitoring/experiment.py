"""Experiment logging and management interface for VisionGPT.

Implements `ExperimentLogger` to manage and serialize metadata,
track history progression, and trigger visualizations/reports.
"""

from dataclasses import asdict, dataclass
import datetime
import json
import logging
import os
import time
from typing import Any, Dict, Optional

from visiongpt.monitoring.utils import (
    ensure_output_directories,
    get_git_commit_hash,
    get_hardware_diagnostics,
)
from visiongpt.monitoring.history import TrainingHistory
from visiongpt.monitoring.metrics import EpochMetrics
from visiongpt.monitoring.plots import generate_all_plots
from visiongpt.monitoring.report import build_training_reports

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata detailing experiment parameters and environment specifications."""

    experiment_name: str
    timestamp: str
    dataset: str
    batch_size: int
    epochs: int
    optimizer: str
    learning_rate: float
    mixed_precision: bool
    checkpoint_path: str
    random_seed: int
    tensorflow_version: str
    cpu: str
    gpu: str
    ram: float
    git_commit_hash: str


class ExperimentLogger:
    """The central orchestrator for experiment monitoring and progress tracking.

    Manages training history, generates report text/plots, and exports data.
    """

    def __init__(
        self,
        experiment_name: str,
        base_dir: str = ".",
        dataset: str = "COCO 2017",
        batch_size: int = 32,
        epochs: int = 10,
        optimizer: str = "Adam",
        learning_rate: float = 1e-4,
        mixed_precision: bool = False,
        checkpoint_path: str = "checkpoints",
        random_seed: int = 42,
    ):
        """Initialize the ExperimentLogger.

        Args:
            experiment_name: A human-readable identifier for the training run.
            base_dir: The root path under which output folders are generated.
            dataset: Name of the dataset being used.
            batch_size: Size of input batch per step.
            epochs: Total scheduled training epochs.
            optimizer: Name/type of the optimizer used (e.g. Adam).
            learning_rate: Optimizer initial learning rate.
            mixed_precision: Boolean flag indicating if mixed precision is enabled.
            checkpoint_path: Destination folder for model checkpoints.
            random_seed: Random seed used in calculations.
        """
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.paths = ensure_output_directories(base_dir)

        # Retrieve system configuration details
        hw = get_hardware_diagnostics()
        git_hash = get_git_commit_hash()

        self.metadata = ExperimentMetadata(
            experiment_name=experiment_name,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=optimizer,
            learning_rate=learning_rate,
            mixed_precision=mixed_precision,
            checkpoint_path=os.path.abspath(checkpoint_path),
            random_seed=random_seed,
            tensorflow_version=hw.get("tensorflow_version", "Unknown"),
            cpu=hw.get("cpu", "Unknown"),
            gpu=hw.get("gpu_devices", ["N/A"])[0] if hw.get("gpu_devices") else "N/A",
            ram=hw.get("ram_gb", 0.0),
            git_commit_hash=git_hash,
        )

        self.history = TrainingHistory(history_dir=self.paths["history"])
        self.start_time = time.time()

    def log_epoch(self, metrics: EpochMetrics) -> None:
        """Record metrics for an epoch.

        Args:
            metrics: The EpochMetrics instance to log.
        """
        self.history.append(metrics.to_dict())

    def save(self) -> None:
        """Trigger updates for all reports, exports, and plots."""
        # 1. Save JSON & CSV history archives
        self.history.save()

        # 2. Export experiment configurations and statistics summaries
        self.export_json()

        # 3. Re-render visual metrics plots
        self.plot()

        # 4. Generate text reports
        self.generate_report()

    def load(self) -> None:
        """Load history records from disk to resume log appending."""
        self.history.load()

    def plot(self) -> None:
        """Render high-DPI progression plots."""
        generate_all_plots(self.history, self.paths["plots"])

    def export_json(self) -> None:
        """Export structured training metadata to the results directory."""
        # Write results/experiment.json
        meta_path = os.path.join(self.paths["results"], "experiment.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.metadata), f, indent=4)
        except Exception as e:
            logger.error("Failed to export experiment.json: %s", e)

        # Skip if no history records exist
        if not self.history.records:
            return

        # Write results/metrics.json (dictionary of overall best metrics values)
        metrics_path = os.path.join(self.paths["results"], "metrics.json")
        try:
            best_metrics = self._calculate_best_metrics()
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(best_metrics, f, indent=4)
        except Exception as e:
            logger.error("Failed to export metrics.json: %s", e)

        # Write results/summary.json (general run summary statistics)
        summary_path = os.path.join(self.paths["results"], "summary.json")
        try:
            total_duration = time.time() - self.start_time
            best_epoch_idx = best_metrics.get("best_validation_loss_epoch", 0)
            
            summary_data = {
                "experiment_name": self.metadata.experiment_name,
                "dataset": self.metadata.dataset,
                "total_epochs_trained": len(self.history.records),
                "best_epoch": best_epoch_idx,
                "lowest_val_loss": best_metrics.get("lowest_validation_loss"),
                "best_bleu_4": best_metrics.get("best_bleu_4"),
                "best_cider": best_metrics.get("best_cider"),
                "total_training_duration_seconds": round(total_duration, 2),
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=4)
        except Exception as e:
            logger.error("Failed to export summary.json: %s", e)

    def export_csv(self) -> None:
        """Force write of training history records to CSV."""
        self.history.save()

    def generate_report(self) -> None:
        """Compile and save training summary reports in TXT and MD formats."""
        if not self.history.records:
            return
        
        best_metrics = self._calculate_best_metrics()
        total_duration_sec = time.time() - self.start_time

        build_training_reports(
            records=self.history.records,
            metadata=asdict(self.metadata),
            best_metrics=best_metrics,
            duration_sec=total_duration_sec,
            report_dir=self.paths["reports"],
        )

        # Generate HTML Dashboard
        from visiongpt.monitoring.dashboard import generate_html_dashboard
        generate_html_dashboard(self, self.paths["reports"])

    def _calculate_best_metrics(self) -> Dict[str, Any]:
        """Compute the optimal validation and evaluation metrics from history records."""
        records = self.history.records
        best = {}

        # 1. Helper to find optimal values
        def find_best(key: str, mode: str = "max") -> tuple[Optional[float], int]:
            valid_runs = [r for r in records if key in r and r[key] is not None]
            if not valid_runs:
                return None, 0
            
            if mode == "min":
                best_run = min(valid_runs, key=lambda x: x[key])
            else:
                best_run = max(valid_runs, key=lambda x: x[key])
            return float(best_run[key]), int(best_run["epoch"])

        best["lowest_validation_loss"], best["best_validation_loss_epoch"] = find_best("validation_loss", "min")
        best["best_bleu_4"], best["best_bleu_4_epoch"] = find_best("bleu_4", "max")
        best["best_meteor"], best["best_meteor_epoch"] = find_best("meteor", "max")
        best["best_rouge_l"], best["best_rouge_l_epoch"] = find_best("rouge_l", "max")
        best["best_cider"], best["best_cider_epoch"] = find_best("cider", "max")
        best["best_exact_match"], best["best_exact_match_epoch"] = find_best("exact_match", "max")

        # Average epoch duration
        durations = [r["time_per_epoch"] for r in records if "time_per_epoch" in r and r["time_per_epoch"] is not None]
        best["average_epoch_time_sec"] = sum(durations) / len(durations) if durations else 0.0

        return best
