"""Training history registry and management.

Maintains an in-memory list of epoch metrics, exports records to CSV/JSON,
and handles training resume checks by pruning overlapping epochs.
"""

import csv
import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TrainingHistory:
    """Manages epoch-level metrics histories for training runs.

    Supports reading, writing, and formatting metrics, as well as pruning
    redundant epochs when resuming training from checkpoints.
    """

    def __init__(self, history_dir: str = "history"):
        """Initialize the training history monitor.

        Args:
            history_dir: Target directory for history file storage.
        """
        self.history_dir = history_dir
        self.json_path = os.path.join(history_dir, "training_history.json")
        self.csv_path = os.path.join(history_dir, "training_history.csv")
        self.records: List[Dict[str, Any]] = []

    def append(self, epoch_metrics: Dict[str, Any]) -> None:
        """Append a new epoch's metrics record to the history registry.

        Args:
            epoch_metrics: A dictionary containing epoch measurements.
        """
        # Ensure 'epoch' key exists
        if "epoch" not in epoch_metrics:
            raise KeyError("The metrics dictionary must contain the 'epoch' identifier.")
        
        # Remove any existing record for this epoch to prevent duplicates
        self.records = [r for r in self.records if r["epoch"] != epoch_metrics["epoch"]]
        self.records.append(epoch_metrics)
        # Keep records sorted by epoch index
        self.records.sort(key=lambda x: x["epoch"])

    def prune_for_resume(self, initial_epoch: int) -> None:
        """Prune all metrics logs for epoch >= initial_epoch.

        Enables seamless checkpoint resumption by eliminating future overlaps
        from prior interrupted training runs.

        Args:
            initial_epoch: The starting epoch index of the resumed training run.
        """
        logger.info(
            "Pruning training history records for epoch >= %d to support resume.",
            initial_epoch,
        )
        before_count = len(self.records)
        self.records = [r for r in self.records if r["epoch"] < initial_epoch]
        logger.info(
            "Pruned %d records from history. Current record count: %d",
            before_count - len(self.records),
            len(self.records),
        )

    def load(self) -> None:
        """Load history records from the JSON archive.

        Falls back to an empty record list if the archive does not exist.
        """
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
                # Ensure they are sorted
                self.records.sort(key=lambda x: x.get("epoch", 0))
                logger.info(
                    "Successfully loaded %d history records from %s",
                    len(self.records),
                    self.json_path,
                )
            except Exception as e:
                logger.error("Failed to load training history JSON: %s. Starting fresh.", e)
                self.records = []
        else:
            logger.debug("No pre-existing history JSON found at %s. Starting fresh.", self.json_path)
            self.records = []

    def save(self) -> None:
        """Serialize current history records to both JSON and CSV files."""
        if not self.records:
            logger.debug("No history records to write.")
            return

        os.makedirs(self.history_dir, exist_ok=True)

        # 1. Save JSON
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=4)
            logger.debug("Saved training history JSON to %s", self.json_path)
        except Exception as e:
            logger.error("Failed to write training history JSON: %s", e)

        # 2. Save CSV
        try:
            # Dynamically extract all headers from all records to accommodate new metrics
            headers_set = set()
            for r in self.records:
                headers_set.update(r.keys())
            
            # Sort headers putting 'epoch' first for readability
            headers = ["epoch"] + sorted(list(headers_set - {"epoch"}))

            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for r in self.records:
                    # Fill missing fields with empty values
                    row = {h: r.get(h, "") for h in headers}
                    writer.writerow(row)
            logger.debug("Saved training history CSV to %s", self.csv_path)
        except Exception as e:
            logger.error("Failed to write training history CSV: %s", e)

    def get_metrics_list(self, metric_name: str) -> List[Any]:
        """Extract a list of values for a specific metric over all tracked epochs.

        Args:
            metric_name: The name of the metric (e.g. 'loss').

        Returns:
            A list of values corresponding to the metric, ordered by epoch.
        """
        return [r[metric_name] for r in self.records if metric_name in r]
