"""Verification tests and simulation script for the VisionGPT monitoring package.

Instantiates a simple model and training loop to execute Keras training callbacks
and verify the generation of all structured JSON/CSV logs, TensorBoard events,
plots, reports, and the HTML dashboard.
"""

import os
import shutil
import unittest
import numpy as np
import tensorflow as tf

from visiongpt.monitoring.experiment import ExperimentLogger
from visiongpt.monitoring.callbacks import VisionGPTTrainingCallback
from visiongpt.monitoring.metrics import EpochMetrics


class TestMonitoringPackage(unittest.TestCase):
    """Test suite to verify VisionGPT Experiment Logger and Callbacks."""

    @classmethod
    def setUpClass(cls):
        """Define directories and construct a mock Keras model for tests."""
        cls.base_dir = "./test_monitoring_runs"
        os.makedirs(cls.base_dir, exist_ok=True)

        # Simple dummy model for training callbacks test
        inputs = tf.keras.Input(shape=(10,))
        outputs = tf.keras.layers.Dense(1)(inputs)
        cls.mock_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        cls.mock_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"]
        )

        # Generate simple dummy inputs for callback execution
        cls.x_train = np.random.randn(32, 10)
        cls.y_train = np.random.randn(32, 1)
        cls.x_val = np.random.randn(10, 10)
        cls.y_val = np.random.randn(10, 1)

    @classmethod
    def tearDownClass(cls):
        """Clean up all folders created during the verification tests."""
        # Remove output folders generated during test calls
        folders_to_clean = ["results", "plots", "logs", "tensorboard", "reports", "history", cls.base_dir]
        for f in folders_to_clean:
            if os.path.exists(f):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)

    def test_logger_metadata_generation(self):
        """Test ExperimentLogger initialization and metadata checks."""
        exp_logger = ExperimentLogger(
            experiment_name="test_logger_run",
            base_dir=self.base_dir,
            epochs=5,
            batch_size=8,
        )

        self.assertEqual(exp_logger.metadata.experiment_name, "test_logger_run")
        self.assertEqual(exp_logger.metadata.epochs, 5)
        self.assertEqual(exp_logger.metadata.batch_size, 8)
        self.assertIn("cpu", asdict_keys := exp_logger.metadata.__dict__.keys())
        self.assertIn("gpu", asdict_keys)
        self.assertNotEqual(exp_logger.metadata.git_commit_hash, "")

    def test_manual_logging_cycle(self):
        """Test manually appending EpochMetrics and saving reports/plots."""
        exp_logger = ExperimentLogger(
            experiment_name="manual_test_run",
            base_dir=self.base_dir,
        )

        # Simulate 2 epochs of manual logging
        m1 = EpochMetrics(
            epoch=0,
            loss=0.45,
            validation_loss=0.40,
            learning_rate=1e-3,
            bleu_4=0.25,
            cider=1.2,
            cpu_memory=250.0,
            gpu_memory=0.0
        )
        m2 = EpochMetrics(
            epoch=1,
            loss=0.35,
            validation_loss=0.32,
            learning_rate=1e-3,
            bleu_4=0.35,
            cider=1.8,
            cpu_memory=255.0,
            gpu_memory=0.0
        )

        exp_logger.log_epoch(m1)
        exp_logger.log_epoch(m2)

        # Trigger reports and export updates
        exp_logger.save()

        # Check file outputs in base_dir subfolders
        self.assertTrue(os.path.exists(os.path.join(self.base_dir, "history", "training_history.json")))
        self.assertTrue(os.path.exists(os.path.join(self.base_dir, "history", "training_history.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.base_dir, "results", "experiment.json")))
        self.assertTrue(os.path.exists(os.path.join(self.base_dir, "results", "metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(self.base_dir, "results", "summary.json")))
        self.assertTrue(os.path.exists(os.path.join(self.base_dir, "reports", "training_report.md")))
        self.assertTrue(os.path.exists(os.path.join(self.base_dir, "reports", "training_report.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.base_dir, "reports", "dashboard.html")))

        # Check plot outputs
        self.assertTrue(os.path.exists(os.path.join(self.base_dir, "plots", "loss.png")))
        self.assertTrue(os.path.exists(os.path.join(self.base_dir, "plots", "bleu.png")))
        self.assertTrue(os.path.exists(os.path.join(self.base_dir, "plots", "cider.png")))

    def test_keras_callback_integration(self):
        """Test Keras Callback execution during a mock model fit cycle."""
        # Use default workspace level directories for this callback test to trace output structure
        shutil.rmtree("results", ignore_errors=True)
        shutil.rmtree("plots", ignore_errors=True)
        shutil.rmtree("history", ignore_errors=True)
        shutil.rmtree("reports", ignore_errors=True)

        exp_logger = ExperimentLogger(
            experiment_name="callback_fit_run",
            base_dir=".",  # Write directly to default workspace path
            epochs=2,
            batch_size=16
        )

        callback = VisionGPTTrainingCallback(
            experiment_logger=exp_logger,
            tensorboard_logdir="tensorboard",
        )

        # Run training loop for 2 epochs
        self.mock_model.fit(
            self.x_train,
            self.y_train,
            epochs=2,
            batch_size=16,
            validation_data=(self.x_val, self.y_val),
            callbacks=[callback],
            verbose=0
        )

        # Verify structured outputs at workspace root
        self.assertTrue(os.path.exists("history/training_history.json"))
        self.assertTrue(os.path.exists("history/training_history.csv"))
        self.assertTrue(os.path.exists("results/experiment.json"))
        self.assertTrue(os.path.exists("reports/training_report.md"))
        self.assertTrue(os.path.exists("reports/dashboard.html"))
        self.assertTrue(os.path.exists("plots/loss.png"))
        self.assertTrue(os.path.exists("plots/val_loss.png"))


if __name__ == "__main__":
    # Import time for timestamp parsing inside dashboard duration helper
    import time
    unittest.main()
