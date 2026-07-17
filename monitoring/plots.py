"""Plot generation utilities for VisionGPT training monitoring.

Generates high-DPI matplotlib plots for loss, learning rate, evaluation scores,
memory usage, and training speeds over training epochs.
"""

import logging
import os
from typing import List
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for server/command-line compatibility
import matplotlib.pyplot as plt

from visiongpt.monitoring.history import TrainingHistory

logger = logging.getLogger(__name__)


def set_plot_style() -> None:
    """Set professional, clean matplotlib styles without seaborn dependencies."""
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#f8f9fa"
    plt.rcParams["axes.edgecolor"] = "#ccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "#e5e5e5"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["font.size"] = 10
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"


def generate_all_plots(history: TrainingHistory, plot_dir: str) -> None:
    """Generate all requested metrics charts and save them in high DPI.

    Args:
        history: The TrainingHistory manager instance.
        plot_dir: Target folder to save the generated PNG files.
    """
    if not history.records:
        logger.debug("No records available in history. Skipping plot rendering.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    set_plot_style()

    epochs = [r["epoch"] for r in history.records]

    def plot_single(
        metric_key: str, title: str, ylabel: str, color: str, filename: str
    ) -> None:
        values = [r[metric_key] for r in history.records if metric_key in r and r[metric_key] is not None]
        if not values or len(values) != len(epochs):
            # Try matching by checking entries dynamically
            matching_epochs = [
                r["epoch"] for r in history.records if metric_key in r and r[metric_key] is not None
            ]
            if not matching_epochs:
                logger.debug("Metric '%s' not present in history; skipping %s", metric_key, filename)
                return
            x_vals = matching_epochs
            y_vals = [r[metric_key] for r in history.records if metric_key in r and r[metric_key] is not None]
        else:
            x_vals = epochs
            y_vals = values

        plt.figure(figsize=(8, 5))
        plt.plot(x_vals, y_vals, color=color, linewidth=2, marker="o", markersize=4)
        plt.title(title, pad=15, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.tight_layout()
        path = os.path.join(plot_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.debug("Saved plot: %s", path)

    # 1. Loss & Val Loss
    plot_single("loss", "Training Loss Progression", "Loss", "#d62728", "loss.png")
    plot_single("validation_loss", "Validation Loss Progression", "Loss", "#1f77b4", "val_loss.png")

    # 2. Learning Rate
    plot_single("learning_rate", "Learning Rate Schedule", "Learning Rate", "#ff7f0e", "learning_rate.png")

    # 3. BLEU Scores (combines BLEU-1 to BLEU-4 in one professional plot)
    try:
        b1 = [r.get("bleu_1") for r in history.records]
        b2 = [r.get("bleu_2") for r in history.records]
        b3 = [r.get("bleu_3") for r in history.records]
        b4 = [r.get("bleu_4") for r in history.records]

        if any(b is not None for b in b1 + b2 + b3 + b4):
            plt.figure(figsize=(8, 5))
            if any(x is not None for x in b1):
                plt.plot(epochs, b1, label="BLEU-1", color="#1f77b4", marker="o", markersize=3)
            if any(x is not None for x in b2):
                plt.plot(epochs, b2, label="BLEU-2", color="#aec7e8", marker="x", markersize=3)
            if any(x is not None for x in b3):
                plt.plot(epochs, b3, label="BLEU-3", color="#ff7f0e", marker="s", markersize=3)
            if any(x is not None for x in b4):
                plt.plot(epochs, b4, label="BLEU-4", color="#ffbb78", marker="^", markersize=3)
            plt.title("BLEU Metric Progressions", pad=15, fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.legend()
            plt.tight_layout()
            path = os.path.join(plot_dir, "bleu.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        logger.error("Failed to render bleu.png: %s", e)

    # 4. METEOR, ROUGE-L, CIDEr
    plot_single("meteor", "METEOR Evaluation Score", "METEOR", "#2ca02c", "meteor.png")
    plot_single("rouge_l", "ROUGE-L F-Measure", "ROUGE-L", "#9467bd", "rouge.png")
    plot_single("cider", "CIDEr Consensus Score", "CIDEr", "#8c564b", "cider.png")

    # 5. VQA Accuracy (combines EM and Token Accuracy)
    try:
        em = [r.get("exact_match") for r in history.records]
        ta = [r.get("token_accuracy") for r in history.records]

        if any(x is not None for x in em + ta):
            plt.figure(figsize=(8, 5))
            if any(x is not None for x in em):
                plt.plot(epochs, em, label="Exact Match", color="#2ca02c", marker="o", markersize=3)
            if any(x is not None for x in ta):
                plt.plot(epochs, ta, label="Token Accuracy", color="#bcbd22", marker="s", markersize=3)
            plt.title("VQA Accuracy Progression", pad=15, fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            path = os.path.join(plot_dir, "accuracy.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        logger.error("Failed to render accuracy.png: %s", e)

    # 6. Memory Allocation (RAM vs VRAM)
    try:
        ram = [r.get("cpu_memory") for r in history.records]
        vram = [r.get("gpu_memory") for r in history.records]

        if any(x is not None for x in ram + vram):
            plt.figure(figsize=(8, 5))
            if any(x is not None for x in ram):
                plt.plot(epochs, ram, label="Peak CPU RAM", color="#17becf", marker="o", markersize=3)
            if any(x is not None for x in vram):
                plt.plot(epochs, vram, label="Peak GPU VRAM", color="#98df8a", marker="s", markersize=3)
            plt.title("System Resource Utilization History", pad=15, fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Memory (MB)")
            plt.legend()
            plt.tight_layout()
            path = os.path.join(plot_dir, "memory.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as e:
        logger.error("Failed to render memory.png: %s", e)

    # 7. Time per Epoch
    plot_single("time_per_epoch", "Epoch Duration", "Duration (seconds)", "#7f7f7f", "epoch_time.png")
