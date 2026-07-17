"""Experiment report generation utilities for VisionGPT.

Compiles and saves training progress summaries in Markdown (`training_report.md`)
and plain text (`training_report.txt`) formats.
"""

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def build_training_reports(
    records: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    best_metrics: Dict[str, Any],
    duration_sec: float,
    report_dir: str,
) -> None:
    """Compile and export training reports in TXT and MD formats.

    Args:
        records: Training history records list.
        metadata: Experiment configuration parameters.
        best_metrics: Dictionary of calculated peak scores.
        duration_sec: Total duration of the training run.
        report_dir: Directory where report files are stored.
    """
    os.makedirs(report_dir, exist_ok=True)
    md_path = os.path.join(report_dir, "training_report.md")
    txt_path = os.path.join(report_dir, "training_report.txt")

    # Format elapsed duration
    hours = int(duration_sec // 3600)
    minutes = int((duration_sec % 3600) // 60)
    seconds = int(duration_sec % 60)
    duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # 1. Compile Markdown report
    try:
        md_lines = []
        md_lines.append(f"# VisionGPT Training Experiment Report: {metadata.get('experiment_name')}")
        md_lines.append("")
        md_lines.append(f"Generated on `{metadata.get('timestamp')}`  ")
        md_lines.append(f"Cumulative Run Duration: `{duration_str}`  ")
        md_lines.append(f"Git Commit Version: `{metadata.get('git_commit_hash', 'N/A')}`")
        md_lines.append("")

        md_lines.append("## 1. Optimal Results Summary")
        md_lines.append("| Metric | Best Value | Achieved Epoch |")
        md_lines.append("| :--- | :---: | :---: |")
        
        # Populate optimal table rows
        metrics_mapping = [
            ("lowest_validation_loss", "best_validation_loss_epoch", "Validation Loss"),
            ("best_bleu_4", "best_bleu_4_epoch", "BLEU-4 Score"),
            ("best_cider", "best_cider_epoch", "CIDEr Score"),
            ("best_meteor", "best_meteor_epoch", "METEOR Score"),
            ("best_rouge_l", "best_rouge_l_epoch", "ROUGE-L Score"),
            ("best_exact_match", "best_exact_match_epoch", "Exact Match Accuracy"),
        ]

        for val_k, epoch_k, label in metrics_mapping:
            val = best_metrics.get(val_k)
            epoch = best_metrics.get(epoch_k, 0)
            val_str = f"{val:.4f}" if val is not None else "N/A"
            epoch_str = str(epoch) if val is not None else "N/A"
            md_lines.append(f"| {label} | {val_str} | {epoch_str} |")
        
        avg_time = best_metrics.get("average_epoch_time_sec", 0.0)
        md_lines.append("")
        md_lines.append(f"**Average Duration per Epoch:** `{avg_time:.2f} seconds`  ")
        md_lines.append(f"**Total Epochs Logged:** `{len(records)}`  ")
        md_lines.append("")

        md_lines.append("## 2. Experiment Configurations")
        md_lines.append(f"- **Dataset name:** {metadata.get('dataset')}")
        md_lines.append(f"- **Scheduled Epochs:** {metadata.get('epochs')}")
        md_lines.append(f"- **Batch Size:** {metadata.get('batch_size')}")
        md_lines.append(f"- **Optimizer:** {metadata.get('optimizer')}")
        md_lines.append(f"- **Initial Learning Rate:** {metadata.get('learning_rate')}")
        md_lines.append(f"- **Mixed Precision Active:** {metadata.get('mixed_precision')}")
        md_lines.append(f"- **Random Seed Value:** {metadata.get('random_seed')}")
        md_lines.append(f"- **Model Weights Checkpoint Path:** `{metadata.get('checkpoint_path')}`")
        md_lines.append("")

        md_lines.append("## 3. Hardware Diagnostics")
        md_lines.append(f"- **Host Operating System:** {metadata.get('os')}")
        md_lines.append(f"- **Processor CPU Model:** {metadata.get('cpu')}")
        md_lines.append(f"- **GPU device:** {metadata.get('gpu')}")
        md_lines.append(f"- **Installed Memory:** {metadata.get('ram')} GB")
        md_lines.append(f"- **TensorFlow framework version:** v{metadata.get('tensorflow_version')}")
        md_lines.append("")

        md_lines.append("## 4. Epoch Summary History Logs")
        md_lines.append("| Epoch | Loss | Val Loss | Learning Rate | BLEU-4 | CIDEr | RAM peak | VRAM peak |")
        md_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in records:
            epoch = r.get("epoch", 0)
            loss = r.get("loss", 0.0)
            val_loss = r.get("validation_loss")
            lr = r.get("learning_rate")
            bleu = r.get("bleu_4")
            cider = r.get("cider")
            cpu_m = r.get("cpu_memory")
            gpu_m = r.get("gpu_memory")

            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            lr_str = f"{lr:.2e}" if lr is not None else "N/A"
            bleu_str = f"{bleu:.4f}" if bleu is not None else "N/A"
            cider_str = f"{cider:.4f}" if cider is not None else "N/A"
            cpu_str = f"{cpu_m:.1f}MB" if cpu_m is not None else "N/A"
            gpu_str = f"{gpu_m:.1f}MB" if gpu_m is not None else "N/A"

            md_lines.append(
                f"| {epoch} | {loss:.4f} | {val_loss_str} | {lr_str} | {bleu_str} | {cider_str} | {cpu_str} | {gpu_str} |"
            )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        logger.debug("Saved markdown report to %s", md_path)
    except Exception as e:
        logger.error("Failed to generate markdown report: %s", e)

    # 2. Compile plain-text report
    try:
        txt_lines = []
        txt_lines.append("======================================================================")
        txt_lines.append(f"VISIONGPT TRAINING EXPERIMENT SUMMARY REPORT: {metadata.get('experiment_name')}")
        txt_lines.append("======================================================================")
        txt_lines.append(f"Timestamp        : {metadata.get('timestamp')}")
        txt_lines.append(f"Runtime Duration : {duration_str}")
        txt_lines.append(f"Git commit hash  : {metadata.get('git_commit_hash', 'N/A')}")
        txt_lines.append("----------------------------------------------------------------------")
        txt_lines.append("\n[OPTIMAL VAL SCORES]")
        
        for val_k, epoch_k, label in metrics_mapping:
            val = best_metrics.get(val_k)
            epoch = best_metrics.get(epoch_k, 0)
            val_str = f"{val:.4f}" if val is not None else "N/A"
            epoch_str = str(epoch) if val is not None else "N/A"
            txt_lines.append(f"  {label:<25}: {val_str:<12} (Epoch {epoch_str})")

        txt_lines.append(f"  Average Epoch Duration   : {avg_time:.2f} seconds")
        txt_lines.append(f"  Total epochs recorded    : {len(records)}")

        txt_lines.append("\n[EXPERIMENT METADATA]")
        txt_lines.append(f"  Dataset Name   : {metadata.get('dataset')}")
        txt_lines.append(f"  Batch Size     : {metadata.get('batch_size')}")
        txt_lines.append(f"  Optimizer      : {metadata.get('optimizer')}")
        txt_lines.append(f"  Learning Rate  : {metadata.get('learning_rate')}")
        txt_lines.append(f"  Checkpoint Path: {metadata.get('checkpoint_path')}")

        txt_lines.append("\n[HARDWARE Vitals]")
        txt_lines.append(f"  Operating System : {metadata.get('os')}")
        txt_lines.append(f"  CPU Model        : {metadata.get('cpu')}")
        txt_lines.append(f"  GPU Model        : {metadata.get('gpu')}")
        txt_lines.append(f"  Total Memory RAM : {metadata.get('ram')} GB")
        txt_lines.append(f"  TensorFlow Ver   : v{metadata.get('tensorflow_version')}")
        txt_lines.append("======================================================================")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))
        logger.debug("Saved plain-text report to %s", txt_path)
    except Exception as e:
        logger.error("Failed to generate plain-text report: %s", e)
