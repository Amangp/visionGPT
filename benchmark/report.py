"""Reporting and visualization generator for VisionGPT.

This module serializes benchmark results into JSON, CSV, and formatted TXT files,
generates a styled console dashboard, and renders latency, memory, and
throughput graphs using matplotlib.
"""

import csv
import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """Helper to recursively flatten a dictionary for CSV exporting.

    Args:
        d: The dictionary to flatten.
        parent_key: Accumulated prefix.
        sep: Separator between nested keys.

    Returns:
        A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def export_reports(results: Dict[str, Any], output_dir: str) -> None:
    """Export benchmark results to JSON, CSV, and TXT formats.

    Args:
        results: Complete benchmark results dictionary.
        output_dir: Directory where the reports should be written.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Export JSON
    json_path = os.path.join(output_dir, "benchmark.json")
    try:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Saved JSON report to %s", json_path)
    except Exception as e:
        logger.error("Failed to write JSON report: %s", e)

    # 2. Export CSV
    csv_path = os.path.join(output_dir, "benchmark.csv")
    try:
        flat_results = flatten_dict(results)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for key, val in sorted(flat_results.items()):
                # Omit raw lists of times from the CSV to keep it clean
                if isinstance(val, list):
                    continue
                writer.writerow([key, val])
        logger.info("Saved CSV report to %s", csv_path)
    except Exception as e:
        logger.error("Failed to write CSV report: %s", e)

    # 3. Export TXT
    txt_path = os.path.join(output_dir, "benchmark.txt")
    try:
        txt_content = generate_text_report(results)
        with open(txt_path, "w") as f:
            f.write(txt_content)
        logger.info("Saved TXT report to %s", txt_path)
    except Exception as e:
        logger.error("Failed to write TXT report: %s", e)


def generate_text_report(results: Dict[str, Any]) -> str:
    """Generate a clean, aligned, text-based summary of the results.

    Args:
        results: Complete benchmark results dictionary.

    Returns:
        A string formatted as a report.
    """
    lines = []
    lines.append("======================================================================")
    lines.append("                     VISIONGPT V3 BENCHMARK REPORT                    ")
    lines.append("======================================================================")

    # Hardware Info
    hw = results.get("hardware", {})
    lines.append("\n[HARDWARE DIAGNOSTICS]")
    lines.append(f"  Operating System : {hw.get('os', 'N/A')}")
    lines.append(f"  CPU Model        : {hw.get('cpu', 'N/A')}")
    lines.append(f"  System RAM       : {hw.get('ram_gb', 0.0)} GB")
    lines.append(f"  TensorFlow       : v{hw.get('tensorflow_version', 'N/A')}")
    lines.append(f"  Python           : v{hw.get('python_version', 'N/A')}")
    lines.append(f"  CUDA Version     : {hw.get('cuda_version', 'N/A')} (Available: {hw.get('cuda_available', False)})")
    lines.append(f"  Visible GPUs     : {hw.get('gpus_available_count', 0)} ({', '.join(hw.get('gpu_devices', []))})")

    # Model Info
    m_info = results.get("model", {})
    if m_info:
        lines.append("\n[MODEL METRICS]")
        lines.append(f"  Total Parameters      : {m_info.get('total_params', 0):,}")
        lines.append(f"  Trainable Params      : {m_info.get('trainable_params', 0):,}")
        lines.append(f"  Non-trainable Params  : {m_info.get('non_trainable_params', 0):,}")
        lines.append(f"  Model Size footprint  : {m_info.get('size_mb', 0.0):.2f} MB")
        lines.append(f"  Total Model Layers    : {m_info.get('layer_count', 0)}")
        lines.append(f"  Encoder Parameters    : {m_info.get('encoder_params', 0):,}")
        lines.append(f"  Decoder Parameters    : {m_info.get('decoder_params', 0):,}")
        lines.append(f"  Fusion Parameters     : {m_info.get('fusion_params', 0):,}")

    # Memory Stats
    mem = results.get("memory", {})
    if mem:
        lines.append("\n[MEMORY AUDIT]")
        lines.append(f"  Process Peak RAM      : {mem.get('peak_ram_mb', 0.0):.2f} MB")
        lines.append(f"  Process Start RAM     : {mem.get('start_ram_mb', 0.0):.2f} MB")
        lines.append(f"  Process End RAM       : {mem.get('end_ram_mb', 0.0):.2f} MB")
        lines.append(f"  Peak GPU VRAM Alloc   : {mem.get('peak_vram_mb', 0.0):.2f} MB")

    # Inference Benchmarks
    inf = results.get("inference", {})
    if inf:
        lines.append("\n[INFERENCE LATENCY & THROUGHPUT]")
        for bs, d in sorted(inf.items(), key=lambda x: int(x[0])):
            lines.append(f"  Batch Size {bs}:")
            lines.append(f"    Mean Latency : {d.get('mean_latency_ms', 0.0):.2f} ms")
            lines.append(f"    P50 / P95    : {d.get('p50_latency_ms', 0.0):.2f} ms / {d.get('p95_latency_ms', 0.0):.2f} ms")
            lines.append(f"    P99 Latency  : {d.get('p99_latency_ms', 0.0):.2f} ms")
            lines.append(f"    Throughput   : {d.get('throughput_images_per_sec', 0.0):.2f} images/sec")
            lines.append(f"    Warm-up Time : {d.get('warmup_time_sec', 0.0):.4f} sec")

    # Training Benchmarks
    train = results.get("training", {})
    if train:
        lines.append("\n[TRAINING PROFILES]")
        lines.append(f"  Batch Size            : {train.get('batch_size', 0)}")
        lines.append(f"  Mean Forward Pass     : {train.get('mean_forward_time_ms', 0.0):.2f} ms")
        lines.append(f"  Mean Backward Pass    : {train.get('mean_backward_time_ms', 0.0):.2f} ms")
        lines.append(f"  Mean Gradient Update  : {train.get('mean_gradient_update_time_ms', 0.0):.2f} ms")
        lines.append(f"  Mean Total Step Time  : {train.get('mean_step_time_ms', 0.0):.2f} ms")
        lines.append(f"  Training Throughput   : {train.get('samples_per_sec', 0.0):.2f} samples/sec")
        lines.append(f"  Estimated Epoch Time  : {train.get('epoch_time_sec', 0.0):.2f} sec")

    lines.append("\n======================================================================")
    return "\n".join(lines)


def print_console_report(results: Dict[str, Any]) -> None:
    """Print the formatted dashboard summary directly to standard output.

    Args:
        results: Complete benchmark results dictionary.
    """
    report = generate_text_report(results)
    print(report)


def generate_plots(results: Dict[str, Any], output_dir: str) -> None:
    """Render and save visualization charts using matplotlib.

    Gracefully logs a warning if matplotlib is not installed.

    Args:
        results: Complete benchmark results dictionary.
        output_dir: Directory where figures should be saved.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib is not installed in the current environment. "
            "Skipping PNG plot visualizations."
        )
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Latency distribution plot (latency.png)
    try:
        plt.figure(figsize=(10, 5))
        inf_data = results.get("inference", {})
        if inf_data:
            for bs, data in sorted(inf_data.items(), key=lambda x: int(x[0])):
                latencies = data.get("latencies_ms", [])
                if latencies:
                    plt.plot(latencies, label=f"Batch Size {bs}", alpha=0.8)
            plt.title("Inference Latency Distribution Across Steps")
            plt.xlabel("Step")
            plt.ylabel("Latency (ms)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No Inference Latency Data Available", ha="center", va="center")
        
        latency_path = os.path.join(output_dir, "latency.png")
        plt.savefig(latency_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved latency visualization to %s", latency_path)
    except Exception as e:
        logger.error("Failed to generate latency.png: %s", e)

    # 2. Peak Memory Usage (memory.png)
    try:
        plt.figure(figsize=(8, 5))
        mem_data = results.get("memory", {})
        categories = []
        values = []
        colors = []

        if mem_data:
            if mem_data.get("peak_ram_mb", 0) > 0:
                categories.append("Peak RAM")
                values.append(mem_data["peak_ram_mb"])
                colors.append("#1f77b4")
            if mem_data.get("peak_vram_mb", 0) > 0:
                categories.append("Peak VRAM")
                values.append(mem_data["peak_vram_mb"])
                colors.append("#2ca02c")

        if values:
            plt.bar(categories, values, color=colors, width=0.5)
            plt.title("Peak Memory Consumption")
            plt.ylabel("Memory (MB)")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            for i, val in enumerate(values):
                plt.text(i, val + (max(values) * 0.01), f"{val:.2f} MB", ha="center", va="bottom", fontweight="bold")
        else:
            plt.text(0.5, 0.5, "No Memory Consumption Data Available", ha="center", va="center")

        memory_path = os.path.join(output_dir, "memory.png")
        plt.savefig(memory_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved memory visualization to %s", memory_path)
    except Exception as e:
        logger.error("Failed to generate memory.png: %s", e)

    # 3. Throughput across Batch Sizes (throughput.png)
    try:
        plt.figure(figsize=(8, 5))
        inf_data = results.get("inference", {})
        batch_sizes = []
        throughputs = []

        if inf_data:
            for bs, data in sorted(inf_data.items(), key=lambda x: int(x[0])):
                batch_sizes.append(str(bs))
                throughputs.append(data.get("throughput_questions_per_sec", 0.0))

        if throughputs:
            plt.bar(batch_sizes, throughputs, color="#9467bd", width=0.5)
            plt.title("Inference Throughput Across Batch Sizes")
            plt.xlabel("Batch Size")
            plt.ylabel("Throughput (Queries/sec)")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            for i, val in enumerate(throughputs):
                plt.text(i, val + (max(throughputs) * 0.01), f"{val:.2f} Q/s", ha="center", va="bottom", fontweight="bold")
        else:
            plt.text(0.5, 0.5, "No Throughput Data Available", ha="center", va="center")

        throughput_path = os.path.join(output_dir, "throughput.png")
        plt.savefig(throughput_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved throughput visualization to %s", throughput_path)
    except Exception as e:
        logger.error("Failed to generate throughput.png: %s", e)
