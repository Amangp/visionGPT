"""Export utilities for diagrams and summaries in VisionGPT.

Handles saving Matplotlib figures as PNG/SVG/PDF, and serializing
model summaries to markdown and text files.
"""

import logging
import os
from typing import Any, List

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def save_matplotlib_figure(
    fig: plt.Figure, filepath: str, dpi: int = 300
) -> None:
    """Save a Matplotlib Figure as a PNG, SVG, or PDF.

    Args:
        fig: The Matplotlib Figure instance.
        filepath: Absolute or relative output file path.
        dpi: Target DPI for raster images (default: 300).
    """
    if fig is None:
        logger.error("No active figure to export.")
        return

    try:
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        fig.savefig(
            filepath,
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            bbox_inches="tight",
        )
        logger.info("Exported figure successfully to %s", filepath)
    except Exception as e:
        logger.error("Failed to export figure to %s: %s", filepath, e)


def write_text_summary(
    nodes: List[Any],
    total: int,
    trainable: int,
    frozen: int,
    size_mb: float,
    filepath: str,
) -> None:
    """Export a tabular, formatted plain-text summary of the model variables.

    Args:
        nodes: Extracted LayerNode objects.
        total: Total parameters count.
        trainable: Trainable parameters count.
        frozen: Frozen parameters count.
        size_mb: Memory size footprint in MB.
        filepath: Destination file path.
    """
    try:
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        lines = []
        lines.append("======================================================================")
        lines.append("                     VISIONGPT V3 MODEL SUMMARY                       ")
        lines.append("======================================================================")
        lines.append(f"  Estimated Size Footprint : {size_mb:.2f} MB")
        lines.append(f"  Total Parameter Count    : {total:,}")
        lines.append(f"  Trainable Parameters     : {trainable:,}")
        lines.append(f"  Frozen/Static Parameters : {frozen:,}")
        lines.append("----------------------------------------------------------------------")
        lines.append(
            f"  {'Layer Name':<22} | {'Output Shape':<30} | {'Params':<12} | {'Act':<8}"
        )
        lines.append("----------------------------------------------------------------------")
        
        for n in nodes:
            # We omit the input node from layers list if it has 0 params
            if n.category == "input":
                continue
            lines.append(
                f"  {n.name:<22} | {n.output_shape:<30} | {n.total_params:<12,} | {n.activation:<8}"
            )
        lines.append("======================================================================")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Saved text summary to %s", filepath)
    except Exception as e:
        logger.error("Failed to write text summary: %s", e)


def write_markdown_summary(
    nodes: List[Any],
    total: int,
    trainable: int,
    frozen: int,
    size_mb: float,
    filepath: str,
) -> str:
    """Export a structured Markdown table summary of the model layers.

    Args:
        nodes: Extracted LayerNode objects.
        total: Total parameters count.
        trainable: Trainable parameters count.
        frozen: Frozen/static parameters count.
        size_mb: Memory size footprint in MB.
        filepath: Destination file path.

    Returns:
        The markdown content string.
    """
    md = []
    try:
        md.append("# VisionGPT v3 Model Structure Summary")
        md.append("")
        md.append("## Core Statistics")
        md.append(f"- **Total Model Size Footprint:** `{size_mb:.2f} MB` ")
        md.append(f"- **Total Parameters:** `{total:,}`")
        md.append(f"- **Trainable Parameters:** `{trainable:,}`")
        md.append(f"- **Frozen/Static Parameters:** `{frozen:,}`")
        md.append("")
        md.append("## Component Blocks Details")
        md.append("| Component Layer | Output Shape | Parameters | Activation | Description |")
        md.append("| :--- | :---: | :---: | :---: | :--- |")
        
        for n in nodes:
            md.append(
                f"| {n.name} | `{n.output_shape}` | {n.total_params:,} | {n.activation} | {n.description} |"
            )

        md_content = "\n".join(md)
        
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Saved markdown summary to %s", filepath)
        return md_content
    except Exception as e:
        logger.error("Failed to write markdown summary: %s", e)
        return ""
