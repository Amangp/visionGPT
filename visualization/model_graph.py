"""Core Matplotlib rendering engine for VisionGPT diagrams.

Draws vertical component architecture flowcharts and horizontal computational pipelines
using patches, labels, arrows, and legends based on active ThemeStyle records.
"""

import logging
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from visiongpt.visualization.layers import LayerNode
from visiongpt.visualization.styles import ThemeStyle

logger = logging.getLogger(__name__)


def draw_architecture_diagram(nodes: List[LayerNode], theme: ThemeStyle) -> plt.Figure:
    """Render a vertical component architecture flowchart.

    Draws boxes for input, encoder, fusion layer, transformer decoder, and output.
    Formats metrics text inside each box, connects them with filled arrows,
    and appends a color-coded legend.

    Args:
        nodes: List of LayerNode configurations.
        theme: Active ThemeStyle preset mapping.

    Returns:
        The generated Matplotlib Figure instance.
    """
    logger.debug("Rendering vertical architecture flowchart under '%s' theme.", theme.name)

    # Set canvas dimensions
    fig, ax = plt.subplots(figsize=(9, 11), facecolor=theme.bg_color)
    ax.set_facecolor(theme.bg_color)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    num_nodes = len(nodes)
    box_w = 64.0
    box_h = 11.5
    
    # Calculate vertical spacing coordinates (from top 88 down to bottom 12)
    y_coords = [88.0 - i * 18.2 for i in range(num_nodes)]

    # Draw nodes
    for i, node in enumerate(nodes):
        x_center = 50.0
        y_center = y_coords[i]

        x0 = x_center - box_w / 2.0
        y0 = y_center - box_h / 2.0

        # Map block colors according to category
        if node.category == "input":
            node_color = theme.input_color
        elif node.category == "encoder":
            node_color = theme.encoder_color
        elif node.category == "fusion":
            node_color = theme.fusion_color
        elif node.category == "decoder":
            node_color = theme.decoder_color
        else:
            node_color = theme.output_color

        # Render rounded rectangle card path
        rect = patches.FancyBboxPatch(
            (x0, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.5,rounding_size=1.0",
            facecolor=node_color,
            edgecolor=theme.border_color,
            linewidth=1.2,
            mutation_scale=1.0,
        )
        ax.add_patch(rect)

        # Draw component labels inside box
        # 1. Header (Title Name)
        ax.text(
            x_center,
            y0 + box_h - 1.8,
            node.name.upper(),
            color=theme.text_primary,
            fontfamily=theme.font_family,
            fontsize=theme.header_font_size,
            fontweight="bold",
            ha="center",
            va="top",
        )

        # 2. Details: shapes, params, activation
        details_y = y0 + box_h - 3.8
        
        # Format shapes text
        shape_text = f"Input: {node.input_shape}   |   Output: {node.output_shape}"
        ax.text(
            x_center,
            details_y,
            shape_text,
            color=theme.text_secondary,
            fontfamily=theme.font_family,
            fontsize=theme.body_font_size,
            ha="center",
            va="top",
        )

        # Format parameter counts and activation
        param_text = f"Params: {node.total_params:,} (Trainable: {node.trainable_params:,})"
        if node.activation != "N/A" and node.activation != "N/A":
            param_text += f"   |   Act: {node.activation}"
            
        ax.text(
            x_center,
            details_y - 2.0,
            param_text,
            color=theme.text_secondary,
            fontfamily=theme.font_family,
            fontsize=theme.body_font_size,
            ha="center",
            va="top",
        )

        # 3. Short description
        ax.text(
            x_center,
            details_y - 4.2,
            node.description,
            color=theme.text_primary,
            fontfamily=theme.font_family,
            fontsize=theme.body_font_size - 0.5,
            fontstyle="italic",
            ha="center",
            va="top",
            wrap=True,
        )

    # Draw arrow connections between nodes
    for i in range(num_nodes - 1):
        y_curr_bottom = y_coords[i] - box_h / 2.0 - 0.5
        y_next_top = y_coords[i + 1] + box_h / 2.0 + 0.5
        
        ax.annotate(
            "",
            xy=(50.0, y_next_top),
            xytext=(50.0, y_curr_bottom),
            arrowprops=dict(
                arrowstyle="-|>",
                color=theme.arrow_color,
                lw=theme.arrow_width,
                mutation_scale=15.0,  # size of arrowhead
                shrinkA=0,
                shrinkB=0,
            ),
        )

    # Render Legend block at bottom (y = 5)
    legend_y = 3.5
    legend_categories = [
        ("Input", theme.input_color),
        ("Vision Encoder", theme.encoder_color),
        ("Fusion Layer", theme.fusion_color),
        ("Decoder Block", theme.decoder_color),
        ("Classifier", theme.output_color),
    ]

    # Draw legend badges
    start_x = 10.0
    spacing_x = 16.5
    for idx, (label, color) in enumerate(legend_categories):
        x = start_x + idx * spacing_x
        # Draw small color patch
        patch = patches.FancyBboxPatch(
            (x, legend_y),
            3.0,
            2.0,
            boxstyle="round,pad=0.2",
            facecolor=color,
            edgecolor=theme.border_color,
            linewidth=0.8,
        )
        ax.add_patch(patch)
        # Write text
        ax.text(
            x + 3.8,
            legend_y + 1.0,
            label,
            color=theme.text_primary,
            fontfamily=theme.font_family,
            fontsize=theme.body_font_size - 0.5,
            va="center",
            ha="left",
        )

    plt.tight_layout()
    return fig


def draw_computational_flow(theme: ThemeStyle) -> plt.Figure:
    """Render a horizontal computational dataflow diagram.

    Draws inputs flowing through preprocessing, encoding, projection, decoding,
    and classification modules.

    Args:
        theme: Active ThemeStyle preset mapping.

    Returns:
        The generated Matplotlib Figure instance.
    """
    logger.debug("Rendering horizontal computational dataflow under '%s' theme.", theme.name)

    fig, ax = plt.subplots(figsize=(11, 4), facecolor=theme.bg_color)
    ax.set_facecolor(theme.bg_color)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    steps = [
        "Raw Input\n(Image & Text)",
        "Image\nPreprocessing",
        "CNN Encoder\n(EfficientNet)",
        "Fusion Projection\n(Dense Layer)",
        "Transformer\nDecoder Layers",
        "Softmax\nClassifier",
        "Target Output\nTokens",
    ]

    num_steps = len(steps)
    box_w = 11.5
    box_h = 24.0
    x_coords = [6.0 + i * 14.5 for i in range(num_steps)]
    y_center = 50.0

    # Draw horizontal blocks
    for i, step_text in enumerate(steps):
        x_center = x_coords[i]
        x0 = x_center - box_w / 2.0
        y0 = y_center - box_h / 2.0

        # Alternate block colors for visualization variety
        if i == 0 or i == num_steps - 1:
            bg = theme.input_color
        elif i == 1 or i == 2:
            bg = theme.encoder_color
        elif i == 3:
            bg = theme.fusion_color
        elif i == 4:
            bg = theme.decoder_color
        else:
            bg = theme.output_color

        rect = patches.FancyBboxPatch(
            (x0, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.4,rounding_size=0.8",
            facecolor=bg,
            edgecolor=theme.border_color,
            linewidth=1.2,
        )
        ax.add_patch(rect)

        # Write text
        ax.text(
            x_center,
            y_center,
            step_text,
            color=theme.text_primary,
            fontfamily=theme.font_family,
            fontsize=theme.body_font_size,
            fontweight="bold",
            ha="center",
            va="center",
        )

    # Draw arrow connections between stages
    for i in range(num_steps - 1):
        x_curr_right = x_coords[i] + box_w / 2.0 + 0.3
        x_next_left = x_coords[i + 1] - box_w / 2.0 - 0.3

        ax.annotate(
            "",
            xy=(x_next_left, y_center),
            xytext=(x_curr_right, y_center),
            arrowprops=dict(
                arrowstyle="-|>",
                color=theme.arrow_color,
                lw=theme.arrow_width,
                mutation_scale=12.0,
                shrinkA=0,
                shrinkB=0,
            ),
        )

    plt.tight_layout()
    return fig
