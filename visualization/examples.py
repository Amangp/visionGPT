"""Demonstration example script for VisionGPT architecture visualizer.

Builds a VisionGPT model instance, analyzes its layout, draws vertical
and horizontal diagrams, renders parameter distributions, and writes files
to target folders.
"""

import logging
import os
import sys

# Configure basic console logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

# Ensure workspace root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from models.visiongpt import VisionGPT
    from visiongpt.visualization.architecture import ArchitectureVisualizer
    from visiongpt.visualization.exporter import save_matplotlib_figure
except ImportError as e:
    logger.error("Failed to import required packages: %s", e)
    sys.exit(1)


def run_visualization_demo() -> None:
    """Run the architecture visualization demonstration script."""
    logger.info("Initializing VisionGPT model...")
    # Instantiate the actual model with smaller dims for rapid visualization
    model = VisionGPT(
        vocab_size=1000,
        embed_dim=128,
        context_dropout_rate=0.10,
    )

    # Initialize model weights by running a single dummy call
    # Note: Keras layer shapes are only fully determined post-build
    import numpy as np
    import tensorflow as tf
    logger.info("Building model weights via a dummy forward pass...")
    dummy_image = tf.zeros((1, 224, 224, 3))
    dummy_text = tf.zeros((1, 5), dtype=tf.int64)
    _ = model((dummy_image, dummy_text), training=False)

    logger.info("Initializing ArchitectureVisualizer...")
    # Initialize visualizer directing outputs to workspace root directories
    visualizer = ArchitectureVisualizer(model=model, output_dir=".")

    # 1. Generate and export the vertical architecture diagrams (PNG, PDF, SVG)
    logger.info("Generating and exporting vertical architecture diagrams...")
    # Render under the default 'light' theme
    fig_light = visualizer.generate_architecture(theme="light")
    visualizer.export_png("architecture_light.png")
    visualizer.export_pdf("architecture_light.pdf")
    visualizer.export_svg("architecture_light.svg")

    # Render under 'dark' theme for variety
    fig_dark = visualizer.generate_architecture(theme="dark")
    visualizer.export_png("architecture_dark.png")

    # Render under 'monochrome' paper publication theme
    fig_mono = visualizer.generate_architecture(theme="monochrome")
    visualizer.export_png("architecture_monochrome.png")

    # 2. Generate and export the horizontal computational flow diagram
    logger.info("Generating and exporting horizontal computational flow diagram...")
    flow_fig = visualizer.generate_computational_flow(theme="light")
    flow_path = os.path.join(visualizer.arch_dir, "computational_flow.png")
    save_matplotlib_figure(flow_fig, flow_path)

    # 3. Generate parameter distribution pie chart
    logger.info("Generating and exporting weight distribution pie chart...")
    param_fig = visualizer.generate_parameter_chart()
    param_path = os.path.join(visualizer.fig_dir, "parameter_distribution.png")
    save_matplotlib_figure(param_fig, param_path)

    # 4. Generate Markdown and TXT model summary reports
    logger.info("Generating and exporting text/markdown summaries...")
    md_summary = visualizer.generate_summary()

    logger.info("======================================================================")
    logger.info("VisionGPT Architecture Visualizer Demo Completed Successfully!")
    logger.info("Look at these folders under the workspace root:")
    logger.info("  - architecture/ (contains png/pdf/svg flowcharts)")
    logger.info("  - figures/      (contains parameter_distribution.png)")
    logger.info("  - docs/         (contains model_summary.md and model_summary.txt)")
    logger.info("======================================================================")


if __name__ == "__main__":
    run_visualization_demo()
