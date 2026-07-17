"""Unit tests and verification script for the VisionGPT visualization package.

Tests model analysis, parameter counting, and the creation of PNG/SVG/PDF/MD/TXT
outputs for architecture layouts, computational flows, and pie charts.
"""

import os
import shutil
import unittest
import numpy as np
import tensorflow as tf

from visiongpt.visualization.architecture import ArchitectureVisualizer
from visiongpt.visualization.utils import count_layer_params, get_activation_name
from models.visiongpt import VisionGPT


class TestVisualizationModule(unittest.TestCase):
    """Test suite to verify VisionGPT Architecture Visualization and Reporting."""

    @classmethod
    def setUpClass(cls):
        """Setup output directories and mock a VisionGPT model."""
        cls.test_dir = "./test_visualization_runs"
        os.makedirs(cls.test_dir, exist_ok=True)

        # Instantiate actual model with small sizes for tests
        cls.model = VisionGPT(
            vocab_size=500,
            embed_dim=32,
            context_dropout_rate=0.10,
        )
        
        # Build model
        dummy_image = tf.zeros((1, 224, 224, 3))
        dummy_text = tf.zeros((1, 5), dtype=tf.int64)
        _ = cls.model((dummy_image, dummy_text), training=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up generated test outputs."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_utils_layer_reflection(self):
        """Test layer parameter counts and activations reflection."""
        total, trainable, frozen = count_layer_params(self.model)
        self.assertGreater(total, 0)
        self.assertGreater(trainable, 0)
        self.assertGreater(frozen, 0)

        # Test sub-module activation checks
        act = get_activation_name(getattr(self.model, "fusion_layer", None))
        self.assertEqual(act, "Linear")

    def test_architecture_generator(self):
        """Test generating architecture diagram, flow, and parameters chart."""
        visualizer = ArchitectureVisualizer(model=self.model, output_dir=self.test_dir)

        # Test vertical architecture diagrams for different themes
        fig_light = visualizer.generate_architecture(theme="light")
        self.assertIsNotNone(fig_light)
        visualizer.export_png("test_arch_light.png")
        visualizer.export_pdf("test_arch_light.pdf")
        visualizer.export_svg("test_arch_light.svg")

        fig_dark = visualizer.generate_architecture(theme="dark")
        self.assertIsNotNone(fig_dark)
        visualizer.export_png("test_arch_dark.png")

        fig_mono = visualizer.generate_architecture(theme="monochrome")
        self.assertIsNotNone(fig_mono)
        visualizer.export_png("test_arch_mono.png")

        # Check file existence in subfolders
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "architecture", "test_arch_light.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "architecture", "test_arch_light.pdf")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "architecture", "test_arch_light.svg")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "architecture", "test_arch_dark.png")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "architecture", "test_arch_mono.png")))

    def test_computational_flow_generator(self):
        """Test horizontal computational dataflow generation."""
        visualizer = ArchitectureVisualizer(model=self.model, output_dir=self.test_dir)
        flow_fig = visualizer.generate_computational_flow(theme="light")
        self.assertIsNotNone(flow_fig)

        path = os.path.join(visualizer.arch_dir, "test_comp_flow.png")
        from visiongpt.visualization.exporter import save_matplotlib_figure
        save_matplotlib_figure(flow_fig, path)
        self.assertTrue(os.path.exists(path))

    def test_parameter_pie_chart(self):
        """Test weight parameter distribution pie chart generation."""
        visualizer = ArchitectureVisualizer(model=self.model, output_dir=self.test_dir)
        param_fig = visualizer.generate_parameter_chart()
        self.assertIsNotNone(param_fig)

        path = os.path.join(visualizer.fig_dir, "test_param_pie.png")
        from visiongpt.visualization.exporter import save_matplotlib_figure
        save_matplotlib_figure(param_fig, path)
        self.assertTrue(os.path.exists(path))

    def test_summary_reports(self):
        """Test text and markdown reports generation."""
        visualizer = ArchitectureVisualizer(model=self.model, output_dir=self.test_dir)
        md_summary = visualizer.generate_summary()

        self.assertNotEqual(md_summary, "")
        self.assertIn("VisionGPT", md_summary)
        self.assertIn("Encoder", md_summary)

        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "docs", "model_summary.md")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "docs", "model_summary.txt")))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    unittest.main()
