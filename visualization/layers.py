"""Layer node definitions and parser for VisionGPT architecture.

Defines the LayerNode dataclass representing node configurations in the
architecture flowchart. Inspects the model to populate nodes dynamically.
"""

from dataclasses import dataclass
from typing import Any, List
import tensorflow as tf

from visiongpt.visualization.utils import (
    count_layer_params,
    get_activation_name,
    get_layer_io_shapes,
)


@dataclass
class LayerNode:
    """Represents a component block in the architecture diagram."""

    name: str
    category: str  # 'input', 'encoder', 'fusion', 'decoder', 'output'
    input_shape: str
    output_shape: str
    total_params: int
    trainable_params: int
    frozen_params: int
    activation: str
    description: str


def extract_visiongpt_layers(model: tf.keras.Model) -> List[LayerNode]:
    """Inspect and extract properties for the core blocks of a VisionGPT model.

    Parses the Vision Encoder, Fusion Layer, and Transformer Decoder.

    Args:
        model: The tf.keras.Model or VisionGPT model to inspect.

    Returns:
        A list of LayerNode configurations.
    """
    nodes: List[LayerNode] = []

    # 1. Input Node
    nodes.append(
        LayerNode(
            name="Input Image",
            category="input",
            input_shape="N/A",
            output_shape="(None, 224, 224, 3)",
            total_params=0,
            trainable_params=0,
            frozen_params=0,
            activation="N/A",
            description="Raw RGB input images (224x224)",
        )
    )

    # 2. Vision Encoder
    encoder = getattr(model, "vision_encoder", None)
    tot, tr, fr = count_layer_params(encoder)
    in_sh, out_sh = get_layer_io_shapes(encoder)
    act = get_activation_name(encoder)
    
    # Pre-build defaults if shapes are not initialized
    if in_sh == "N/A":
        in_sh = "(None, 224, 224, 3)"
    if out_sh == "N/A":
        out_sh = "(None, 7, 7, 1280)"
        
    nodes.append(
        LayerNode(
            name="Vision Encoder",
            category="encoder",
            input_shape=in_sh,
            output_shape=out_sh,
            total_params=tot if tot > 0 else 4049589,  # Default fallback if unbuilt
            trainable_params=tr if tot > 0 else 0,
            frozen_params=fr if tot > 0 else 4049589,
            activation=act if act != "N/A" else "Swish/Relu",
            description="CNN Feature Extractor (e.g. EfficientNet)",
        )
    )

    # 3. Fusion Layer
    fusion = getattr(model, "fusion_layer", None)
    tot, tr, fr = count_layer_params(fusion)
    in_sh, out_sh = get_layer_io_shapes(fusion)
    act = get_activation_name(fusion)
    
    if in_sh == "N/A":
        in_sh = "(None, 7, 7, 1280)"
    if out_sh == "N/A":
        out_sh = "(None, 49, 256)"

    nodes.append(
        LayerNode(
            name="Fusion Layer",
            category="fusion",
            input_shape=in_sh,
            output_shape=out_sh,
            total_params=tot if tot > 0 else 248136,
            trainable_params=tr if tot > 0 else 248136,
            frozen_params=fr if tot > 0 else 0,
            activation=act if act != "N/A" else "Linear",
            description="Projects & aligns visual features to embedding dimension",
        )
    )

    # 4. Transformer Decoder
    decoder = getattr(model, "answer_decoder", None)
    tot, tr, fr = count_layer_params(decoder)
    in_sh, out_sh = get_layer_io_shapes(decoder)
    act = get_activation_name(decoder)
    
    if in_sh == "N/A":
        in_sh = "(None, seq_len), (None, 49, 256)"
    if out_sh == "N/A":
        out_sh = "(None, seq_len, 256)"

    # We subtract output layer parameters from the main decoder body parameters
    output_layer = getattr(decoder, "output_layer", None) if decoder else None
    out_tot, out_tr, out_fr = count_layer_params(output_layer)

    decoder_tot = max(0, tot - out_tot)
    decoder_tr = max(0, tr - out_tr)
    decoder_fr = max(0, fr - out_fr)

    nodes.append(
        LayerNode(
            name="Transformer Decoder",
            category="decoder",
            input_shape=in_sh,
            output_shape=out_sh,
            total_params=decoder_tot if tot > 0 else 434882,
            trainable_params=decoder_tr if tot > 0 else 434882,
            frozen_params=decoder_fr if tot > 0 else 0,
            activation="Gelu",
            description="Autoregressively decodes tokens using masked self-attention",
        )
    )

    # 5. Output Classifier Layer
    vocab_size = getattr(model, "vocab_size", 10000)
    in_sh = "(None, seq_len, 256)"
    out_sh = f"(None, seq_len, {vocab_size})"

    nodes.append(
        LayerNode(
            name="Output Layer",
            category="output",
            input_shape=in_sh,
            output_shape=out_sh,
            total_params=out_tot if out_tot > 0 else (256 * vocab_size + vocab_size),
            trainable_params=out_tr if out_tot > 0 else (256 * vocab_size + vocab_size),
            frozen_params=out_fr if out_tot > 0 else 0,
            activation="Softmax",
            description="Probability distribution classifier over vocabulary tokens",
        )
    )

    return nodes
