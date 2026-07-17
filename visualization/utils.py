"""Utilities for model reflection and weight analysis in VisionGPT.

Extracts layer parameters, trainable/frozen splits, output shapes, and
computes memory footprints from tf.keras.Model objects.
"""

import logging
from typing import Any, Dict, Tuple
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def count_layer_params(layer: Any) -> Tuple[int, int, int]:
    """Compute (total, trainable, frozen) parameters for a layer or model.

    Args:
        layer: A tf.keras.layers.Layer or tf.keras.Model.

    Returns:
        A tuple of (total_params, trainable_params, frozen_params).
    """
    if layer is None or not hasattr(layer, "variables"):
        return 0, 0, 0

    total = sum(int(np.prod(v.shape)) for v in layer.variables)
    trainable = sum(int(np.prod(v.shape)) for v in layer.trainable_variables)
    frozen = total - trainable
    return total, trainable, frozen


def get_activation_name(layer: Any) -> str:
    """Retrieve the activation function name of a layer.

    Args:
        layer: Keras Layer instance.

    Returns:
        The name of the activation function, or 'Linear' if none, or 'N/A'.
    """
    if layer is None:
        return "N/A"

    # Standard activation attribute check
    if hasattr(layer, "activation") and layer.activation is not None:
        act = layer.activation
        if hasattr(act, "__name__"):
            return str(act.__name__).title()
        return str(act).split()[-1].strip(">").title()

    # Sequential model check (inspect last layer)
    if isinstance(layer, tf.keras.Sequential) and layer.layers:
        return get_activation_name(layer.layers[-1])

    return "Linear"


def get_layer_io_shapes(layer: Any) -> Tuple[str, str]:
    """Inspect and format the input and output shapes of a layer.

    Args:
        layer: Keras Layer or Model instance.

    Returns:
        A tuple of (input_shape_str, output_shape_str).
    """
    if layer is None:
        return "N/A", "N/A"

    def format_shape(shape) -> str:
        if shape is None:
            return "N/A"
        if isinstance(shape, list):
            if len(shape) == 1:
                return format_shape(shape[0])
            return str([format_shape(s) for s in shape])
        if hasattr(shape, "as_list") and callable(shape.as_list):
            try:
                shape = shape.as_list()
            except Exception:
                pass
        return str(tuple(shape))

    input_shape = "N/A"
    output_shape = "N/A"

    try:
        if hasattr(layer, "input_shape") and layer.input_shape is not None:
            input_shape = format_shape(layer.input_shape)
    except Exception:
        pass

    try:
        if hasattr(layer, "output_shape") and layer.output_shape is not None:
            output_shape = format_shape(layer.output_shape)
    except Exception:
        pass

    return input_shape, output_shape


def compute_memory_usage_mb(total_params: int) -> float:
    """Calculate the estimated memory footprint in MB assuming 32-bit floats.

    Args:
        total_params: The number of parameters.

    Returns:
        The estimated memory size in MB.
    """
    return (total_params * 4) / (1024 * 1024)
