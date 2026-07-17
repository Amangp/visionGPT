"""Utility functions for VisionGPT evaluation package.

This module provides text normalization, tokenization, n-gram generation,
and robust input parsing to handle lists of strings, numpy arrays, and
TensorFlow tensors in training/evaluation loops.
"""

import re
import string
from typing import Any, Dict, List, Tuple


def preprocess_text(text: str) -> str:
    """Normalize a string: lowercase, remove punctuation, collapse whitespace.

    Args:
        text: The input string to normalize.

    Returns:
        The normalized string.
    """
    if not isinstance(text, str):
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        else:
            text = str(text) if text is not None else ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation using regex to eliminate any non-alphanumeric/non-space character
    text = re.sub(r"[^\w\s]", "", text)

    # Collapse multiple whitespaces and strip
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize a string by normalizing it and splitting on whitespace.

    Args:
        text: The input string to tokenize.

    Returns:
        A list of string tokens.
    """
    normalized = preprocess_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def get_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    """Generate n-gram frequency counts for a token list.

    Args:
        tokens: A list of string tokens.
        n: The size of the n-gram (e.g., 1 for unigrams, 2 for bigrams).

    Returns:
        A dictionary mapping n-gram tuples to their frequency counts.
    """
    counts: Dict[Tuple[str, ...], int] = {}
    if n < 1 or len(tokens) < n:
        return counts

    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def parse_input_to_strings(inputs: Any) -> List[str]:
    """Parse various inputs (lists, numpy arrays, TF Tensors) into a list of strings.

    Handles eager TensorFlow tensors, NumPy arrays, lists of bytes/strings,
    and single inputs.

    Args:
        inputs: Input data to parse.

    Returns:
        A list of Python strings.
    """
    if inputs is None:
        return []

    # Handle TensorFlow tensors (eager)
    if hasattr(inputs, "numpy") and callable(inputs.numpy):
        inputs = inputs.numpy()

    # Handle NumPy arrays or other list-like objects with a tolist() method
    if hasattr(inputs, "tolist") and callable(inputs.tolist):
        inputs = inputs.tolist()

    # If it's a single string or bytes object, wrap in a list
    if isinstance(inputs, (str, bytes)):
        inputs = [inputs]

    result: List[str] = []
    if isinstance(inputs, (list, tuple)):
        for item in inputs:
            if item is None:
                result.append("")
            elif isinstance(item, bytes):
                result.append(item.decode("utf-8", errors="ignore"))
            elif isinstance(item, str):
                result.append(item)
            elif hasattr(item, "numpy") and callable(item.numpy):
                # Eager tensor nested inside list
                val = item.numpy()
                if isinstance(val, bytes):
                    result.append(val.decode("utf-8", errors="ignore"))
                else:
                    result.append(str(val))
            else:
                result.append(str(item))
    else:
        # Fallback for single values
        result.append(str(inputs))

    return result
