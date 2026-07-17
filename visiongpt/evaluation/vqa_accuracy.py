"""VQA (Visual Question Answering) accuracy metrics implementation.

This module provides implementations for Exact Match, Token-level Accuracy (F1-score),
and Word-level Accuracy (normalized Levenshtein edit distance) for evaluation.
"""

import collections
import logging
from typing import List
from visiongpt.evaluation.utils import parse_input_to_strings, tokenize

logger = logging.getLogger(__name__)


def compute_pair_exact_match(pred: str, ref: str) -> float:
    """Compute Exact Match for a single prediction and reference.

    Args:
        pred: Predicted answer.
        ref: Reference/ground-truth answer.

    Returns:
        1.0 if preprocessed strings are identical, else 0.0.
    """
    # Using tokenize as it normalizes text by lowercasing and removing punctuation
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    return 1.0 if pred_tokens == ref_tokens else 0.0


def compute_pair_token_accuracy(pred: str, ref: str) -> float:
    """Compute Token Accuracy (F1 score of tokens) for a single prediction and reference.

    Args:
        pred: Predicted answer.
        ref: Reference/ground-truth answer.

    Returns:
        Token F1 score as a float between 0.0 and 1.0.
    """
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = collections.Counter(pred_tokens)
    ref_counter = collections.Counter(ref_tokens)

    # Compute intersection count
    intersection = sum((pred_counter & ref_counter).values())

    precision = intersection / len(pred_tokens)
    recall = intersection / len(ref_tokens)

    if precision + recall == 0.0:
        return 0.0

    f1 = (2.0 * precision * recall) / (precision + recall)
    return f1


def edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences.

    This uses a space-efficient O(min(len(seq1), len(seq2))) dynamic programming algorithm.

    Args:
        seq1: First sequence of tokens.
        seq2: Second sequence of tokens.

    Returns:
        The Levenshtein distance as an integer.
    """
    m, n = len(seq1), len(seq2)
    # Ensure seq2 is the shorter sequence
    if m < n:
        seq1, seq2 = seq2, seq1
        m, n = n, m

    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = min(
                    dp[j] + 1,  # Deletion
                    dp[j - 1] + 1,  # Insertion
                    prev + 1,  # Substitution
                )
            prev = temp

    return dp[n]


def compute_pair_word_accuracy(pred: str, ref: str) -> float:
    """Compute Word Accuracy (normalized Levenshtein distance) for a single pair.

    Args:
        pred: Predicted answer.
        ref: Reference/ground-truth answer.

    Returns:
        Word Accuracy score as a float between 0.0 and 1.0.
    """
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)

    max_len = max(len(pred_tokens), len(ref_tokens))
    if max_len == 0:
        return 1.0

    dist = edit_distance(pred_tokens, ref_tokens)
    accuracy = 1.0 - (dist / max_len)
    return accuracy


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Compute Exact Match (EM) over a batch of predictions and references.

    Args:
        predictions: List of candidate/predicted strings.
        references: List of reference/ground-truth strings.

    Returns:
        The average Exact Match score.
    """
    preds_parsed = parse_input_to_strings(predictions)
    refs_parsed = parse_input_to_strings(references)

    if not preds_parsed or not refs_parsed:
        return 0.0

    if len(preds_parsed) != len(refs_parsed):
        raise ValueError(
            f"Predictions (len={len(preds_parsed)}) and references (len={len(refs_parsed)}) "
            "must have the same length."
        )

    total_em = sum(
        compute_pair_exact_match(pred, ref) for pred, ref in zip(preds_parsed, refs_parsed)
    )
    return total_em / len(preds_parsed)


def compute_token_accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute average Token Accuracy (F1) over a batch of predictions and references.

    Args:
        predictions: List of candidate/predicted strings.
        references: List of reference/ground-truth strings.

    Returns:
        The average Token Accuracy score.
    """
    preds_parsed = parse_input_to_strings(predictions)
    refs_parsed = parse_input_to_strings(references)

    if not preds_parsed or not refs_parsed:
        return 0.0

    if len(preds_parsed) != len(refs_parsed):
        raise ValueError(
            f"Predictions (len={len(preds_parsed)}) and references (len={len(refs_parsed)}) "
            "must have the same length."
        )

    total_token_acc = sum(
        compute_pair_token_accuracy(pred, ref)
        for pred, ref in zip(preds_parsed, refs_parsed)
    )
    return total_token_acc / len(preds_parsed)


def compute_word_accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute average Word Accuracy (Levenshtein) over a batch of predictions and references.

    Args:
        predictions: List of candidate/predicted strings.
        references: List of reference/ground-truth strings.

    Returns:
        The average Word Accuracy score.
    """
    preds_parsed = parse_input_to_strings(predictions)
    refs_parsed = parse_input_to_strings(references)

    if not preds_parsed or not refs_parsed:
        return 0.0

    if len(preds_parsed) != len(refs_parsed):
        raise ValueError(
            f"Predictions (len={len(preds_parsed)}) and references (len={len(refs_parsed)}) "
            "must have the same length."
        )

    total_word_acc = sum(
        compute_pair_word_accuracy(pred, ref)
        for pred, ref in zip(preds_parsed, refs_parsed)
    )
    return total_word_acc / len(preds_parsed)
