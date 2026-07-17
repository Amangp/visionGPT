"""ROUGE-L (Longest Common Subsequence) metric implementation.

This module provides the implementation of ROUGE-L using a space-optimized
dynamic programming LCS algorithm. It computes ROUGE-L precision, recall, and
F-measure (supporting the standard COCO beta parameter of 1.2).
"""

import logging
from typing import List
from visiongpt.evaluation.utils import parse_input_to_strings, tokenize

logger = logging.getLogger(__name__)


def lcs_length(x: List[str], y: List[str]) -> int:
    """Compute the length of the longest common subsequence of two lists of tokens.

    This uses a space-efficient dynamic programming approach with O(min(len(x), len(y)))
    extra space.

    Args:
        x: First sequence of tokens.
        y: Second sequence of tokens.

    Returns:
        The length of the longest common subsequence.
    """
    if not x or not y:
        return 0

    # Ensure y is the shorter sequence to minimize space complexity
    if len(x) < len(y):
        x, y = y, x

    m, n = len(x), len(y)
    dp = [0] * (n + 1)

    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    return dp[n]


def compute_pair_rouge_l(cand_text: str, ref_text: str, beta: float = 1.2) -> float:
    """Calculate the ROUGE-L score for a single candidate and reference pair.

    Args:
        cand_text: Candidate sentence.
        ref_text: Reference sentence.
        beta: Weighting parameter between precision and recall (default: 1.2 as in COCO).

    Returns:
        The ROUGE-L score as a float between 0.0 and 1.0.
    """
    cand_tokens = tokenize(cand_text)
    ref_tokens = tokenize(ref_text)

    if not cand_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(cand_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)

    # ROUGE-L F-measure formula
    beta_sq = beta**2
    score = ((1.0 + beta_sq) * precision * recall) / (recall + beta_sq * precision)
    return score


def compute_rouge_l(
    predictions: List[str], references: List[str], beta: float = 1.2
) -> float:
    """Compute the average ROUGE-L score for a batch of predictions and references.

    Args:
        predictions: List of candidate/predicted strings.
        references: List of reference/ground-truth strings.
        beta: Weighting parameter between precision and recall (default: 1.2 as in COCO).

    Returns:
        The average ROUGE-L score over the batch.
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

    total_score = 0.0
    for pred, ref in zip(preds_parsed, refs_parsed):
        total_score += compute_pair_rouge_l(pred, ref, beta=beta)

    return total_score / len(preds_parsed)
