"""BLEU (Bilingual Evaluation Understudy) metric implementation.

This module provides functions to calculate BLEU-1, BLEU-2, BLEU-3, and BLEU-4
scores for a batch of predictions and references. It handles edge cases like
empty strings, short sentences, and employs smoothing to avoid zero precision
on higher-order n-grams.
"""

import math
import logging
from typing import Dict, List, Tuple
from visiongpt.evaluation.utils import tokenize, get_ngrams, parse_input_to_strings

logger = logging.getLogger(__name__)


def compute_bleu(
    predictions: List[str], references: List[str]
) -> Dict[str, float]:
    """Compute BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores.

    Args:
        predictions: List of candidate/predicted strings.
        references: List of reference/ground-truth strings.

    Returns:
        A dictionary containing "BLEU-1", "BLEU-2", "BLEU-3", and "BLEU-4" scores.
        All scores are float values in the range [0.0, 1.0].
    """
    preds_parsed = parse_input_to_strings(predictions)
    refs_parsed = parse_input_to_strings(references)

    if not preds_parsed or not refs_parsed:
        return {"BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0}

    if len(preds_parsed) != len(refs_parsed):
        raise ValueError(
            f"Predictions (len={len(preds_parsed)}) and references (len={len(refs_parsed)}) "
            "must have the same length."
        )

    # Tokenize inputs
    tokenized_preds = [tokenize(p) for p in preds_parsed]
    tokenized_refs = [tokenize(r) for r in refs_parsed]

    # Calculate total lengths for Brevity Penalty (BP)
    # c: total length of candidate corpus
    # r: total length of reference corpus
    c = sum(len(p) for p in tokenized_preds)
    r = sum(len(r) for r in tokenized_refs)

    # If the candidate corpus is empty, BLEU is 0
    if c == 0:
        return {"BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0}

    # Compute brevity penalty (BP)
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (r / c))

    precisions: List[float] = []

    # Calculate modified n-gram precisions for n = 1 to 4
    for n in range(1, 5):
        total_pred_ngrams = 0
        total_matched_ngrams = 0

        for tokens_pred, tokens_ref in zip(tokenized_preds, tokenized_refs):
            pred_ngrams = get_ngrams(tokens_pred, n)
            ref_ngrams = get_ngrams(tokens_ref, n)

            for ngram, count in pred_ngrams.items():
                max_ref_count = ref_ngrams.get(ngram, 0)
                total_matched_ngrams += min(count, max_ref_count)
                total_pred_ngrams += count

        if total_pred_ngrams == 0:
            # If there are no n-grams of size n in the whole prediction corpus
            precisions.append(0.0)
        else:
            if total_matched_ngrams == 0:
                # Smoothing: add a small epsilon to the numerator to avoid zero precision
                # and prevent log(0) in geometric mean.
                p_n = 0.1 / total_pred_ngrams
            else:
                p_n = total_matched_ngrams / total_pred_ngrams
            precisions.append(p_n)

    # Compute BLEU-1 to BLEU-4
    scores: Dict[str, float] = {}
    for n in range(1, 5):
        # Determine geometric mean of precisions up to n
        sub_precisions = precisions[:n]
        if any(p <= 0.0 for p in sub_precisions):
            score = 0.0
        else:
            # Compute exponential of average log-precisions
            log_sum = sum(math.log(p) for p in sub_precisions)
            geo_mean = math.exp(log_sum / n)
            score = bp * geo_mean
        scores[f"BLEU-{n}"] = score

    return scores
