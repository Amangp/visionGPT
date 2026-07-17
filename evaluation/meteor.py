"""METEOR (Metric for Evaluation of Translation with Explicit ORdering) implementation.

This module provides a pure-Python implementation of the METEOR metric.
It aligns unigrams between prediction and reference using exact matches
and stemming fallbacks, calculates contiguous chunk penalties, and computes
the final METEOR score.
"""

import logging
from typing import List, Tuple

from visiongpt.evaluation.utils import parse_input_to_strings, tokenize

logger = logging.getLogger(__name__)

# Try to import NLTK's PorterStemmer if available.
# Otherwise, use a simple rule-based English stemmer fallback to keep the code
# self-contained and run-time robust in environments without NLTK data.
try:
    from nltk.stem.porter import PorterStemmer

    _stemmer = PorterStemmer()

    def stem(word: str) -> str:
        """Stem a word using NLTK Porter Stemmer."""
        return _stemmer.stem(word)

except ImportError:
    logger.info("NLTK not found. Using custom rule-based fallback stemmer.")

    def stem(word: str) -> str:
        """Stem a word using a simple rule-based fallback stemmer.

        Strips common English suffixes for basic morphological normalization.
        """
        w = word.lower()
        if len(w) <= 3:
            return w
        if w.endswith("ing"):
            return w[:-3]
        if w.endswith("ed"):
            return w[:-2]
        if w.endswith("es"):
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        if w.endswith("ly"):
            return w[:-2]
        return w


def align_words(cand_words: List[str], ref_words: List[str]) -> List[Tuple[int, int]]:
    """Align words in candidate and reference using exact and stem matching.

    Args:
        cand_words: List of words from the candidate sentence.
        ref_words: List of words from the reference sentence.

    Returns:
        A list of aligned index pairs (cand_idx, ref_idx) sorted by cand_idx.
    """
    matched_pairs: List[Tuple[int, int]] = []
    cand_matched = set()
    ref_matched = set()

    # Phase 1: Exact matching
    for i, c_word in enumerate(cand_words):
        for j, r_word in enumerate(ref_words):
            if j in ref_matched:
                continue
            if c_word == r_word:
                matched_pairs.append((i, j))
                cand_matched.add(i)
                ref_matched.add(j)
                break

    # Phase 2: Stem-based matching
    for i, c_word in enumerate(cand_words):
        if i in cand_matched:
            continue
        c_stem = stem(c_word)
        for j, r_word in enumerate(ref_words):
            if j in ref_matched:
                continue
            if c_stem == stem(r_word):
                matched_pairs.append((i, j))
                cand_matched.add(i)
                ref_matched.add(j)
                break

    return sorted(matched_pairs, key=lambda x: x[0])


def count_chunks(matched_pairs: List[Tuple[int, int]]) -> int:
    """Count the number of contiguous chunks in the alignment.

    A chunk is a set of matched unigrams that are adjacent and in the same order
    in both candidate and reference sentences.

    Args:
        matched_pairs: List of aligned index pairs sorted by candidate index.

    Returns:
        The number of contiguous matched chunks.
    """
    if not matched_pairs:
        return 0

    chunks = 1
    for k in range(len(matched_pairs) - 1):
        i_curr, j_curr = matched_pairs[k]
        i_next, j_next = matched_pairs[k + 1]

        # Contiguous if candidate indices increase by 1 and reference indices increase by 1
        if i_next == i_curr + 1 and j_next == j_curr + 1:
            continue
        else:
            chunks += 1

    return chunks


def compute_pair_meteor(cand_text: str, ref_text: str) -> float:
    """Calculate the METEOR score for a single candidate and reference pair.

    Args:
        cand_text: Candidate sentence.
        ref_text: Reference sentence.

    Returns:
        The METEOR score as a float between 0.0 and 1.0.
    """
    cand_tokens = tokenize(cand_text)
    ref_tokens = tokenize(ref_text)

    if not cand_tokens or not ref_tokens:
        return 0.0

    matched_pairs = align_words(cand_tokens, ref_tokens)
    m = len(matched_pairs)

    if m == 0:
        return 0.0

    # Unigram precision & recall
    precision = m / len(cand_tokens)
    recall = m / len(ref_tokens)

    # F-mean (harmonic mean weighting recall 9 times more than precision)
    f_mean = (10.0 * precision * recall) / (recall + 9.0 * precision)

    # Chunks penalty
    chunks = count_chunks(matched_pairs)
    penalty = 0.5 * ((chunks / m) ** 3)

    # Final score
    score = f_mean * (1.0 - penalty)
    return score


def compute_meteor(predictions: List[str], references: List[str]) -> float:
    """Compute the average METEOR score for a batch of predictions and references.

    Args:
        predictions: List of candidate/predicted strings.
        references: List of reference/ground-truth strings.

    Returns:
        The average METEOR score over the batch.
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
        total_score += compute_pair_meteor(pred, ref)

    return total_score / len(preds_parsed)
