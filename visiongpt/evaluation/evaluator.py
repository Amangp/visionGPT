"""Unified Evaluator class for VisionGPT evaluation.

This module provides the main Evaluator class which computes BLEU, METEOR,
ROUGE-L, CIDEr, Exact Match, and Token Accuracy scores on batches of predictions
and references. It provides robust parsing, error handling, and structured logging.
"""

import logging
from typing import Dict, List, Any

from visiongpt.evaluation.bleu import compute_bleu
from visiongpt.evaluation.cider import CIDEr
from visiongpt.evaluation.meteor import compute_meteor
from visiongpt.evaluation.rouge import compute_rouge_l
from visiongpt.evaluation.utils import parse_input_to_strings
from visiongpt.evaluation.vqa_accuracy import (
    compute_exact_match,
    compute_token_accuracy,
    compute_word_accuracy,
)

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Base exception for all evaluation-related errors."""


class LengthMismatchError(EvaluationError, ValueError):
    """Raised when predictions and references lists have different sizes."""


class EmptyInputError(EvaluationError, ValueError):
    """Raised when the predictions or references inputs are empty."""


class Evaluator:
    """Evaluator class to calculate a suite of evaluation metrics for VisionGPT models.

    Computes BLEU-1 to 4, METEOR, ROUGE-L, CIDEr, Exact Match, and Token Accuracy.
    Input formats can be lists of strings, lists of bytes, NumPy arrays, or
    eager TensorFlow tensors.
    """

    def __init__(self, cider_sigma: float = 6.0, rouge_beta: float = 1.2):
        """Initialize the Evaluator with parameters for metrics.

        Args:
            cider_sigma: Length tolerance parameter for CIDEr Gaussian penalty.
            rouge_beta: Weighting parameter for ROUGE-L precision and recall.
        """
        self.cider_scorer = CIDEr(sigma=cider_sigma)
        self.rouge_beta = rouge_beta

    def evaluate(
        self, predictions: List[Any], references: List[Any]
    ) -> Dict[str, float]:
        """Perform batch evaluation of predictions against references.

        Parses inputs into standard python strings, checks shape compatibility,
        and computes all requested metrics.

        Args:
            predictions: List of model predictions. Can be a list of strings, bytes,
                NumPy array, or eager TensorFlow tensor.
            references: List of reference ground truths. Can be a list of strings,
                bytes, NumPy array, or eager TensorFlow tensor.

        Returns:
            A dictionary containing the calculated metrics:
            {
                "BLEU-1": ...,
                "BLEU-2": ...,
                "BLEU-3": ...,
                "BLEU-4": ...,
                "METEOR": ...,
                "ROUGE-L": ...,
                "CIDEr": ...,
                "Exact Match": ...,
                "Token Accuracy": ...,
                "Word Accuracy": ...
            }

        Raises:
            EmptyInputError: If predictions or references are empty.
            LengthMismatchError: If the length of predictions does not match references.
            EvaluationError: General evaluation failure.
        """
        logger.info("Starting batch evaluation.")

        try:
            preds_parsed = parse_input_to_strings(predictions)
            refs_parsed = parse_input_to_strings(references)
        except Exception as e:
            logger.error("Failed to parse inputs into string lists: %s", e)
            raise EvaluationError(f"Input parsing failed: {e}") from e

        if not preds_parsed or not refs_parsed:
            logger.error("Empty input predictions or references provided.")
            raise EmptyInputError("Predictions and references cannot be empty.")

        if len(preds_parsed) != len(refs_parsed):
            logger.error(
                "Length mismatch: predictions (len=%d) and references (len=%d).",
                len(preds_parsed),
                len(refs_parsed),
            )
            raise LengthMismatchError(
                f"Predictions (len={len(preds_parsed)}) and references (len={len(refs_parsed)}) "
                "must have the same length."
            )

        logger.info("Evaluating %d samples.", len(preds_parsed))

        results: Dict[str, float] = {}

        # 1. Compute BLEU metrics
        try:
            bleu_scores = compute_bleu(preds_parsed, refs_parsed)
            results.update(bleu_scores)
            logger.debug("Computed BLEU scores successfully.")
        except Exception as e:
            logger.exception("Failed to compute BLEU scores.")
            raise EvaluationError(f"BLEU computation failed: {e}") from e

        # 2. Compute METEOR
        try:
            results["METEOR"] = compute_meteor(preds_parsed, refs_parsed)
            logger.debug("Computed METEOR score successfully.")
        except Exception as e:
            logger.exception("Failed to compute METEOR score.")
            raise EvaluationError(f"METEOR computation failed: {e}") from e

        # 3. Compute ROUGE-L
        try:
            results["ROUGE-L"] = compute_rouge_l(
                preds_parsed, refs_parsed, beta=self.rouge_beta
            )
            logger.debug("Computed ROUGE-L score successfully.")
        except Exception as e:
            logger.exception("Failed to compute ROUGE-L score.")
            raise EvaluationError(f"ROUGE-L computation failed: {e}") from e

        # 4. Compute CIDEr
        try:
            cider_val, _ = self.cider_scorer.compute_score(
                preds_parsed, refs_parsed
            )
            results["CIDEr"] = cider_val
            logger.debug("Computed CIDEr score successfully.")
        except Exception as e:
            logger.exception("Failed to compute CIDEr score.")
            raise EvaluationError(f"CIDEr computation failed: {e}") from e

        # 5. Compute Exact Match
        try:
            results["Exact Match"] = compute_exact_match(
                preds_parsed, refs_parsed
            )
            logger.debug("Computed Exact Match successfully.")
        except Exception as e:
            logger.exception("Failed to compute Exact Match.")
            raise EvaluationError(f"Exact Match computation failed: {e}") from e

        # 6. Compute Token Accuracy
        try:
            results["Token Accuracy"] = compute_token_accuracy(
                preds_parsed, refs_parsed
            )
            logger.debug("Computed Token Accuracy successfully.")
        except Exception as e:
            logger.exception("Failed to compute Token Accuracy.")
            raise EvaluationError(
                f"Token Accuracy computation failed: {e}"
            ) from e

        # 7. Compute Word Accuracy
        try:
            results["Word Accuracy"] = compute_word_accuracy(
                preds_parsed, refs_parsed
            )
            logger.debug("Computed Word Accuracy successfully.")
        except Exception as e:
            logger.exception("Failed to compute Word Accuracy.")
            raise EvaluationError(
                f"Word Accuracy computation failed: {e}"
            ) from e

        logger.info("Evaluation completed successfully.")
        return results
