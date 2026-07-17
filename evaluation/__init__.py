"""VisionGPT Evaluation Module.

This package provides metrics and a unified Evaluator class for evaluating
image captioning, machine translation, and visual question answering models.
"""

from visiongpt.evaluation.evaluator import Evaluator
from visiongpt.evaluation.bleu import compute_bleu
from visiongpt.evaluation.meteor import compute_meteor
from visiongpt.evaluation.rouge import compute_rouge_l
from visiongpt.evaluation.cider import CIDEr
from visiongpt.evaluation.vqa_accuracy import (
    compute_exact_match,
    compute_token_accuracy,
    compute_word_accuracy,
)

__all__ = [
    "Evaluator",
    "compute_bleu",
    "compute_meteor",
    "compute_rouge_l",
    "CIDEr",
    "compute_exact_match",
    "compute_token_accuracy",
    "compute_word_accuracy",
]
