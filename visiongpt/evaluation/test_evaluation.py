"""Verification tests for the VisionGPT Evaluation Module.

Runs unit tests for each individual metric and the unified Evaluator class.
Supports testing lists of strings, numpy arrays, and TensorFlow string tensors.
"""

import sys
import unittest
import numpy as np
import logging

logger = logging.getLogger(__name__)
# Import metrics and utility
from visiongpt.evaluation.utils import preprocess_text, tokenize, parse_input_to_strings
from visiongpt.evaluation.bleu import compute_bleu
from visiongpt.evaluation.meteor import compute_meteor
from visiongpt.evaluation.rouge import compute_rouge_l
from visiongpt.evaluation.cider import CIDEr
from visiongpt.evaluation.vqa_accuracy import (
    compute_exact_match,
    compute_token_accuracy,
    compute_word_accuracy,
)
from visiongpt.evaluation.evaluator import (
    Evaluator,
    EmptyInputError,
    LengthMismatchError,
)

# Attempt to import tensorflow for loop compatibility tests
try:
    import tensorflow as tf
except ImportError:
    tf = None


class TestEvaluationModule(unittest.TestCase):
    """Test suite for VisionGPT Evaluation module components."""

    def test_utils_text_normalization(self):
        """Test text preprocessing and tokenization."""
        text = "Hello, World! This is a test... with punctuation, and UPPERCASE letters."
        expected_norm = "hello world this is a test with punctuation and uppercase letters"
        self.assertEqual(preprocess_text(text), expected_norm)

        expected_tokens = ["hello", "world", "this", "is", "a", "test", "with", "punctuation", "and", "uppercase", "letters"]
        self.assertEqual(tokenize(text), expected_tokens)

    def test_utils_empty_and_special_strings(self):
        """Test preprocessing on empty and None strings."""
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(preprocess_text(None), "")
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])

    def test_bleu_score(self):
        """Test BLEU score calculations."""
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        scores = compute_bleu(predictions, references)
        # Should be a perfect match (1.0)
        self.assertAlmostEqual(scores["BLEU-1"], 1.0)
        self.assertAlmostEqual(scores["BLEU-4"], 1.0)

        # Mismatched case (should apply brevity penalty / precision clipping)
        predictions = ["the cat"]
        references = ["the cat sat on the mat"]
        scores = compute_bleu(predictions, references)
        self.assertTrue(0.0 <= scores["BLEU-1"] <= 1.0)

        # Completely different sentences (should get close to 0)
        predictions = ["dogs bark at night"]
        references = ["the cat sat on the mat"]
        scores = compute_bleu(predictions, references)
        self.assertTrue(scores["BLEU-1"] == 0.0 or scores["BLEU-1"] < 0.1)

    def test_meteor_score(self):
        """Test METEOR score calculations."""
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        score = compute_meteor(predictions, references)
        # Identical sentences of length 6 has a chunk penalty of 0.5 * (1/6)^3
        expected_score = 1.0 * (1.0 - 0.5 * (1.0 / 6.0) ** 3)
        self.assertAlmostEqual(score, expected_score)

        # Test synonym/stem matches or near-identical alignments
        predictions = ["the cats sat on mats"]
        references = ["the cat sat on the mat"]
        score = compute_meteor(predictions, references)
        # Stemmer should align "cats" -> "cat" and "mats" -> "mat"
        self.assertTrue(score > 0.5)

    def test_rouge_l_score(self):
        """Test ROUGE-L score calculations."""
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        score = compute_rouge_l(predictions, references)
        self.assertAlmostEqual(score, 1.0)

        predictions = ["the cat is sitting on a mat"]
        references = ["the cat sat on the mat"]
        # LCS is "the cat on a mat" (length 5) or similar, ROUGE-L should be high
        score = compute_rouge_l(predictions, references)
        self.assertTrue(0.5 < score < 1.0)

    def test_cider_score(self):
        """Test CIDEr score calculations."""
        # Test perfect matches
        predictions = ["the cat sat on the mat", "a dog ran fast"]
        references = ["the cat sat on the mat", "a dog ran fast"]
        cider_scorer = CIDEr()
        score, scores = cider_scorer.compute_score(predictions, references)
        # Perfect consensus with default TF weights
        self.assertTrue(score > 9.0)  # Standard CIDEr range is 0 to 10
        self.assertEqual(len(scores), 2)

    def test_vqa_accuracy(self):
        """Test Exact Match, Token Accuracy, and Word Accuracy."""
        predictions = ["red and blue"]
        references = ["red and blue"]

        self.assertEqual(compute_exact_match(predictions, references), 1.0)
        self.assertEqual(compute_token_accuracy(predictions, references), 1.0)
        self.assertEqual(compute_word_accuracy(predictions, references), 1.0)

        # Mismatched answers
        predictions = ["red"]
        references = ["red and blue"]
        self.assertEqual(compute_exact_match(predictions, references), 0.0)
        # Token accuracy: precision = 1.0 (1/1), recall = 0.333 (1/3), F1 = 2 * 1 * 0.333 / 1.333 = 0.5
        self.assertAlmostEqual(compute_token_accuracy(predictions, references), 0.5)
        # Word accuracy: len_pred = 1, len_ref = 3, edit_dist = 2, accuracy = 1 - 2/3 = 0.333
        self.assertAlmostEqual(compute_word_accuracy(predictions, references), 1.0 / 3.0)

    def test_evaluator_dict_output(self):
        """Test standard outputs from Evaluator class."""
        predictions = ["the cat sat on the mat", "red and blue"]
        references = ["the cat sat on the mat", "red and blue"]

        evaluator = Evaluator()
        results = evaluator.evaluate(predictions, references)

        # Check key coverage
        expected_keys = {
            "BLEU-1",
            "BLEU-2",
            "BLEU-3",
            "BLEU-4",
            "METEOR",
            "ROUGE-L",
            "CIDEr",
            "Exact Match",
            "Token Accuracy",
            "Word Accuracy",
        }
        self.assertTrue(expected_keys.issubset(results.keys()))
        for key in expected_keys:
            self.assertTrue(0.0 <= results[key] <= 10.0, f"Value for {key} out of range: {results[key]}")

    def test_evaluator_error_handling(self):
        """Test exception raising for invalid inputs."""
        evaluator = Evaluator()

        # Length mismatch
        with self.assertRaises(LengthMismatchError):
            evaluator.evaluate(["cat"], ["cat", "dog"])

        # Empty inputs
        with self.assertRaises(EmptyInputError):
            evaluator.evaluate([], [])

    def test_tensorflow_compatibility(self):
        """Test compatibility with bytes, NumPy arrays, and TensorFlow string tensors."""
        evaluator = Evaluator()

        # Bytes inputs
        preds_bytes = [b"the cat sat", b"red"]
        refs_bytes = [b"the cat sat", b"red"]
        results = evaluator.evaluate(preds_bytes, refs_bytes)
        self.assertEqual(results["Exact Match"], 1.0)

        # NumPy arrays
        preds_np = np.array(["the cat sat", "red"])
        refs_np = np.array(["the cat sat", "red"])
        results = evaluator.evaluate(preds_np, refs_np)
        self.assertEqual(results["Exact Match"], 1.0)

        # TensorFlow tensors (if installed and executing eagerly)
        if tf is not None:
            # Enable eager execution check
            if tf.executing_eagerly():
                preds_tf = tf.constant(["the cat sat", "red"])
                refs_tf = tf.constant(["the cat sat", "red"])
                results = evaluator.evaluate(preds_tf, refs_tf)
                self.assertEqual(results["Exact Match"], 1.0)
            else:
                logger.info("TensorFlow eager execution is disabled; skipping TF Tensor test.")
        else:
            logger.info("TensorFlow is not installed in the test environment; skipping TF test.")


if __name__ == "__main__":
    unittest.main()
