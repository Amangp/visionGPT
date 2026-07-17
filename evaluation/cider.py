"""CIDEr (Consensus-based Image Description Evaluation) metric implementation.

This module provides a pure-Python, mathematically correct implementation of the
CIDEr metric. It computes term frequency-inverse document frequency (TF-IDF)
weights for n-grams (1 to 4), calculates cosine similarity between candidates
and references, and applies a Gaussian length penalty.
"""

import collections
import logging
import math
from typing import Any, Dict, List, Tuple, Union

from visiongpt.evaluation.utils import get_ngrams, parse_input_to_strings, tokenize

logger = logging.getLogger(__name__)


class CIDEr:
    """CIDEr evaluation metric class.

    This class computes the CIDEr consensus metric for image captioning.
    It supports both parallel lists of predictions and references, and
    dictionaries of image keys to caption lists (drop-in compatibility with
    coco-caption).
    """

    def __init__(self, sigma: float = 6.0):
        """Initialize the CIDEr scorer.

        Args:
            sigma: The length tolerance parameter for the Gaussian penalty.
        """
        self.sigma = sigma

    def compute_score(
        self,
        predictions: Union[List[str], Dict[Any, List[str]]],
        references: Union[List[str], Dict[Any, List[str]]],
    ) -> Tuple[float, List[float]]:
        """Compute the average CIDEr score and individual scores.

        Args:
            predictions: Candidate predictions. Can be a list of strings
                or a dictionary mapping image IDs to a list of strings.
            references: Reference captions. Can be a list of strings
                or a dictionary mapping image IDs to a list of strings.

        Returns:
            A tuple of (average_cider_score, list_of_individual_scores).
        """
        # Convert lists to the dictionary-based format if needed
        if isinstance(predictions, list) and isinstance(references, list):
            preds_parsed = parse_input_to_strings(predictions)
            refs_parsed = parse_input_to_strings(references)

            if len(preds_parsed) != len(refs_parsed):
                raise ValueError(
                    f"Predictions (len={len(preds_parsed)}) and references (len={len(refs_parsed)}) "
                    "must have the same length."
                )

            # Map index to a list of a single string
            preds_dict = {i: [pred] for i, pred in enumerate(preds_parsed)}
            refs_dict = {i: [ref] for i, ref in enumerate(refs_parsed)}
        elif isinstance(predictions, dict) and isinstance(references, dict):
            # Already in dict format
            preds_dict = predictions
            refs_dict = references
        else:
            raise TypeError(
                "Predictions and references must be both lists of strings or both dictionaries."
            )

        if not preds_dict or not refs_dict:
            return 0.0, []

        # Tokenize predictions and references
        preds_tokenized = {
            img_id: [tokenize(c) for c in cands] for img_id, cands in preds_dict.items()
        }
        refs_tokenized = {
            img_id: [tokenize(r) for r in refs] for img_id, refs in refs_dict.items()
        }

        # Calculate document frequency (DF) over reference corpus for each n-gram size (1 to 4)
        num_images = len(refs_tokenized)
        df_counts: Dict[int, Dict[Tuple[str, ...], int]] = {
            n: collections.defaultdict(int) for n in range(1, 5)
        }

        for img_id, refs in refs_tokenized.items():
            for n in range(1, 5):
                # An n-gram is counted at most once per image
                unique_ngrams_for_image = set()
                for ref_tokens in refs:
                    ref_ngrams = get_ngrams(ref_tokens, n)
                    unique_ngrams_for_image.update(ref_ngrams.keys())

                for ngram in unique_ngrams_for_image:
                    df_counts[n][ngram] += 1

        # Calculate IDF values
        # If num_images is 1, DF will be 1 for all reference n-grams. To avoid log(1) = 0,
        # we assign a default IDF of 1.0 so that cosine similarity is computed based on TF.
        idfs: Dict[int, Dict[Tuple[str, ...], float]] = {n: {} for n in range(1, 5)}
        for n in range(1, 5):
            for ngram, df in df_counts[n].items():
                if num_images == 1:
                    idfs[n][ngram] = 1.0
                else:
                    idfs[n][ngram] = math.log(float(num_images)) - math.log(float(df))

        individual_scores: List[float] = []

        # Compute CIDEr score for each image/prediction pair
        for img_id, pred_captions in preds_tokenized.items():
            # If no reference or no prediction, score is 0
            if img_id not in refs_tokenized or not pred_captions or not refs_tokenized[img_id]:
                individual_scores.append(0.0)
                continue

            # Standard CIDEr uses the first candidate caption
            pred_tokens = pred_captions[0]
            pred_len = len(pred_tokens)

            refs = refs_tokenized[img_id]

            cider_n_scores = []
            for n in range(1, 5):
                pred_ngrams = get_ngrams(pred_tokens, n)

                # Compute prediction TF-IDF vector and its L2 norm
                pred_tfidf = {}
                pred_norm_sq = 0.0
                for ngram, count in pred_ngrams.items():
                    idf = idfs[n].get(ngram, 0.0)
                    val = count * idf
                    pred_tfidf[ngram] = val
                    pred_norm_sq += val**2
                pred_norm = math.sqrt(pred_norm_sq)

                sim_sum = 0.0
                for ref_tokens in refs:
                    ref_len = len(ref_tokens)
                    ref_ngrams = get_ngrams(ref_tokens, n)

                    # Compute reference TF-IDF vector and its L2 norm
                    ref_tfidf = {}
                    ref_norm_sq = 0.0
                    for ngram, count in ref_ngrams.items():
                        idf = idfs[n].get(ngram, 0.0)
                        val = count * idf
                        ref_tfidf[ngram] = val
                        ref_norm_sq += val**2
                    ref_norm = math.sqrt(ref_norm_sq)

                    # Calculate cosine similarity
                    dot_product = 0.0
                    for ngram, val in pred_tfidf.items():
                        dot_product += val * ref_tfidf.get(ngram, 0.0)

                    if pred_norm > 0.0 and ref_norm > 0.0:
                        cos_sim = dot_product / (pred_norm * ref_norm)
                    else:
                        cos_sim = 0.0

                    # Calculate length penalty
                    # Penalty is e^(-(pred_len - ref_len)^2 / (2 * sigma^2))
                    len_diff = pred_len - ref_len
                    length_penalty = math.exp(-(len_diff**2) / (2.0 * (self.sigma**2)))

                    sim_sum += cos_sim * length_penalty

                # Average similarity across all references for this image
                avg_sim = sim_sum / len(refs)
                cider_n_scores.append(avg_sim)

            # Sum CIDEr_n scores weighted by 10 / 4 = 2.5
            # This scales the final CIDEr score to be in range [0, 10]
            img_cider_score = sum(cider_n_scores) * 2.5
            individual_scores.append(img_cider_score)

        avg_cider_score = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
        return avg_cider_score, individual_scores
