"""
Real validation script — run this AFTER training, instead of eyeballing
a handful of manually-picked test_images/.

Why this matters: test_images/ (tree, dinosaur, person, car, dog) are
out-of-pipeline photos you picked by hand. Some (dinosaur, tree) aren't
even in COCO's 80 object categories, so a model can look "wrong" on them
while actually being fine. This script instead samples real held-out
validation examples (same distribution as training, zero overlap with
training images per your own image-level split) and scores generated
captions against the REAL ground-truth captions using BLEU/METEOR/
ROUGE-L/CIDEr/Token-Accuracy — the same metrics used to evaluate real
captioning models.

Run from your project root:
    python -m validate_on_heldout
"""

import random
import sys

sys.path.append(".")

from models.visiongpt import VisionGPT
from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.image_preprocessor import ImagePreprocessor
from dataset_tools.coco_adapter import COCOAdapter
from dataset_tools.textvqa_adapter import TextVQAAdapter
from training.image_splitter import ImageLevelSplitter
from evaluation.evaluator import Evaluator

import tensorflow as tf

# =====================================================
# CONFIG - match train.py
# =====================================================

COCO_ANNOTATION_PATH = "Dataset/COCO/annotations/captions_train2017.json"
COCO_IMAGE_DIRECTORY = "Dataset/COCO/train2017"
TEXTVQA_ANNOTATION_PATH = "future_datasets/textvqa/annotations/TextVQA_0.5.1_train.json"
TEXTVQA_IMAGE_DIRECTORY = "future_datasets/textvqa/train_images/train_images"

COCO_LIMIT = 100000
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42
VOCAB_PATH = "vocab_v4.json"

NUM_SAMPLES = 30  # how many held-out examples to spot-check

import glob
import os

WEIGHT_FILES = glob.glob("checkpoints/visiongpt_v4_phase2_*.weights.h5")
if not WEIGHT_FILES:
    WEIGHT_FILES = glob.glob("checkpoints/visiongpt_v4_*_*.weights.h5")
MODEL_PATH = max(WEIGHT_FILES, key=os.path.getmtime)
print("Using checkpoint:", MODEL_PATH)


def main():
    random.seed(RANDOM_SEED)

    # Rebuild the exact same val split train.py produces
    coco_samples = COCOAdapter(
        annotation_path=COCO_ANNOTATION_PATH,
        image_directory=COCO_IMAGE_DIRECTORY,
    ).load(limit=COCO_LIMIT)
    textvqa_samples = TextVQAAdapter(
        annotation_path=TEXTVQA_ANNOTATION_PATH,
        image_directory=TEXTVQA_IMAGE_DIRECTORY,
    ).load()

    def split(samples):
        pairs = [(s.image_path, s) for s in samples]
        splitter = ImageLevelSplitter(
            validation_split=VALIDATION_SPLIT, random_seed=RANDOM_SEED
        )
        _, val_pairs = splitter.split(pairs)
        return [s for _, s in val_pairs]

    val_samples = split(coco_samples) + split(textvqa_samples)
    random.shuffle(val_samples)
    val_samples = val_samples[:NUM_SAMPLES]

    print(f"Evaluating on {len(val_samples)} held-out validation samples "
          f"(zero overlap with training images)\n")

    # Load model via the same path inference.py uses
    from inference import VisionGPTInference

    bot = VisionGPTInference(model_path=MODEL_PATH, vocab_path=VOCAB_PATH)

    predictions, references = [], []

    for i, sample in enumerate(val_samples, 1):
        try:
            caption = bot.generate(sample.image_path, method="beam")
        except Exception as e:
            print(f"[{i}] ERROR on {sample.image_path}: {e}")
            continue

        predictions.append(caption)
        references.append(str(sample.answer))

        print(f"[{i}] {sample.image_path}")
        print(f"    ground truth: {sample.answer}")
        print(f"    predicted:    {caption}\n")

    print("=" * 50)
    print("METRICS (held-out validation set)")
    print("=" * 50)

    evaluator = Evaluator()
    scores = evaluator.evaluate(predictions, references)
    for name, value in scores.items():
        print(f"{name:15s}: {value:.4f}")


if __name__ == "__main__":
    main()
