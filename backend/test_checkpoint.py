from backend.inference.checkpoint_loader import (
    CheckpointLoader
)

from models.visiongpt import VisionGPT


# =========================================================
# CONFIGURATION
# =========================================================

CHECKPOINT_PATH = (
    "checkpoints/"
    "visiongpt_v4_best_2026_07_15_10_42.weights.h5"
)


VOCAB_SIZE = 10000

MASK_TOKEN_ID = 1

START_TOKEN_ID = 4

END_TOKEN_ID = 5


# =========================================================
# CREATE MODEL
# =========================================================

print(
    "\nCreating VisionGPT v4..."
)


model = VisionGPT(

    vocab_size=VOCAB_SIZE,

    context_dropout_rate=0.10,

    mask_token_id=MASK_TOKEN_ID,

    start_token_id=START_TOKEN_ID,

    end_token_id=END_TOKEN_ID,

    answer_token_id=6,

    task_caption_token_id=7,

    task_ocr_token_id=14

)


# =========================================================
# CREATE CHECKPOINT LOADER
# =========================================================

checkpoint_loader = CheckpointLoader(

    sequence_length=50

)


# =========================================================
# LOAD CHECKPOINT
# =========================================================

model = checkpoint_loader.load(

    model=model,

    checkpoint_path=CHECKPOINT_PATH

)


# =========================================================
# SUCCESS
# =========================================================

print(
    "\n=========================================="
)

print(
    "VISIONGPT v4 CHECKPOINT TEST SUCCESSFUL"
)

print(
    "=========================================="
)