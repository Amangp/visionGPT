import sys
import os
import datetime
import random

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import tensorflow as tf
from tensorflow.keras import mixed_precision
gpus = tf.config.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("Available GPUs:", gpus)
mixed_precision.set_global_policy(
    "mixed_float16"
)

print(
    "Mixed precision enabled:",
    mixed_precision.global_policy()
)
from models.visiongpt import VisionGPT
from training.trainer import Trainer
from training.data_pipeline import DataPipeline
from training.image_splitter import ImageLevelSplitter
from preprocessing.text_preprocessor import TextPreprocessor
from dataset_tools.coco_adapter import COCOAdapter
from dataset_tools.textvqa_adapter import TextVQAAdapter


# =========================================================
# CONFIGURATION
# =========================================================

COCO_ANNOTATION_PATH = (
    "Dataset/COCO/"
    "annotations_trainval2017/"
    "annotations/"
    "captions_train2017.json"
)

COCO_IMAGE_DIRECTORY = (
    "Dataset/COCO/train2017/train2017"
)

TEXTVQA_ANNOTATION_PATH = (
    "future_datasets/textvqa/"
    "annotations/"
    "TextVQA_0.5.1_train.json"
)

TEXTVQA_IMAGE_DIRECTORY = (
    "future_datasets/textvqa/"
    "train_images/"
    "train_images"
)

COCO_LIMIT = 100000

BATCH_SIZE = 16
EPOCHS = 25
VALIDATION_SPLIT = 0.10
RANDOM_SEED = 42

VOCAB_SIZE = 10000
MAX_LENGTH = 50

CONTEXT_DROPOUT_RATE = 0.10
LEARNING_RATE = 0.0001


# =========================================================
# IMAGE-LEVEL SAMPLE SPLIT
# =========================================================

def split_samples(samples):

    pairs = [
        (
            sample.image_path,
            sample
        )
        for sample in samples
    ]

    splitter = ImageLevelSplitter(
        validation_split=VALIDATION_SPLIT,
        random_seed=RANDOM_SEED
    )

    train_pairs, val_pairs = splitter.split(pairs)

    train_samples = [
        sample
        for image_path, sample in train_pairs
    ]

    val_samples = [
        sample
        for image_path, sample in val_pairs
    ]

    return train_samples, val_samples


# =========================================================
# MAIN
# =========================================================

def main():

    random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("\n==========================================")
    print("Starting VisionGPT v4 Training")
    print("==========================================")

    # =====================================================
    # 1. GPU CHECK
    # =====================================================

    print("\n[1/10] Checking GPU...")

    current_gpus = tf.config.list_physical_devices("GPU")

    print("Available GPUs:", current_gpus)

    if current_gpus:
        print("GPU training enabled")
    else:
        print("WARNING: No GPU detected")

    # =====================================================
    # 2. CHECK DATASETS
    # =====================================================

    print("\n[2/10] Checking datasets...")

    required_paths = [
        COCO_ANNOTATION_PATH,
        COCO_IMAGE_DIRECTORY,
        TEXTVQA_ANNOTATION_PATH,
        TEXTVQA_IMAGE_DIRECTORY
    ]

    for path in required_paths:

        if not os.path.exists(path):

            raise FileNotFoundError(
                "Dataset path not found: "
                + path
            )

    print("COCO and TextVQA paths verified")

    # =====================================================
    # 3. LOAD BALANCED MULTIMODAL DATA
    # =====================================================

    print("\n[3/10] Loading balanced multimodal data...")

    coco_adapter = COCOAdapter(
        annotation_path=COCO_ANNOTATION_PATH,
        image_directory=COCO_IMAGE_DIRECTORY
    )

    textvqa_adapter = TextVQAAdapter(
        annotation_path=TEXTVQA_ANNOTATION_PATH,
        image_directory=TEXTVQA_IMAGE_DIRECTORY
    )

    coco_samples = coco_adapter.load(
        limit=COCO_LIMIT
    )
    textvqa_samples = textvqa_adapter.load()
    print(type(coco_samples))
    print(coco_samples[:5])
    print("Selected COCO samples:", len(coco_samples))
    print("TextVQA samples:", len(textvqa_samples)) 

    print("Selected COCO samples:", len(coco_samples))
    print("Selected TextVQA samples:", len(textvqa_samples))

    # =====================================================
    # 4. IMAGE-LEVEL SPLIT PER DATASET
    # =====================================================

    print("\n[4/10] Creating image-level splits...")

    coco_train, coco_val = split_samples(coco_samples)

    textvqa_train, textvqa_val = split_samples(
        textvqa_samples
    )

    train_samples = coco_train + textvqa_train
    val_samples = coco_val + textvqa_val

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    train_image_set = {
        sample.image_path
        for sample in train_samples
    }

    val_image_set = {
        sample.image_path
        for sample in val_samples
    }

    image_overlap = train_image_set & val_image_set

    print("Training samples:", len(train_samples))
    print("Validation samples:", len(val_samples))
    print("Unique training images:", len(train_image_set))
    print("Unique validation images:", len(val_image_set))
    print("Image overlap:", len(image_overlap))

    if image_overlap:

        raise RuntimeError(
            "Training and validation image overlap detected"
        )

    # =====================================================
    # 5. BUILD V4 TRAINING SEQUENCES
    # =====================================================

    print("\n[5/10] Building v4 training sequences...")

    text_processor = TextPreprocessor(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH
    )

    train_texts = text_processor.build_dataset_texts(
        train_samples
    )

    val_texts = text_processor.build_dataset_texts(
        val_samples
    )

    train_image_paths = [
        sample.image_path
        for sample in train_samples
    ]

    val_image_paths = [
        sample.image_path
        for sample in val_samples
    ]

    print("Example training sequence:")
    print(train_texts[0])

    # =====================================================
    # 6. BUILD V4 VOCABULARY
    # =====================================================

    print("\n[6/10] Building v4 vocabulary...")

    text_processor.build_vocabulary(train_texts)

    text_processor.save_vocab("vocab_v4.json")

    vocabulary = (
        text_processor
        .vectorizer
        .get_vocabulary()
    )

    vocab_size = len(vocabulary)

    word_to_index = {
        word: index
        for index, word in enumerate(vocabulary)
    }

    mask_token_id = word_to_index.get("[UNK]", 1)

    special_token_ids = (
        text_processor.get_special_token_ids()
    )

    start_token_id = special_token_ids[
        "start_token_id"
    ]

    end_token_id = special_token_ids[
        "end_token_id"
    ]

    answer_token_id = special_token_ids[
        "answer_token_id"
    ]

    task_caption_token_id = special_token_ids[
        "task_caption_token_id"
    ]

    task_ocr_token_id = special_token_ids[
        "task_ocr_token_id"
    ]

    print("Vocabulary size:", vocab_size)
    print("Mask token ID:", mask_token_id)
    print("Start token ID:", start_token_id)
    print("End token ID:", end_token_id)
    print("Answer token ID:", answer_token_id)
    print("Task caption token ID:", task_caption_token_id)
    print("Task OCR token ID:", task_ocr_token_id)

    # =====================================================
    # 7. TOKENIZE
    # =====================================================

    print("\n[7/10] Tokenizing v4 sequences...")

    train_tokens = text_processor.process(train_texts)
    val_tokens = text_processor.process(val_texts)

    print("Train token shape:", train_tokens.shape)
    print("Validation token shape:", val_tokens.shape)

    # =====================================================
    # 8. CREATE RAW-IMAGE DATASETS
    # =====================================================

    print("\n[8/10] Creating raw-image datasets...")

    pipeline = DataPipeline(
        batch_size=BATCH_SIZE,
        answer_token_id=answer_token_id
    )

    train_dataset = pipeline.create(
        train_image_paths,
        train_tokens,
        training=True
    )

    val_dataset = pipeline.create(
        val_image_paths,
        val_tokens,
        training=False
    )

    for (
        inputs,
        targets,
        target_mask
    ) in train_dataset.take(1):

        images, decoder_inputs = inputs

        print("\nVisionGPT v4 batch verification")
        print("Image shape:", images.shape)
        print(
            "Decoder input shape:",
            decoder_inputs.shape
        )
        print("Target shape:", targets.shape)
        print(
            "Target mask shape:",
            target_mask.shape
        )
        print(
            "Supervised tokens:",
            float(
                tf.reduce_sum(target_mask).numpy()
            )
        )

    # =====================================================
    # 9. CREATE VISIONGPT V4
    # =====================================================

    print("\n[9/10] Creating VisionGPT v4...")

    model = VisionGPT(
        vocab_size=vocab_size,
        context_dropout_rate=CONTEXT_DROPOUT_RATE,
        mask_token_id=mask_token_id,
        start_token_id=start_token_id,
        end_token_id=end_token_id,
        answer_token_id=answer_token_id,
        task_caption_token_id=task_caption_token_id,
        task_ocr_token_id=task_ocr_token_id
    )

    dummy_image = tf.zeros(
        (
            1,
            224,
            224,
            3
        ),
        dtype=tf.float32
    )

    dummy_text = tf.ones(
        (
            1,
            MAX_LENGTH - 1
        ),
        dtype=tf.int64
    )

    output = model(
        (
            dummy_image,
            dummy_text
        ),
        training=False
    )

    print(
        "VisionGPT output shape:",
        output.shape
    )

    print(
        "VisionGPT v4 architecture built successfully"
    )

    print(
        "\nVisionGPT v4 uses a new token vocabulary."
    )

    print(
        "v3.2 decoder weights are not loaded because "
        "v4 token IDs have new meanings."
    )

    print(
        "EfficientNet remains ImageNet initialized and frozen."
    )

    # =====================================================
    # 10. COMPILE
    # =====================================================

    print("\n[10/10] Compiling VisionGPT v4...")

    trainer = Trainer(
        model,
        learning_rate=LEARNING_RATE
    )

    trainer.compile()

    # =====================================================
    # CALLBACKS
    # =====================================================

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    version = datetime.datetime.now().strftime(
        "%Y_%m_%d_%H_%M"
    )

    checkpoint_path = (
        "checkpoints/"
        f"visiongpt_v4_best_"
        f"{version}.weights.h5"
    )

    checkpoint_callback = (
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=1
        )
    )

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    )

    reduce_lr = (
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    )

    callbacks = [
        checkpoint_callback,
        early_stopping,
        reduce_lr
    ]

    # =====================================================
    # TRAIN
    # =====================================================

    print("\n==========================================")
    print("VISIONGPT v4 TRAINING STARTED")
    print("==========================================")

    print("COCO training samples:", len(coco_train))
    print(
        "TextVQA training samples:",
        len(textvqa_train)
    )
    print(
        "Total training samples:",
        len(train_samples)
    )
    print(
        "Total validation samples:",
        len(val_samples)
    )
    print("Image overlap:", len(image_overlap))
    print(
        "Context dropout rate:",
        CONTEXT_DROPOUT_RATE
    )
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", EPOCHS)
    print("Checkpoint:", checkpoint_path)

    history = trainer.train(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # =====================================================
    # COMPLETE
    # =====================================================

    print("\n==========================================")
    print("VISIONGPT v4 TRAINING COMPLETED")
    print("==========================================")

    print("Best weights saved at:")
    print(checkpoint_path)

    print(
        "\nVisionGPT v4 trained successfully 🚀"
    )


if __name__ == "__main__":
    main()
