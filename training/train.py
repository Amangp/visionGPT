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

from models.visiongpt import VisionGPT

from training.trainer import Trainer
from training.data_pipeline import DataPipeline
from training.coco_training_loader import COCOTrainingLoader

from preprocessing.text_preprocessor import TextPreprocessor


# =========================================================
# CONFIGURATION
# =========================================================

IMAGE_FOLDER = (
    "Dataset/COCO/train2017/train2017"
)

CAPTION_FILE = (
    "Dataset/COCO/"
    "annotations_trainval2017/"
    "annotations/"
    "captions_train2017.json"
)


TRAINING_LIMIT = 30000

BATCH_SIZE = 32

EPOCHS = 15

VALIDATION_SPLIT = 0.10

RANDOM_SEED = 42


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\n"
        "=========================================="
    )

    print(
        "Starting VisionGPT v2.4 Training"
    )

    print(
        "=========================================="
    )


    # =====================================================
    # 1. GPU CHECK
    # =====================================================

    print(
        "\n[1/9] Checking GPU..."
    )

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    print(
        "Available GPUs:",
        gpus
    )

    if gpus:

        print(
            "GPU training enabled"
        )

    else:

        print(
            "WARNING: No GPU detected"
        )


    # =====================================================
    # 2. LOAD COCO DATA
    # =====================================================

    print(
        "\n[2/9] Loading COCO training data..."
    )

    loader = COCOTrainingLoader(

        image_folder=IMAGE_FOLDER,

        caption_file=CAPTION_FILE,

        limit=TRAINING_LIMIT

    )

    pairs = loader.load()

    print(
        f"Loaded {len(pairs)} caption-image pairs"
    )


    # =====================================================
    # 3. SHUFFLE DATA
    # =====================================================

    print(
        "\n[3/9] Shuffling dataset..."
    )

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        pairs
    )


    # =====================================================
    # 4. SPLIT IMAGE PATHS AND CAPTIONS
    # =====================================================

    image_paths = [

        item[0]

        for item in pairs

    ]

    captions = [

        item[1]

        for item in pairs

    ]


    captions = [

        f"startseq {caption} endseq"

        for caption in captions

    ]


    # =====================================================
    # 5. TRAIN / VALIDATION SPLIT
    # =====================================================

    print(
        "\n[4/9] Creating train-validation split..."
    )

    validation_size = int(

        len(image_paths)
        *
        VALIDATION_SPLIT

    )


    train_image_paths = (
        image_paths[
            validation_size:
        ]
    )

    train_captions = (
        captions[
            validation_size:
        ]
    )


    val_image_paths = (
        image_paths[
            :validation_size
        ]
    )

    val_captions = (
        captions[
            :validation_size
        ]
    )


    print(
        "Training examples:",
        len(train_image_paths)
    )

    print(
        "Validation examples:",
        len(val_image_paths)
    )


    # =====================================================
    # 6. BUILD VOCABULARY
    # =====================================================

    print(
        "\n[5/9] Building vocabulary..."
    )

    text_processor = TextPreprocessor()


    # IMPORTANT:
    # Vocabulary is built only using training captions

    text_processor.build_vocabulary(
        train_captions
    )


    text_processor.save_vocab(
        "vocab.json"
    )


    vocab_size = len(

        text_processor
        .vectorizer
        .get_vocabulary()

    )


    print(
        "Vocabulary size:",
        vocab_size
    )

    print(
        "Vocabulary saved successfully"
    )


    # =====================================================
    # 7. TOKENIZE CAPTIONS
    # =====================================================

    print(
        "\n[6/9] Tokenizing captions..."
    )


    train_tokens = (
        text_processor.process(
            train_captions
        )
    )


    val_tokens = (
        text_processor.process(
            val_captions
        )
    )


    print(
        "Train token shape:",
        train_tokens.shape
    )

    print(
        "Validation token shape:",
        val_tokens.shape
    )


    # =====================================================
    # 8. CREATE STREAMING DATASETS
    # =====================================================

    print(
        "\n[7/9] Creating streaming datasets..."
    )


    pipeline = DataPipeline(

        batch_size=BATCH_SIZE

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


    # Test one batch

    for inputs, targets in train_dataset.take(1):

        images, decoder_inputs = inputs

        print(
            "Batch image shape:",
            images.shape
        )

        print(
            "Decoder input shape:",
            decoder_inputs.shape
        )

        print(
            "Target shape:",
            targets.shape
        )


    # =====================================================
    # 9. CREATE VISIONGPT
    # =====================================================

    print(
        "\n[8/9] Creating VisionGPT v2.4..."
    )


    model = VisionGPT(

        vocab_size=vocab_size

    )


    # Build model variables

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
            29
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
        "VisionGPT v2.4 architecture built successfully"
    )


    # =====================================================
    # TRAINER
    # =====================================================

    trainer = Trainer(

        model,

        learning_rate=0.001

    )


    trainer.compile()


    # =====================================================
    # CALLBACKS
    # =====================================================

    print(
        "\n[9/9] Preparing training callbacks..."
    )


    os.makedirs(

        "checkpoints",

        exist_ok=True

    )


    version = datetime.datetime.now().strftime(

        "%Y_%m_%d_%H_%M"

    )


    checkpoint_path = (

        "checkpoints/"

        f"visiongpt_v2_4_best_{version}.weights.h5"

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

    print(
        "\n"
        "=========================================="
    )

    print(
        "VISIONGPT v2.4 TRAINING STARTED"
    )

    print(
        "=========================================="
    )


    print(
        "Training pairs:",
        len(train_image_paths)
    )

    print(
        "Validation pairs:",
        len(val_image_paths)
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Maximum epochs:",
        EPOCHS
    )

    print(
        "Checkpoint:",
        checkpoint_path
    )


    history = model.fit(

        train_dataset,

        validation_data=val_dataset,

        epochs=EPOCHS,

        callbacks=callbacks

    )


    # =====================================================
    # TRAINING COMPLETE
    # =====================================================

    print(
        "\n"
        "=========================================="
    )

    print(
        "VISIONGPT v2.4 TRAINING COMPLETED"
    )

    print(
        "=========================================="
    )


    print(
        "Best weights saved at:"
    )

    print(
        checkpoint_path
    )


    print(
        "\nVisionGPT v2.4 trained successfully 🚀"
    )


if __name__ == "__main__":

    main()