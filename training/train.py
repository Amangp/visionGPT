import sys
import os
import datetime

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

from training.coco_training_loader import COCOTrainingLoader

from training.data_pipeline import DataPipeline


from preprocessing.image_preprocessor import ImagePreprocessor

from preprocessing.text_preprocessor import TextPreprocessor


# =====================================================
# CONFIGURATION
# =====================================================


VISIONGPT_VERSION = "v2"

TRAINING_LIMIT = 5000

BATCH_SIZE = 8

EPOCHS = 5


def main():


    print(
        "\n"
        "========================================"
    )

    print(
        f"Starting VisionGPT {VISIONGPT_VERSION} Training"
    )

    print(
        "========================================"
        "\n"
    )


    # =================================================
    # 1. LOAD COCO DATA
    # =================================================


    print(
        "[1/7] Loading COCO training data..."
    )


    loader = COCOTrainingLoader(

        image_folder=(
            "Dataset/COCO/train2017/train2017"
        ),

        caption_file=(
            "Dataset/COCO/"
            "annotations_trainval2017/"
            "annotations/"
            "captions_train2017.json"
        ),

        limit=TRAINING_LIMIT

    )


    pairs = loader.load()


    print(
        f"Loaded {len(pairs)} training examples"
    )


    image_paths = [

        item[0]

        for item in pairs

    ]


    captions = [

        item[1]

        for item in pairs

    ]


    # =================================================
    # 2. INITIALIZE PREPROCESSORS
    # =================================================


    print(
        "\n[2/7] Initializing preprocessors..."
    )


    image_processor = ImagePreprocessor()


    text_processor = TextPreprocessor()


    # =================================================
    # 3. BUILD VOCABULARY
    # =================================================


    print(
        "\n[3/7] Building vocabulary..."
    )


    captions = [

        f"startseq {caption} endseq"

        for caption in captions

    ]


    text_processor.build_vocabulary(
        captions
    )


    text_processor.save_vocab(
        "vocab.json"
    )


    vocabulary = (
        text_processor
        .vectorizer
        .get_vocabulary()
    )


    print(
        "Vocabulary saved successfully"
    )


    print(
        f"Vocabulary size: {len(vocabulary)}"
    )


    # =================================================
    # 4. PREPROCESS IMAGES
    # =================================================


    print(
        "\n[4/7] Processing images..."
    )


    processed_images = []


    total_images = len(
        image_paths
    )


    for index, path in enumerate(
        image_paths
    ):


        image = (
            image_processor.process(
                path
            )
        )


        processed_images.append(
            image
        )


        if (
            (index + 1) % 500 == 0
            or
            index + 1 == total_images
        ):


            print(

                f"Processed "
                f"{index + 1}/"
                f"{total_images} images"

            )


    images = tf.stack(
        processed_images
    )


    print(
        "Image tensor shape:",
        images.shape
    )


    # =================================================
    # 5. PROCESS CAPTIONS
    # =================================================


    print(
        "\n[5/7] Processing captions..."
    )


    text_tokens = (
        text_processor.process(
            captions
        )
    )


    print(
        "Caption tensor shape:",
        text_tokens.shape
    )


    # =================================================
    # 6. CREATE AUTOREGRESSIVE DATASET
    # =================================================


    print(
        "\n[6/7] Creating autoregressive dataset..."
    )


    pipeline = DataPipeline(

        batch_size=BATCH_SIZE,

        shuffle_buffer=1000

    )


    dataset = pipeline.create(

        images,

        text_tokens

    )


    # -------------------------------------------------
    # VERIFY DATASET SHAPES
    # -------------------------------------------------


    for inputs, targets in dataset.take(1):


        batch_images, decoder_inputs = inputs


        print(
            "Batch image shape:",
            batch_images.shape
        )


        print(
            "Decoder input shape:",
            decoder_inputs.shape
        )


        print(
            "Target shape:",
            targets.shape
        )


    # =================================================
    # 7. CREATE VISIONGPT V2
    # =================================================


    print(
        "\n[7/7] Creating VisionGPT v2..."
    )


    vocab_size = len(
    text_processor.vectorizer.get_vocabulary()
    )

    print(
        f"Vocabulary size: {vocab_size}"
    )

    model = VisionGPT(
        vocab_size=vocab_size
    )


    # -------------------------------------------------
    # BUILD MODEL
    # -------------------------------------------------


    dummy_image = tf.random.normal(

        (
            1,
            224,
            224,
            3
        )

    )


    dummy_text = tf.ones(

        (
            1,
            29
        ),

        dtype=tf.int32

    )


    dummy_output = model(

        (
            dummy_image,
            dummy_text
        ),

        training=False

    )


    print(
        "VisionGPT output shape:",
        dummy_output.shape
    )


    print(
        "\nVisionGPT v2 architecture built successfully"
    )


    # =================================================
    # CREATE TRAINER
    # =================================================


    trainer = Trainer(
        model
    )


    trainer.compile()


    # =================================================
    # TRAIN MODEL
    # =================================================


    print(
        "\n"
        "========================================"
    )


    print(
        "VISIONGPT V2 TRAINING STARTED"
    )


    print(
        "========================================"
        "\n"
    )


    history = trainer.train(

        dataset,

        epochs=EPOCHS

    )


    # =================================================
    # SAVE VERSIONED CHECKPOINT
    # =================================================


    os.makedirs(

        "checkpoints",

        exist_ok=True

    )


    timestamp = (

        datetime
        .datetime
        .now()
        .strftime(
            "%Y_%m_%d_%H_%M"
        )

    )


    save_path = (

        "checkpoints/"

        f"visiongpt_{VISIONGPT_VERSION}_"

        f"{timestamp}.weights.h5"

    )


    model.save_weights(
        save_path
    )


    print(
        "\n"
        "========================================"
    )


    print(
        "TRAINING COMPLETED"
    )


    print(
        "========================================"
    )


    print(
        f"VisionGPT Version: "
        f"{VISIONGPT_VERSION}"
    )


    print(
        f"Training Examples: "
        f"{TRAINING_LIMIT}"
    )


    print(
        f"Epochs: "
        f"{EPOCHS}"
    )


    print(
        f"Batch Size: "
        f"{BATCH_SIZE}"
    )


    print(
        f"Weights saved at: "
        f"{save_path}"
    )


    print(
        "\nVisionGPT v2 saved successfully 🚀"
    )


if __name__ == "__main__":

    main()