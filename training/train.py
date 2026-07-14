import sys
import os
import datetime
import shutil

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import tensorflow as tf
gpus = tf.config.list_physical_devices(
    "GPU"
)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(
        gpu,
        True
    )

print(
    "Available GPUs:",
    gpus
)

from models.visiongpt import VisionGPT

from training.trainer import Trainer
from training.data_pipeline import DataPipeline
from training.coco_training_loader import COCOTrainingLoader
from training.feature_cache import FeatureCache
from training.image_splitter import ImageLevelSplitter

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
BATCH_SIZE = 8
FEATURE_CACHE_BATCH_SIZE = 16
EPOCHS = 25
VALIDATION_SPLIT = 0.10
RANDOM_SEED = 42
FEATURE_CACHE_DIR = (
    "feature_cache"
)


DRIVE_BACKUP_DIR = (
    "/content/drive/MyDrive/"
    "VisionGPT/v2.6"
)


CONTEXT_DROPOUT_RATE = 0.10


# =========================================================
# DRIVE BACKUP CALLBACK
# =========================================================

class DriveCheckpointBackup(
    tf.keras.callbacks.Callback
):

    def __init__(
        self,
        checkpoint_path,
        vocab_path,
        drive_backup_dir
    ):

        super().__init__()

        self.checkpoint_path = checkpoint_path

        self.vocab_path = vocab_path

        self.drive_backup_dir = drive_backup_dir

        self.best_val_loss = float("inf")


    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        logs = logs or {}

        val_loss = logs.get(
            "val_loss"
        )

        if val_loss is None:

            return

        if val_loss >= self.best_val_loss:

            return

        self.best_val_loss = val_loss

        if not os.path.exists(
            self.checkpoint_path
        ):

            print(
                "\nCheckpoint not available "
                "for Drive backup yet"
            )

            return

        if not os.path.exists(
            "/content/drive/MyDrive"
        ):

            print(
                "\nGoogle Drive is not mounted. "
                "Skipping Drive backup."
            )

            return

        os.makedirs(

            self.drive_backup_dir,

            exist_ok=True

        )

        checkpoint_destination = os.path.join(

            self.drive_backup_dir,

            os.path.basename(
                self.checkpoint_path
            )

        )

        vocab_destination = os.path.join(

            self.drive_backup_dir,

            "vocab.json"

        )

        shutil.copy2(

            self.checkpoint_path,

            checkpoint_destination

        )

        shutil.copy2(

            self.vocab_path,

            vocab_destination

        )

        print(
            "\n=========================================="
        )

        print(
            "BEST V2.6 MODEL BACKED UP TO DRIVE"
        )

        print(
            "=========================================="
        )

        print(
            "Validation loss:",
            val_loss
        )

        print(
            "Checkpoint:",
            checkpoint_destination
        )

        print(
            "Vocabulary:",
            vocab_destination
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\n=========================================="
    )

    print(
        "Starting VisionGPT v2.6 Training"
    )

    print(
        "=========================================="
    )


    # =====================================================
    # 1. GPU CHECK
    # =====================================================

    print(
        "\n[1/11] Checking GPU..."
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
    # 2. CHECK DATASET
    # =====================================================

    print(
        "\n[2/11] Checking COCO dataset..."
    )

    if not os.path.exists(
        IMAGE_FOLDER
    ):

        raise FileNotFoundError(

            "COCO image folder not found: "
            + IMAGE_FOLDER

        )

    if not os.path.exists(
        CAPTION_FILE
    ):

        raise FileNotFoundError(

            "COCO caption file not found: "
            + CAPTION_FILE

        )

    print(
        "Image folder found"
    )

    print(
        "Caption file found"
    )


    # =====================================================
    # 3. LOAD COCO DATA
    # =====================================================

    print(
        "\n[3/11] Loading COCO training data..."
    )

    loader = COCOTrainingLoader(

        image_folder=IMAGE_FOLDER,

        caption_file=CAPTION_FILE,

    )

    pairs = loader.load()
    FULL_DATASET_SIZE = len(pairs)

    TRAINING_LIMIT = 100000

    pairs = pairs[:TRAINING_LIMIT]

    print(
        f"Full COCO pairs: {FULL_DATASET_SIZE}"
    )

    print(
        f"Selected training pairs: {len(pairs)}"
    )
    print(
        "Loaded caption-image pairs:",
        len(pairs)
    )


    # =====================================================
    # 4. IMAGE-LEVEL SPLIT
    # =====================================================

    print(
        "\n[4/11] Creating image-level split..."
    )

    splitter = ImageLevelSplitter(

        validation_split=VALIDATION_SPLIT,

        random_seed=RANDOM_SEED

    )

    train_pairs, val_pairs = splitter.split(
        pairs
    )

    train_image_set = {

        image_path

        for image_path, caption in train_pairs

    }

    val_image_set = {

        image_path

        for image_path, caption in val_pairs

    }

    image_overlap = (

        train_image_set

        &

        val_image_set

    )

    print(
        "\nSplit verification:"
    )

    print(
        "Unique training images:",
        len(train_image_set)
    )

    print(
        "Unique validation images:",
        len(val_image_set)
    )

    print(
        "Image overlap:",
        len(image_overlap)
    )

    if image_overlap:

        raise RuntimeError(

            "Training and validation "
            "image overlap detected"

        )


    # =====================================================
    # 5. EXTRACT PATHS AND CAPTIONS
    # =====================================================

    print(
        "\n[5/11] Preparing captions..."
    )

    train_image_paths = [

        image_path

        for image_path, caption in train_pairs

    ]

    train_captions = [

        f"startseq {caption} endseq"

        for image_path, caption in train_pairs

    ]

    val_image_paths = [

        image_path

        for image_path, caption in val_pairs

    ]

    val_captions = [

        f"startseq {caption} endseq"

        for image_path, caption in val_pairs

    ]

    print(
        "Training caption pairs:",
        len(train_image_paths)
    )

    print(
        "Validation caption pairs:",
        len(val_image_paths)
    )


    # =====================================================
    # 6. BUILD VOCABULARY
    # =====================================================

    print(
        "\n[6/11] Building vocabulary..."
    )

    text_processor = TextPreprocessor()

    text_processor.build_vocabulary(
        train_captions
    )

    text_processor.save_vocab(
        "vocab.json"
    )

    vocabulary = (

        text_processor
        .vectorizer
        .get_vocabulary()

    )

    vocab_size = len(
        vocabulary
    )

    word_to_index = {

        word: index

        for index, word in enumerate(
            vocabulary
        )

    }

    mask_token_id = word_to_index.get(
        "[UNK]",
        1
    )

    start_token_id = word_to_index.get(
        "startseq"
    )

    end_token_id = word_to_index.get(
        "endseq"
    )

    if start_token_id is None:

        raise ValueError(
            "startseq token not found"
        )

    if end_token_id is None:

        raise ValueError(
            "endseq token not found"
        )

    print(
        "Vocabulary size:",
        vocab_size
    )

    print(
        "Mask token ID:",
        mask_token_id
    )

    print(
        "Start token ID:",
        start_token_id
    )

    print(
        "End token ID:",
        end_token_id
    )


    # =====================================================
    # 7. TOKENIZE CAPTIONS
    # =====================================================

    print(
        "\n[7/11] Tokenizing captions..."
    )

    train_tokens = text_processor.process(
        train_captions
    )

    val_tokens = text_processor.process(
        val_captions
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
    # 8. CACHE EFFICIENTNET FEATURES
    # =====================================================

    print(
        "\n[8/11] Preparing EfficientNet feature cache..."
    )


    all_image_paths = list(

        dict.fromkeys(

            train_image_paths

            +

            val_image_paths

        )

    )

    print(
        "Unique images selected:",
        len(all_image_paths)
    )


    # =====================================================
    # 9. CREATE DATASETS
    # =====================================================

    print(
        "\n[9/11] Creating cached datasets..."
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

    for inputs, targets in train_dataset.take(1):

        visual_features, decoder_inputs = inputs

        print(
            "\nCached batch verification"
        )

        print(
            "Visual feature shape:",
            visual_features.shape
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
    # 10. CREATE VISIONGPT V2.6
    # =====================================================

    print(
        "\n[10/11] Creating VisionGPT v2.6..."
    )

    model = VisionGPT(

        vocab_size=vocab_size,

        context_dropout_rate=CONTEXT_DROPOUT_RATE,

        mask_token_id=mask_token_id,

        start_token_id=start_token_id,

        end_token_id=end_token_id

    )

    dummy_features = tf.zeros(

        (
            1,
            7,
            7,
            1280
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
            dummy_features,
            dummy_text
        ),

        training=False,

        use_cached_features=True

    )

    print(
        "VisionGPT output shape:",
        output.shape
    )

    print(
        "VisionGPT v2.6 architecture "
        "built successfully"
    )


    # =====================================================
    # 11. COMPILE
    # =====================================================

    print(
        "\n[11/11] Compiling VisionGPT v2.6..."
    )

    trainer = Trainer(

        model,

        learning_rate=0.001

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

        f"visiongpt_v2_6_best_"
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

    drive_backup = DriveCheckpointBackup(

        checkpoint_path=checkpoint_path,

        vocab_path="vocab.json",

        drive_backup_dir=DRIVE_BACKUP_DIR

    )

    callbacks = [

        checkpoint_callback,

        drive_backup,

        early_stopping,

        reduce_lr

    ]


    # =====================================================
    # TRAIN
    # =====================================================

    print(
        "\n=========================================="
    )

    print(
        "VISIONGPT v2.6 TRAINING STARTED"
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
        "Unique training images:",
        len(train_image_set)
    )

    print(
        "Unique validation images:",
        len(val_image_set)
    )

    print(
        "Image overlap:",
        len(image_overlap)
    )

    print(
        "Context dropout rate:",
        CONTEXT_DROPOUT_RATE
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

    print(
        "Drive backup directory:",
        DRIVE_BACKUP_DIR
    )

    history = model.fit(

        train_dataset,

        validation_data=val_dataset,

        epochs=EPOCHS,

        callbacks=callbacks

    )


    # =====================================================
    # FINAL DRIVE BACKUP
    # =====================================================

    if os.path.exists(
        checkpoint_path
    ):

        if os.path.exists(
            "/content/drive/MyDrive"
        ):

            os.makedirs(

                DRIVE_BACKUP_DIR,

                exist_ok=True

            )

            final_checkpoint_destination = os.path.join(

                DRIVE_BACKUP_DIR,

                os.path.basename(
                    checkpoint_path
                )

            )

            shutil.copy2(

                checkpoint_path,

                final_checkpoint_destination

            )

            shutil.copy2(

                "vocab.json",

                os.path.join(
                    DRIVE_BACKUP_DIR,
                    "vocab.json"
                )

            )

            print(
                "\nFinal Drive backup verified"
            )

            print(
                "Checkpoint:",
                final_checkpoint_destination
            )

            print(
                "Vocabulary:",
                os.path.join(
                    DRIVE_BACKUP_DIR,
                    "vocab.json"
                )
            )


    # =====================================================
    # COMPLETE
    # =====================================================

    print(
        "\n=========================================="
    )

    print(
        "VISIONGPT v2.6 TRAINING COMPLETED"
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
        "\nVisionGPT v2.6 trained successfully 🚀"
    )


if __name__ == "__main__":

    main()