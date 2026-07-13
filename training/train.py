import sys
import os
import datetime
import random
import shutil

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
from training.feature_cache import FeatureCache

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


FEATURE_CACHE_BATCH_SIZE = 64


EPOCHS = 15


VALIDATION_SPLIT = 0.10


RANDOM_SEED = 42


FEATURE_CACHE_DIR = (
    "/content/visiongpt_feature_cache"
)


DRIVE_BACKUP_DIR = (
    "/content/drive/MyDrive/"
    "VisionGPT/v2.5"
)


# =========================================================
# BEST CHECKPOINT DRIVE BACKUP CALLBACK
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
            "BEST MODEL BACKED UP TO DRIVE"
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
        "\n"
        "=========================================="
    )

    print(
        "Starting VisionGPT v2.5 Training"
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

        limit=TRAINING_LIMIT

    )


    pairs = loader.load()


    print(
        f"Loaded {len(pairs)} caption-image pairs"
    )


    # =====================================================
    # 4. SHUFFLE DATA
    # =====================================================

    print(
        "\n[4/11] Shuffling dataset..."
    )


    random.seed(
        RANDOM_SEED
    )


    random.shuffle(
        pairs
    )


    print(
        "Random seed:",
        RANDOM_SEED
    )


    # =====================================================
    # 5. SPLIT IMAGE PATHS AND CAPTIONS
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


    validation_size = int(

        len(image_paths)

        *

        VALIDATION_SPLIT

    )


    train_image_paths = image_paths[
        validation_size:
    ]


    train_captions = captions[
        validation_size:
    ]


    val_image_paths = image_paths[
        :validation_size
    ]


    val_captions = captions[
        :validation_size
    ]


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
        "\n[5/11] Building vocabulary..."
    )


    text_processor = TextPreprocessor()


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
        "\n[6/11] Tokenizing captions..."
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
    # 8. CREATE EFFICIENTNET FEATURE CACHE
    # =====================================================

    print(
        "\n[7/11] Caching EfficientNet features..."
    )


    feature_cache = FeatureCache(

        cache_dir=FEATURE_CACHE_DIR,

        batch_size=FEATURE_CACHE_BATCH_SIZE

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


    feature_cache.cache_features(
        all_image_paths
    )


    print(
        "EfficientNet feature caching completed"
    )


    # =====================================================
    # 9. CREATE CACHED DATASETS
    # =====================================================

    print(
        "\n[8/11] Creating cached datasets..."
    )


    pipeline = DataPipeline(

        feature_cache=feature_cache,

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


    # =====================================================
    # TEST ONE CACHED BATCH
    # =====================================================

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
    # 10. CREATE VISIONGPT V2.5
    # =====================================================

    print(
        "\n[9/11] Creating VisionGPT v2.5..."
    )


    model = VisionGPT(

        vocab_size=vocab_size

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
        "VisionGPT v2.5 architecture "
        "built successfully"
    )


    # =====================================================
    # TRAINER
    # =====================================================

    print(
        "\n[10/11] Compiling VisionGPT v2.5..."
    )


    trainer = Trainer(

        model,

        learning_rate=0.001

    )


    trainer.compile()


    # =====================================================
    # CALLBACKS
    # =====================================================

    print(
        "\n[11/11] Preparing training callbacks..."
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

        f"visiongpt_v2_5_best_"
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
        "\n"
        "=========================================="
    )


    print(
        "VISIONGPT v2.5 TRAINING STARTED"
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
        "Unique cached images:",
        len(all_image_paths)
    )


    print(
        "Batch size:",
        BATCH_SIZE
    )


    print(
        "Feature cache batch size:",
        FEATURE_CACHE_BATCH_SIZE
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
    # TRAINING COMPLETE
    # =====================================================

    print(
        "\n"
        "=========================================="
    )


    print(
        "VISIONGPT v2.5 TRAINING COMPLETED"
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
        "\nVisionGPT v2.5 trained successfully 🚀"
    )


if __name__ == "__main__":

    main()