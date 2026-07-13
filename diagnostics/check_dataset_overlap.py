import random

from training.coco_training_loader import (
    COCOTrainingLoader
)


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

VALIDATION_SPLIT = 0.10

RANDOM_SEED = 42


loader = COCOTrainingLoader(

    image_folder=IMAGE_FOLDER,

    caption_file=CAPTION_FILE,

    limit=TRAINING_LIMIT

)


pairs = loader.load()


random.seed(
    RANDOM_SEED
)


random.shuffle(
    pairs
)


validation_size = int(

    len(pairs)

    *

    VALIDATION_SPLIT

)


val_pairs = pairs[
    :validation_size
]


train_pairs = pairs[
    validation_size:
]


train_images = set(

    image_path

    for image_path, caption in train_pairs

)


val_images = set(

    image_path

    for image_path, caption in val_pairs

)


overlap = (

    train_images

    &

    val_images

)


print(
    "\n=========================================="
)


print(
    "VISIONGPT DATASET OVERLAP TEST"
)


print(
    "=========================================="
)


print(
    "Training pairs:",
    len(train_pairs)
)


print(
    "Validation pairs:",
    len(val_pairs)
)


print(
    "Unique training images:",
    len(train_images)
)


print(
    "Unique validation images:",
    len(val_images)
)


print(
    "OVERLAPPING IMAGES:",
    len(overlap)
)


overlap_percentage = (

    len(overlap)

    /

    max(
        len(val_images),
        1
    )

    *

    100

)


print(

    "Validation image overlap:",

    f"{overlap_percentage:.2f}%"

)


print(
    "\nExample overlapping images:"
)


for image_path in list(
    overlap
)[:10]:

    print(
        image_path
    )


print(
    "=========================================="
)