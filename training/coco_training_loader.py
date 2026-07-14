import json
import os
import random


class COCOTrainingLoader:
    """
    VisionGPT COCO Caption Training Loader

    Loads image-caption pairs from the COCO captions annotation file.

    Features:
    - Full COCO dataset support
    - Optional pair limit for experiments
    - Deterministic shuffling
    - Missing-image filtering
    - Dataset statistics
    """

    def __init__(
        self,
        image_folder,
        caption_file,
        limit=None,
        seed=42,
    ):
        self.image_folder = image_folder
        self.caption_file = caption_file
        self.limit = limit
        self.seed = seed


    def load(self):

        print()
        print("=" * 50)
        print("VISIONGPT COCO TRAINING LOADER")
        print("=" * 50)

        # --------------------------------------------------
        # Validate paths
        # --------------------------------------------------

        if not os.path.isdir(self.image_folder):
            raise FileNotFoundError(
                f"COCO image folder not found: "
                f"{self.image_folder}"
            )

        if not os.path.isfile(self.caption_file):
            raise FileNotFoundError(
                f"COCO caption file not found: "
                f"{self.caption_file}"
            )


        print(
            f"Image folder: {self.image_folder}"
        )

        print(
            f"Caption file: {self.caption_file}"
        )


        # --------------------------------------------------
        # Load COCO caption JSON
        # --------------------------------------------------

        print()
        print("Loading COCO annotations...")


        with open(
            self.caption_file,
            "r",
            encoding="utf-8",
        ) as file:

            coco_data = json.load(file)


        if "annotations" not in coco_data:
            raise ValueError(
                "Invalid COCO caption file. "
                "'annotations' key not found."
            )


        annotations = coco_data["annotations"]


        print(
            f"Total COCO annotations: "
            f"{len(annotations)}"
        )


        # --------------------------------------------------
        # Create image-caption pairs
        # --------------------------------------------------

        print()
        print("Creating image-caption pairs...")


        pairs = []



        for annotation in annotations:

            image_id = annotation.get(
                "image_id"
            )

            caption = annotation.get(
                "caption"
            )


            if image_id is None:
                continue


            if caption is None:
                continue


            caption = caption.strip()


            if not caption:
                continue


            image_filename = (
                f"{int(image_id):012d}.jpg"
            )


            image_path = os.path.join(
                self.image_folder,
                image_filename
            )

            pairs.append(
                (
                    image_path,
                    caption
                )
            )


        if not pairs:
            raise ValueError(
                "No valid COCO image-caption pairs "
                "were found."
            )


        print(
            f"Valid caption-image pairs: "
            f"{len(pairs)}"
        )


        # --------------------------------------------------
        # Deterministic shuffle
        # --------------------------------------------------

        print()
        print(
            f"Shuffling dataset with seed "
            f"{self.seed}..."
        )


        random_generator = random.Random(
            self.seed
        )


        random_generator.shuffle(
            pairs
        )


        # --------------------------------------------------
        # Optional dataset limit
        # --------------------------------------------------

        if self.limit is None:

            selected_pairs = pairs

            print()
            print(
                "Dataset mode: FULL COCO DATASET"
            )

        else:

            if self.limit <= 0:
                raise ValueError(
                    "Dataset limit must be greater "
                    "than 0 or None."
                )


            selected_pairs = pairs[
                :self.limit
            ]


            print()
            print(
                "Dataset mode: LIMITED DATASET"
            )

            print(
                f"Dataset limit: {self.limit}"
            )


        # --------------------------------------------------
        # Dataset statistics
        # --------------------------------------------------

        unique_images = {
            image_path
            for image_path, caption
            in selected_pairs
        }


        repeated_caption_image_pairs = (
            len(selected_pairs)
            - len(unique_images)
        )


        print()
        print("=" * 50)
        print("COCO DATASET STATISTICS")
        print("=" * 50)


        print(
            f"Selected caption pairs: "
            f"{len(selected_pairs)}"
        )


        print(
            f"Unique images: "
            f"{len(unique_images)}"
        )


        print(
            f"Repeated caption-image pairs: "
            f"{repeated_caption_image_pairs}"
        )


        if unique_images:

            average_captions = (
                len(selected_pairs)
                / len(unique_images)
            )

        else:

            average_captions = 0.0


        print(
            f"Average captions per image: "
            f"{average_captions:.2f}"
        )


        print("=" * 50)


        return selected_pairs


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    IMAGE_FOLDER = (
        "Dataset/COCO/"
        "train2017/train2017"
    )


    CAPTION_FILE = (
        "Dataset/COCO/"
        "annotations_trainval2017/"
        "annotations/"
        "captions_train2017.json"
    )


    loader = COCOTrainingLoader(
        image_folder=IMAGE_FOLDER,
        caption_file=CAPTION_FILE,
        limit=None,
        seed=42,
    )


    pairs = loader.load()


    print()
    print("Example caption-image pairs:")
    print()


    for image_path, caption in pairs[:5]:

        print(
            f"Image: {image_path}"
        )

        print(
            f"Caption: {caption}"
        )

        print("-" * 50)