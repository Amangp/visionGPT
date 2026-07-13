import random
from collections import defaultdict


class ImageLevelSplitter:

    def __init__(
        self,
        validation_split=0.10,
        random_seed=42
    ):

        self.validation_split = validation_split
        self.random_seed = random_seed


    def split(
        self,
        pairs
    ):

        print(
            "\n=========================================="
        )

        print(
            "IMAGE LEVEL DATASET SPLIT"
        )

        print(
            "=========================================="
        )


        # =================================================
        # GROUP CAPTIONS BY IMAGE
        # =================================================

        image_groups = defaultdict(
            list
        )


        for image_path, caption in pairs:

            image_groups[
                image_path
            ].append(
                caption
            )


        unique_images = list(
            image_groups.keys()
        )


        print(
            "Total caption pairs:",
            len(pairs)
        )


        print(
            "Unique images:",
            len(unique_images)
        )


        # =================================================
        # SHUFFLE UNIQUE IMAGES
        # =================================================

        random_generator = random.Random(
            self.random_seed
        )


        random_generator.shuffle(
            unique_images
        )


        # =================================================
        # SPLIT IMAGES
        # =================================================

        validation_image_count = int(

            len(unique_images)

            *

            self.validation_split

        )


        val_images = set(

            unique_images[
                :validation_image_count
            ]

        )


        train_images = set(

            unique_images[
                validation_image_count:
            ]

        )


        # =================================================
        # RESTORE CAPTION PAIRS
        # =================================================

        train_pairs = []


        for image_path in train_images:

            for caption in image_groups[
                image_path
            ]:

                train_pairs.append(

                    (
                        image_path,
                        caption
                    )

                )


        val_pairs = []


        for image_path in val_images:

            for caption in image_groups[
                image_path
            ]:

                val_pairs.append(

                    (
                        image_path,
                        caption
                    )

                )


        # =================================================
        # SHUFFLE PAIRS INSIDE EACH SPLIT
        # =================================================

        random_generator.shuffle(
            train_pairs
        )


        random_generator.shuffle(
            val_pairs
        )


        # =================================================
        # VERIFY ZERO OVERLAP
        # =================================================

        overlap = (

            train_images

            &

            val_images

        )


        print(
            "Training images:",
            len(train_images)
        )


        print(
            "Validation images:",
            len(val_images)
        )


        print(
            "Training caption pairs:",
            len(train_pairs)
        )


        print(
            "Validation caption pairs:",
            len(val_pairs)
        )


        print(
            "Image overlap:",
            len(overlap)
        )


        if overlap:

            raise RuntimeError(

                "Image-level split failed. "
                "Training and validation overlap detected."

            )


        print(
            "Image-level split successful"
        )


        print(
            "=========================================="
        )


        return (
            train_pairs,
            val_pairs
        )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    sample_pairs = [

        (
            "image_1.jpg",
            "a dog standing"
        ),

        (
            "image_1.jpg",
            "a dog outside"
        ),

        (
            "image_2.jpg",
            "a car"
        ),

        (
            "image_2.jpg",
            "a black car"
        ),

        (
            "image_3.jpg",
            "a tree"
        ),

        (
            "image_4.jpg",
            "a person"
        ),

        (
            "image_5.jpg",
            "a cat"
        ),

        (
            "image_6.jpg",
            "a bicycle"
        ),

        (
            "image_7.jpg",
            "a bus"
        ),

        (
            "image_8.jpg",
            "a train"
        ),

        (
            "image_9.jpg",
            "a horse"
        ),

        (
            "image_10.jpg",
            "a bird"
        )

    ]


    splitter = ImageLevelSplitter(

        validation_split=0.20,

        random_seed=42

    )


    train_pairs, val_pairs = splitter.split(
        sample_pairs
    )


    train_images = {

        image_path

        for image_path, caption in train_pairs

    }


    val_images = {

        image_path

        for image_path, caption in val_pairs

    }


    assert len(

        train_images & val_images

    ) == 0


    print(
        "\nImageLevelSplitter test successful"
    )