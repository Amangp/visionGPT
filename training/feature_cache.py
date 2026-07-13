import os

import numpy as np
import tensorflow as tf

from models.vision_encoder import VisionEncoder


class FeatureCache:

    def __init__(
        self,
        cache_dir="feature_cache",
        batch_size=64
    ):

        self.cache_dir = cache_dir

        self.batch_size = batch_size

        os.makedirs(
            self.cache_dir,
            exist_ok=True
        )

        print(
            "\nCreating VisionEncoder..."
        )

        self.vision_encoder = VisionEncoder()

        self.vision_encoder.trainable = False

        print(
            "VisionEncoder ready"
        )

        print(
            "Feature cache directory:",
            self.cache_dir
        )


    # =====================================================
    # CREATE CACHE FILE PATH
    # =====================================================

    def get_cache_path(
        self,
        image_path
    ):

        image_name = os.path.basename(
            image_path
        )

        image_id = os.path.splitext(
            image_name
        )[0]

        return os.path.join(
            self.cache_dir,
            image_id + ".npy"
        )


    # =====================================================
    # PROCESS IMAGE
    # =====================================================

    def process_image(
        self,
        image_path
    ):

        image = tf.io.read_file(
            image_path
        )

        image = tf.image.decode_jpeg(
            image,
            channels=3
        )

        image = tf.cast(
            image,
            tf.float32
        )

        image = tf.image.resize(
            image,
            (
                224,
                224
            )
        )

        return image


    # =====================================================
    # CACHE FEATURES
    # =====================================================

    def cache_features(
        self,
        image_paths
    ):

        unique_image_paths = list(
            dict.fromkeys(
                image_paths
            )
        )

        total_images = len(
            unique_image_paths
        )

        print(
            "\n=========================================="
        )

        print(
            "VISIONGPT FEATURE CACHING"
        )

        print(
            "=========================================="
        )

        print(
            "Unique images:",
            total_images
        )

        print(
            "Batch size:",
            self.batch_size
        )


        uncached_paths = []

        for image_path in unique_image_paths:

            cache_path = self.get_cache_path(
                image_path
            )

            if not os.path.exists(
                cache_path
            ):

                uncached_paths.append(
                    image_path
                )


        print(
            "Already cached:",
            total_images - len(
                uncached_paths
            )
        )

        print(
            "Need caching:",
            len(
                uncached_paths
            )
        )


        if not uncached_paths:

            print(
                "\nAll image features already cached"
            )

            return


        dataset = (
            tf.data.Dataset
            .from_tensor_slices(
                uncached_paths
            )
        )


        dataset = dataset.map(

            self.process_image,

            num_parallel_calls=tf.data.AUTOTUNE,

            deterministic=False

        )


        dataset = dataset.batch(
            self.batch_size
        )


        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )


        processed = 0


        for batch_index, image_batch in enumerate(
            dataset
        ):

            features = self.vision_encoder.model(

                image_batch,

                training=False

            )


            features = features.numpy()


            batch_start = (
                batch_index
                *
                self.batch_size
            )


            batch_paths = uncached_paths[

                batch_start:

                batch_start
                +
                features.shape[0]

            ]


            for image_path, feature in zip(
                batch_paths,
                features
            ):

                cache_path = self.get_cache_path(
                    image_path
                )


                np.save(

                    cache_path,

                    feature.astype(
                        np.float16
                    )

                )


                processed += 1


            print(

                f"\rCached "
                f"{processed}/"
                f"{len(uncached_paths)} images",

                end="",

                flush=True

            )


        print(
            "\n"
        )

        print(
            "Feature caching completed"
        )


    # =====================================================
    # LOAD FEATURE
    # =====================================================

    def load_feature(
        self,
        image_path
    ):

        if isinstance(
            image_path,
            bytes
        ):

            image_path = image_path.decode(
                "utf-8"
            )


        cache_path = self.get_cache_path(
            image_path
        )


        if not os.path.exists(
            cache_path
        ):

            raise FileNotFoundError(

                "Cached feature not found: "
                + cache_path

            )


        feature = np.load(
            cache_path
        )


        feature = feature.astype(
            np.float32
        )


        return feature


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "Testing VisionGPT FeatureCache"
    )

    print(
        "=========================================="
    )


    sample_images = [

        (
            "Dataset/COCO/"
            "train2017/train2017/"
            "000000391895.jpg"
        ),

        (
            "Dataset/COCO/"
            "train2017/train2017/"
            "000000522418.jpg"
        )

    ]


    feature_cache = FeatureCache(

        cache_dir="feature_cache_test",

        batch_size=2

    )


    feature_cache.cache_features(
        sample_images
    )


    feature = feature_cache.load_feature(
        sample_images[0]
    )


    print(
        "Loaded feature shape:",
        feature.shape
    )


    print(
        "Loaded feature dtype:",
        feature.dtype
    )


    print(
        "FeatureCache test successful"
    )