import gc
import os

import numpy as np
import tensorflow as tf

from models.vision_encoder import VisionEncoder


class FeatureCache:

    def __init__(
        self,
        cache_dir="feature_cache",
        batch_size=32,
    ):

        self.cache_dir = cache_dir
        self.batch_size = batch_size

        os.makedirs(
            self.cache_dir,
            exist_ok=True,
        )

        print()
        print("Creating VisionEncoder...")

        self.vision_encoder = VisionEncoder()

        self.vision_encoder.trainable = False

        print("VisionEncoder ready")

        print(
            "Feature cache directory:",
            self.cache_dir,
        )

        print(
            "Feature cache batch size:",
            self.batch_size,
        )


    # =====================================================
    # CREATE CACHE FILE PATH
    # =====================================================

    def get_cache_path(
        self,
        image_path,
    ):

        if isinstance(
            image_path,
            bytes,
        ):

            image_path = image_path.decode(
                "utf-8"
            )

        image_name = os.path.basename(
            image_path
        )

        image_id = os.path.splitext(
            image_name
        )[0]

        return os.path.join(
            self.cache_dir,
            image_id + ".npy",
        )


    # =====================================================
    # PROCESS SINGLE IMAGE
    # =====================================================

    def process_image(
        self,
        image_path,
    ):

        image = tf.io.read_file(
            image_path
        )

        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False,
        )

        image = tf.cast(
            image,
            tf.float32,
        )

        image = tf.image.resize(
            image,
            (
                224,
                224,
            ),
            antialias=True,
        )

        image.set_shape(
            (
                224,
                224,
                3,
            )
        )

        return image


    # =====================================================
    # PROCESS ONE BATCH
    # =====================================================

    def process_batch(
        self,
        batch_paths,
    ):

        images = []

        valid_paths = []

        for image_path in batch_paths:

            try:

                image = self.process_image(
                    image_path
                )

                images.append(
                    image
                )

                valid_paths.append(
                    image_path
                )

            except Exception as error:

                print()

                print(
                    "Skipping image:"
                )

                print(
                    image_path
                )

                print(
                    "Reason:",
                    error,
                )


        if not images:

            return 0


        image_batch = tf.stack(
            images,
            axis=0,
        )


        features = self.vision_encoder.model(
            image_batch,
            training=False,
        )


        features = features.numpy()


        saved_count = 0


        for image_path, feature in zip(
            valid_paths,
            features,
        ):

            cache_path = self.get_cache_path(
                image_path
            )


            np.save(
                cache_path,
                feature.astype(
                    np.float16
                ),
            )


            saved_count += 1


        # -------------------------------------------------
        # Explicitly release batch memory
        # -------------------------------------------------

        del images

        del valid_paths

        del image_batch

        del features


        return saved_count


    # =====================================================
    # CACHE FEATURES
    # =====================================================

    def cache_features(
        self,
        image_paths,
    ):

        print()
        print("=" * 50)
        print("VISIONGPT STREAMING FEATURE CACHE")
        print("=" * 50)


        # -------------------------------------------------
        # Remove duplicate images
        # -------------------------------------------------

        unique_image_paths = list(
            dict.fromkeys(
                image_paths
            )
        )


        total_images = len(
            unique_image_paths
        )


        print(
            "Unique images:",
            total_images,
        )

        print(
            "Batch size:",
            self.batch_size,
        )


        # -------------------------------------------------
        # Count cache state without holding another
        # massive path list
        # -------------------------------------------------

        already_cached = 0


        for image_path in unique_image_paths:

            cache_path = self.get_cache_path(
                image_path
            )

            if os.path.exists(
                cache_path
            ):

                already_cached += 1


        need_caching = (
            total_images
            -
            already_cached
        )


        print(
            "Already cached:",
            already_cached,
        )

        print(
            "Need caching:",
            need_caching,
        )


        if need_caching == 0:

            print()

            print(
                "All image features already cached"
            )

            return


        # -------------------------------------------------
        # TRUE STREAMING CACHE
        # -------------------------------------------------

        batch_paths = []

        processed = already_cached

        newly_cached = 0

        skipped = 0


        for image_index, image_path in enumerate(
            unique_image_paths
        ):

            cache_path = self.get_cache_path(
                image_path
            )


            # Resume support
            if os.path.exists(
                cache_path
            ):

                continue


            batch_paths.append(
                image_path
            )


            # ---------------------------------------------
            # Process full batch
            # ---------------------------------------------

            if len(batch_paths) >= self.batch_size:

                expected_count = len(
                    batch_paths
                )


                saved_count = self.process_batch(
                    batch_paths
                )


                newly_cached += saved_count

                processed += saved_count

                skipped += (
                    expected_count
                    -
                    saved_count
                )


                print(
                    f"\rCached "
                    f"{processed}/"
                    f"{total_images} images "
                    f"| New: {newly_cached} "
                    f"| Skipped: {skipped}",
                    end="",
                    flush=True,
                )


                batch_paths.clear()


                # -----------------------------------------
                # Periodic garbage collection
                # -----------------------------------------

                if newly_cached % 1024 < self.batch_size:

                    gc.collect()


        # -------------------------------------------------
        # Process final partial batch
        # -------------------------------------------------

        if batch_paths:

            expected_count = len(
                batch_paths
            )


            saved_count = self.process_batch(
                batch_paths
            )


            newly_cached += saved_count

            processed += saved_count

            skipped += (
                expected_count
                -
                saved_count
            )


            batch_paths.clear()


        gc.collect()


        print()
        print()

        print("=" * 50)
        print("FEATURE CACHING COMPLETED")
        print("=" * 50)

        print(
            "Total images:",
            total_images,
        )

        print(
            "Previously cached:",
            already_cached,
        )

        print(
            "Newly cached:",
            newly_cached,
        )

        print(
            "Skipped:",
            skipped,
        )

        print(
            "Total cached:",
            processed,
        )

        print("=" * 50)


    # =====================================================
    # LOAD FEATURE
    # =====================================================

    def load_feature(
        self,
        image_path,
    ):

        if isinstance(
            image_path,
            bytes,
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
                +
                cache_path
            )


        feature = np.load(
            cache_path,
            allow_pickle=False,
        )


        return feature.astype(
            np.float32
        )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 50)
    print("Testing VisionGPT FeatureCache")
    print("=" * 50)


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
        ),

    ]


    feature_cache = FeatureCache(
        cache_dir="feature_cache_test",
        batch_size=2,
    )


    feature_cache.cache_features(
        sample_images
    )


    feature = feature_cache.load_feature(
        sample_images[0]
    )


    print(
        "Loaded feature shape:",
        feature.shape,
    )

    print(
        "Loaded feature dtype:",
        feature.dtype,
    )

    print(
        "FeatureCache test successful"
    )