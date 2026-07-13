import tensorflow as tf
import numpy as np


class DataPipeline:

    def __init__(
        self,
        feature_cache,
        batch_size=32,
        shuffle_buffer=10000
    ):

        self.feature_cache = feature_cache
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer


    # =====================================================
    # LOAD CACHED FEATURE
    # =====================================================

    def load_cached_feature(
        self,
        image_path
    ):

        feature = tf.numpy_function(

            func=self.feature_cache.load_feature,

            inp=[image_path],

            Tout=tf.float32

        )

        # Cached EfficientNetB0 feature shape

        feature.set_shape(
            (
                7,
                7,
                1280
            )
        )

        return feature


    # =====================================================
    # PREPARE TRAINING EXAMPLE
    # =====================================================

    def prepare_example(
        self,
        image_path,
        tokens
    ):

        image_features = self.load_cached_feature(
            image_path
        )

        decoder_input = tokens[:-1]

        target = tokens[1:]

        return (
            (
                image_features,
                decoder_input
            ),
            target
        )


    # =====================================================
    # CREATE DATASET
    # =====================================================

    def create(
        self,
        image_paths,
        text_tokens,
        training=True
    ):

        print(
            "\nCreating cached feature dataset..."
        )

        print(
            "Samples:",
            len(image_paths)
        )

        print(
            "Batch size:",
            self.batch_size
        )


        dataset = tf.data.Dataset.from_tensor_slices(
            (
                image_paths,
                text_tokens
            )
        )


        # =================================================
        # SHUFFLE
        # =================================================

        if training:

            shuffle_size = min(

                len(image_paths),

                self.shuffle_buffer

            )

            dataset = dataset.shuffle(

                buffer_size=shuffle_size,

                reshuffle_each_iteration=True

            )

            print(
                "Shuffle buffer:",
                shuffle_size
            )


        # =================================================
        # LOAD CACHED FEATURES IN PARALLEL
        # =================================================

        dataset = dataset.map(

            self.prepare_example,

            num_parallel_calls=tf.data.AUTOTUNE,

            deterministic=False

        )


        # =================================================
        # BATCH
        # =================================================

        dataset = dataset.batch(

            self.batch_size,

            drop_remainder=False

        )


        # =================================================
        # PREFETCH
        # =================================================

        dataset = dataset.prefetch(

            tf.data.AUTOTUNE

        )


        print(
            "Cached feature loading: parallel"
        )

        print(
            "Prefetch: AUTOTUNE"
        )

        print(
            "Dataset created successfully"
        )


        return dataset