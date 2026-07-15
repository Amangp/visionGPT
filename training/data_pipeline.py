import tensorflow as tf


class DataPipeline:

    def __init__(
        self,
        batch_size=8,
        shuffle_buffer=10000,
        answer_token_id=5
    ):

        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.answer_token_id = answer_token_id


    # =====================================================
    # LOAD IMAGE
    # =====================================================

    def load_image(
        self,
        image_path
    ):

        image = tf.io.read_file(
            image_path
        )

        image = tf.image.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image.set_shape(
            (
                None,
                None,
                3
            )
        )

        image = tf.image.resize(
            image,
            (
                224,
                224
            )
        )

        image = tf.cast(
            image,
            tf.float32
        )

        image = (
            tf.keras.applications
            .efficientnet
            .preprocess_input(
                image
            )
        )

        return image


    # =====================================================
    # CREATE ANSWER MASK
    # =====================================================

    def create_answer_mask(
        self,
        tokens
    ):

        answer_positions = tf.equal(
            tokens,
            tf.cast(
                self.answer_token_id,
                tokens.dtype
            )
        )

        answer_positions = tf.cast(
            answer_positions,
            tf.int32
        )

        answer_seen = tf.cumsum(
            answer_positions
        )

        answer_mask = tf.greater(
            answer_seen,
            0
        )

        answer_mask = tf.logical_and(
            answer_mask,
            tf.not_equal(
                tokens,
                self.answer_token_id
            )
        )

        answer_mask = tf.logical_and(
            answer_mask,
            tf.not_equal(
                tokens,
                0
            )
        )

        return tf.cast(
            answer_mask,
            tf.float32
        )


    # =====================================================
    # PREPARE SAMPLE
    # =====================================================

    def prepare_example(
        self,
        image_path,
        tokens
    ):

        image = self.load_image(
            image_path
        )

        decoder_input = tokens[:-1]

        target = tokens[1:]

        target_mask = self.create_answer_mask(
            target
        )

        return (
            (
                image,
                decoder_input
            ),
            target,
            target_mask
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
            "\n=========================================="
        )

        print(
            "CREATING VISIONGPT v4 DATASET"
        )

        print(
            "=========================================="
        )

        print(
            "Samples:",
            len(image_paths)
        )

        print(
            "Batch size:",
            self.batch_size
        )

        print(
            "Answer token ID:",
            self.answer_token_id
        )


        dataset = (
            tf.data.Dataset
            .from_tensor_slices(
                (
                    image_paths,
                    text_tokens
                )
            )
        )


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


        dataset = dataset.map(

            self.prepare_example,

            num_parallel_calls=tf.data.AUTOTUNE

        )

        # ---------------------------------------------
        # Cache decoded images
        # ---------------------------------------------

        if training:

            dataset = dataset.cache(
                "cache/train.cache"
            )

        else:

            dataset = dataset.cache(
                "cache/val.cache"
            )

        # ---------------------------------------------
        # Batch
        # ---------------------------------------------

        dataset = dataset.batch(

            self.batch_size,

            drop_remainder=False

        )

        # ---------------------------------------------
        # Prefetch
        # ---------------------------------------------

        dataset = dataset.prefetch(

            tf.data.AUTOTUNE

        )


        print(
            "VisionGPT v4 dataset created"
        )


        return dataset


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    pipeline = DataPipeline(
        batch_size=2,
        answer_token_id=5
    )


    sample_tokens = tf.constant(

        [
            [
                3,
                9,
                18,
                2,
                16,
                5,
                6,
                17,
                10,
                12,
                6,
                20,
                4,
                0,
                0
            ],

            [
                3,
                8,
                7,
                15,
                2,
                19,
                13,
                11,
                5,
                14,
                4,
                0,
                0,
                0,
                0
            ]
        ],

        dtype=tf.int64

    )


    for tokens in sample_tokens:

        target = tokens[1:]

        mask = pipeline.create_answer_mask(
            target
        )


        print(
            "\nTarget:"
        )

        print(
            target.numpy()
        )


        print(
            "Answer mask:"
        )

        print(
            mask.numpy()
        )


    print(
        "\nDataPipeline v4 mask test successful"
    )