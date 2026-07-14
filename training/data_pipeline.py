import tensorflow as tf


class DataPipeline:

    def __init__(
        self,
        batch_size=8,
        shuffle_buffer=10000
    ):

        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer


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

        image = tf.image.decode_jpeg(
            image,
            channels=3
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

        image = tf.keras.applications.efficientnet.preprocess_input(
            image
        )

        return image


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

        return (
            (
                image,
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

        print("\n==========================================")
        print("CREATING IMAGE DATASET")
        print("==========================================")

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

        # ---------------------------------------------
        # Shuffle
        # ---------------------------------------------

        if training:

            dataset = dataset.shuffle(

                min(
                    len(image_paths),
                    self.shuffle_buffer
                ),

                reshuffle_each_iteration=True

            )

        # ---------------------------------------------
        # Load Images
        # ---------------------------------------------

        dataset = dataset.map(

            self.prepare_example,

            num_parallel_calls=tf.data.AUTOTUNE

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
            "Dataset created successfully"
        )

        return dataset


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "Testing DataPipeline"
    )

    print(
        "=========================================="
    )

    sample_images = [

        "Dataset/COCO/train2017/train2017/000000391895.jpg",

        "Dataset/COCO/train2017/train2017/000000522418.jpg"

    ]

    sample_tokens = tf.constant(

        [

            [3,10,20,30,40,4],

            [3,15,25,35,45,4]

        ],

        dtype=tf.int64

    )

    pipeline = DataPipeline(

        batch_size=2

    )

    dataset = pipeline.create(

        sample_images,

        sample_tokens,

        training=False

    )

    for (images, decoder_input), target in dataset.take(1):

        print()

        print(
            "Image shape:",
            images.shape
        )

        print(
            "Decoder input shape:",
            decoder_input.shape
        )

        print(
            "Target shape:",
            target.shape
        )

    print()

    print(
        "DataPipeline test successful"
    )
