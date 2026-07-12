import tensorflow as tf


class DataPipeline:

    def __init__(
            self,
            batch_size=32,
            image_size=(224, 224)
    ):

        self.batch_size = batch_size
        self.image_size = image_size


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
            self.image_size
        )

        # IMPORTANT:
        # EfficientNetB0 expects [0, 255]
        # Do NOT divide by 255

        image = tf.cast(
            image,
            tf.float32
        )

        return image


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


    def create(
            self,
            image_paths,
            text_tokens,
            training=True
    ):

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                image_paths,
                text_tokens
            )
        )

        if training:

            dataset = dataset.shuffle(
                buffer_size=5000,
                reshuffle_each_iteration=True
            )

        dataset = dataset.map(
            self.prepare_example,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.batch(
            self.batch_size
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset


if __name__ == "__main__":

    pipeline = DataPipeline(
        batch_size=8
    )

    fake_paths = tf.constant(
        [
            "test.jpg",
            "test.jpg"
        ]
    )

    fake_tokens = tf.constant(
        [
            [3, 10, 20, 30, 4, 0],
            [3, 15, 25, 35, 4, 0]
        ],
        dtype=tf.int64
    )

    dataset = pipeline.create(
        fake_paths,
        fake_tokens
    )

    for inputs, targets in dataset.take(1):

        images, decoder_inputs = inputs

        print(
            "Image shape:",
            images.shape
        )

        print(
            "Decoder input shape:",
            decoder_inputs.shape
        )

        print(
            "Target shape:",
            targets.shape
        )