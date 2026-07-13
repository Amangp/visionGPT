import tensorflow as tf


class DataPipeline:

    def __init__(
        self,
        image_processor,
        text_processor,
        batch_size=8,
        shuffle_buffer=10000
    ):

        self.image_processor = image_processor

        self.text_processor = text_processor

        self.batch_size = batch_size

        self.shuffle_buffer = shuffle_buffer


    # =====================================================
    # PROCESS SINGLE SAMPLE
    # =====================================================

    def process_sample(
        self,
        image_path,
        caption
    ):

        # =================================================
        # READ IMAGE
        # =================================================

        image = tf.io.read_file(
            image_path
        )


        # =================================================
        # DECODE IMAGE
        # =================================================

        image = tf.image.decode_jpeg(
            image,
            channels=3
        )


        # =================================================
        # CONVERT IMAGE TO FLOAT32
        # =================================================

        image = tf.cast(
            image,
            tf.float32
        )


        # =================================================
        # RESIZE IMAGE
        # =================================================

        image = tf.image.resize(
            image,
            (
                224,
                224
            )
        )


        # =================================================
        # IMPORTANT
        #
        # DO NOT DIVIDE BY 255
        #
        # EfficientNet handles preprocessing internally.
        # =================================================


        # =================================================
        # PROCESS CAPTION
        # =================================================

        caption_tokens = (
            self.text_processor
            .vectorizer(
                tf.expand_dims(
                    caption,
                    axis=0
                )
            )
        )


        caption_tokens = tf.squeeze(
            caption_tokens,
            axis=0
        )


        # =================================================
        # AUTOREGRESSIVE DECODER INPUT
        #
        # startseq a dog is running
        # =================================================

        decoder_input = caption_tokens[
            :-1
        ]


        # =================================================
        # AUTOREGRESSIVE TARGET
        #
        # a dog is running endseq
        # =================================================

        target = caption_tokens[
            1:
        ]


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

    def create_dataset(
        self,
        image_paths,
        captions,
        training=True
    ):

        print(
            "\nCreating TensorFlow dataset..."
        )


        print(
            "Samples:",
            len(image_paths)
        )


        print(
            "Batch size:",
            self.batch_size
        )


        # =================================================
        # CREATE BASE DATASET
        # =================================================

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                image_paths,
                captions
            )
        )


        # =================================================
        # SHUFFLE BEFORE MAP
        #
        # Shuffle paths and captions instead of full images.
        # This uses less memory.
        # =================================================

        if training:

            shuffle_size = min(
                len(image_paths),
                self.shuffle_buffer
            )


            print(
                "Shuffle buffer:",
                shuffle_size
            )


            dataset = dataset.shuffle(

                buffer_size=shuffle_size,

                reshuffle_each_iteration=True

            )


        # =================================================
        # PARALLEL IMAGE + TEXT PROCESSING
        #
        # Multiple CPU workers decode and resize images.
        # =================================================

        dataset = dataset.map(

            self.process_sample,

            num_parallel_calls=tf.data.AUTOTUNE,

            deterministic=False

        )


        # =================================================
        # BATCH DATASET
        # =================================================

        dataset = dataset.batch(

            self.batch_size,

            drop_remainder=False

        )


        # =================================================
        # PREFETCH
        #
        # CPU prepares the next batch while GPU trains
        # the current batch.
        # =================================================

        dataset = dataset.prefetch(

            tf.data.AUTOTUNE

        )


        print(
            "Parallel mapping: AUTOTUNE"
        )


        print(
            "Prefetch: AUTOTUNE"
        )


        print(
            "Dataset created successfully"
        )


        return dataset


# =========================================================
# TEST PIPELINE
# =========================================================

if __name__ == "__main__":

    from preprocessing.image_preprocessor import (
        ImagePreprocessor
    )

    from preprocessing.text_preprocessor import (
        TextPreprocessor
    )


    print(
        "\n=========================================="
    )

    print(
        "Testing VisionGPT DataPipeline"
    )

    print(
        "=========================================="
    )


    image_processor = ImagePreprocessor()


    text_processor = TextPreprocessor()


    sample_captions = [

        "startseq a dog is running endseq",

        "startseq a man is walking endseq"

    ]


    # =====================================================
    # BUILD TEST VOCABULARY
    # =====================================================

    text_processor.vectorizer.adapt(

        tf.data.Dataset.from_tensor_slices(
            sample_captions
        ).batch(
            2
        )

    )


    sample_images = [

        (
            "Dataset/COCO/train2017/train2017/"
            "000000391895.jpg"
        ),

        (
            "Dataset/COCO/train2017/train2017/"
            "000000522418.jpg"
        )

    ]


    pipeline = DataPipeline(

        image_processor=image_processor,

        text_processor=text_processor,

        batch_size=2

    )


    dataset = pipeline.create_dataset(

        image_paths=sample_images,

        captions=sample_captions,

        training=True

    )


    # =====================================================
    # INSPECT ONE BATCH
    # =====================================================

    for inputs, targets in dataset.take(1):

        images, decoder_inputs = inputs


        print(
            "\n=========================================="
        )


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


        print(
            "Image dtype:",
            images.dtype
        )


        print(
            "Image min:",
            tf.reduce_min(
                images
            ).numpy()
        )


        print(
            "Image max:",
            tf.reduce_max(
                images
            ).numpy()
        )


        print(
            "=========================================="
        )