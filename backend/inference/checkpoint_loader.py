import os

import tensorflow as tf


class CheckpointLoader:

    def __init__(
        self,
        sequence_length=50
    ):

        self.sequence_length = sequence_length
        self.model = None


    # =====================================================
    # LOAD CHECKPOINT
    # =====================================================

    def load(
        self,
        model,
        checkpoint_path
    ):

        print(
            "\n=========================================="
        )

        print(
            "LOADING VISIONGPT CHECKPOINT"
        )

        print(
            "=========================================="
        )


        # =================================================
        # CHECK CHECKPOINT
        # =================================================

        if not os.path.exists(
            checkpoint_path
        ):

            raise FileNotFoundError(

                "VisionGPT checkpoint not found: "
                f"{checkpoint_path}"

            )


        print(
            "Checkpoint:",
            checkpoint_path
        )


        # =================================================
        # BUILD MODEL
        #
        # VisionGPT is a subclassed Keras model.
        # Variables must exist before load_weights().
        # =================================================

        print(
            "Building VisionGPT architecture..."
        )


        dummy_image = tf.zeros(

            (
                1,
                224,
                224,
                3
            ),

            dtype=tf.float32

        )


        dummy_text = tf.ones(

            (
                1,
                self.sequence_length - 1
            ),

            dtype=tf.int64

        )


        output = model(

            (
                dummy_image,
                dummy_text
            ),

            training=False

        )


        print(
            "Model output shape:",
            output.shape
        )


        print(
            "VisionGPT architecture built"
        )


        # =================================================
        # LOAD WEIGHTS
        # =================================================

        print(
            "Loading trained weights..."
        )


        model.load_weights(
            checkpoint_path
        )


        print(
            "Weights loaded successfully"
        )


        print(
            "=========================================="
        )


        self.model = model

        return model


visiongpt_loader = CheckpointLoader()