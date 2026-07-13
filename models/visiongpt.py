import tensorflow as tf

from models.vision_encoder import VisionEncoder
from models.fusion_layer import FusionLayer
from models.answer_decoder import AnswerDecoder


class VisionGPT(tf.keras.Model):

    def __init__(
        self,
        vocab_size=10000,
        embed_dim=256
    ):

        super().__init__()

        self.vision_encoder = VisionEncoder(
            embed_dim=embed_dim
        )

        self.fusion_layer = FusionLayer(
            embed_dim=embed_dim
        )

        self.answer_decoder = AnswerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_decoder_layers=3
        )


    # =====================================================
    # CALL
    # =====================================================

    def call(
        self,
        inputs,
        training=False,
        use_cached_features=False
    ):

        visual_input, text = inputs


        # =================================================
        # CACHED FEATURE TRAINING
        # =================================================

        if use_cached_features:

            image_features = visual_input


        # =================================================
        # NORMAL IMAGE INFERENCE
        # =================================================

        else:

            image_features = (
                self.vision_encoder.model(

                    visual_input,

                    training=False

                )
            )


        # =================================================
        # VISUAL TRANSFORMER
        # =================================================

        image_features = self.fusion_layer(

            image_features,

            training=training

        )


        # =================================================
        # TRANSFORMER DECODER
        # =================================================

        output = self.answer_decoder(

            text,

            image_features,

            training=training

        )


        return output


    # =====================================================
    # TRAIN STEP
    #
    # Training dataset contains cached visual features.
    # =====================================================

    def train_step(
        self,
        data
    ):

        inputs, targets = data

        visual_features, text = inputs


        with tf.GradientTape() as tape:

            predictions = self(

                (
                    visual_features,
                    text
                ),

                training=True,

                use_cached_features=True

            )


            loss = self.compute_loss(

                y=targets,

                y_pred=predictions

            )


        trainable_variables = (
            self.trainable_variables
        )


        gradients = tape.gradient(

            loss,

            trainable_variables

        )


        self.optimizer.apply_gradients(

            zip(
                gradients,
                trainable_variables
            )

        )


        for metric in self.metrics:

            if metric.name == "loss":

                metric.update_state(
                    loss
                )

            else:

                metric.update_state(

                    targets,

                    predictions

                )


        return {

            metric.name: metric.result()

            for metric in self.metrics

        }


    # =====================================================
    # VALIDATION STEP
    # =====================================================

    def test_step(
        self,
        data
    ):

        inputs, targets = data

        visual_features, text = inputs


        predictions = self(

            (
                visual_features,
                text
            ),

            training=False,

            use_cached_features=True

        )


        loss = self.compute_loss(

            y=targets,

            y_pred=predictions

        )


        for metric in self.metrics:

            if metric.name == "loss":

                metric.update_state(
                    loss
                )

            else:

                metric.update_state(

                    targets,

                    predictions

                )


        return {

            metric.name: metric.result()

            for metric in self.metrics

        }


if __name__ == "__main__":

    model = VisionGPT()

    image = tf.random.normal(
        (
            1,
            224,
            224,
            3
        )
    )

    text = tf.constant(
        [
            [
                10,
                20,
                30,
                0,
                0
            ]
        ],
        dtype=tf.int64
    )

    result = model(
        (
            image,
            text
        ),
        training=False
    )

    print(
        "Normal image output shape:",
        result.shape
    )


    cached_features = tf.random.normal(
        (
            1,
            7,
            7,
            1280
        )
    )

    cached_result = model(
        (
            cached_features,
            text
        ),
        training=False,
        use_cached_features=True
    )

    print(
        "Cached feature output shape:",
        cached_result.shape
    )