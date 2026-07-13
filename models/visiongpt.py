import tensorflow as tf

from models.vision_encoder import VisionEncoder
from models.fusion_layer import FusionLayer
from models.answer_decoder import AnswerDecoder
from models.context_dropout import ContextDropout


class VisionGPT(tf.keras.Model):

    def __init__(
        self,
        vocab_size=10000,
        embed_dim=256,
        context_dropout_rate=0.10,
        mask_token_id=1,
        start_token_id=3,
        end_token_id=4
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # =================================================
        # VISION ENCODER
        # =================================================

        self.vision_encoder = VisionEncoder(
            embed_dim=embed_dim
        )

        # =================================================
        # VISUAL FUSION TRANSFORMER
        # =================================================

        self.fusion_layer = FusionLayer(
            embed_dim=embed_dim
        )

        # =================================================
        # CAPTION CONTEXT DROPOUT
        # =================================================

        self.context_dropout = ContextDropout(

            dropout_rate=context_dropout_rate,

            mask_token_id=mask_token_id,

            protected_token_ids=[
                0,
                mask_token_id,
                start_token_id,
                end_token_id
            ]

        )

        # =================================================
        # ANSWER DECODER
        # =================================================

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
        # CACHED VISUAL FEATURES
        # =================================================

        if use_cached_features:

            image_features = visual_input

        # =================================================
        # RAW IMAGE INFERENCE
        # =================================================

        else:

            image_features = self.vision_encoder.model(

                visual_input,

                training=False

            )

        # =================================================
        # VISUAL TRANSFORMER
        # =================================================

        image_features = self.fusion_layer(

            image_features,

            training=training

        )

        # =================================================
        # CONTEXT DROPOUT
        #
        # ACTIVE ONLY DURING TRAINING
        # =================================================

        text = self.context_dropout(

            text,

            training=training

        )

        # =================================================
        # DECODER
        # =================================================

        output = self.answer_decoder(

            text,

            image_features,

            training=training

        )

        return output


    # =====================================================
    # TRAIN STEP
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

        trainable_variables = self.trainable_variables

        gradients = tape.gradient(

            loss,

            trainable_variables

        )

        gradients_and_variables = [

            (
                gradient,
                variable
            )

            for gradient, variable in zip(

                gradients,

                trainable_variables

            )

            if gradient is not None

        ]

        self.optimizer.apply_gradients(

            gradients_and_variables

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


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "Testing VisionGPT v2.6"
    )

    print(
        "=========================================="
    )

    model = VisionGPT(

        vocab_size=10000,

        context_dropout_rate=0.10,

        mask_token_id=1,

        start_token_id=3,

        end_token_id=4

    )

    # =====================================================
    # RAW IMAGE TEST
    # =====================================================

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
                3,
                10,
                20,
                30,
                4
            ]
        ],

        dtype=tf.int64

    )

    normal_output = model(

        (
            image,
            text
        ),

        training=False

    )

    print(

        "Normal image output shape:",

        normal_output.shape

    )

    # =====================================================
    # CACHED FEATURE TEST
    # =====================================================

    cached_features = tf.random.normal(

        (
            1,
            7,
            7,
            1280
        )

    )

    cached_output = model(

        (
            cached_features,
            text
        ),

        training=False,

        use_cached_features=True

    )

    print(

        "Cached feature output shape:",

        cached_output.shape

    )

    # =====================================================
    # TRAINING MODE TEST
    # =====================================================

    training_output = model(

        (
            cached_features,
            text
        ),

        training=True,

        use_cached_features=True

    )

    print(

        "Training output shape:",

        training_output.shape

    )

    print(
        "\nVisionGPT v2.6 test successful"
    )