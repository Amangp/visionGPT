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
        end_token_id=4,
        answer_token_id=5,
        task_caption_token_id=7,
        task_ocr_token_id=14
    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.vision_encoder = VisionEncoder(
            embed_dim=embed_dim
        )

        self.fusion_layer = FusionLayer(
            embed_dim=embed_dim
        )

        self.context_dropout = ContextDropout(
            dropout_rate=context_dropout_rate,
            mask_token_id=mask_token_id,
            protected_token_ids=[
                0,
                mask_token_id,
                start_token_id,
                end_token_id,
                answer_token_id,
                task_caption_token_id,
                task_ocr_token_id
            ]
        )

        self.answer_decoder = AnswerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_decoder_layers=3
        )


    def call(
        self,
        inputs,
        training=False
    ):

        images, text = inputs

        image_features = self.vision_encoder.encode(
            images,
            training=training
        )

        image_features = self.fusion_layer(
            image_features,
            training=training
        )

        text = self.context_dropout(
            text,
            training=training
        )

        return self.answer_decoder(
            text,
            image_features,
            training=training
        )


    def train_step(
    self,
    data
    ):

        inputs, targets, target_mask = data

        with tf.GradientTape() as tape:

            predictions = self(
                inputs,
                training=True
            )

            loss = self.compute_loss(
                y=targets,
                y_pred=predictions,
                sample_weight=target_mask
            )

        trainable_variables = (
            self.trainable_variables
        )

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
                    predictions,
                    sample_weight=target_mask
                )

        return {

            metric.name:
                metric.result()

            for metric in self.metrics

    }


    def test_step(
        self,
        data
    ):

        inputs, targets, target_mask = data

        predictions = self(
            inputs,
            training=False
        )

        loss = self.compute_loss(
            y=targets,
            y_pred=predictions,
            sample_weight=target_mask
        )

        for metric in self.metrics:

            if metric.name == "loss":

                metric.update_state(
                    loss
                )

            else:

                metric.update_state(
                    targets,
                    predictions,
                    sample_weight=target_mask
                )

        return {

            metric.name:
                metric.result()

            for metric in self.metrics

        }


if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "Testing VisionGPT v3.2"
    )

    print(
        "=========================================="
    )

    model = VisionGPT(
        vocab_size=10000,
        context_dropout_rate=0.10,
        mask_token_id=1,
        start_token_id=3,
        end_token_id=4,
        answer_token_id=5,
        task_caption_token_id=7,
        task_ocr_token_id=14
    )

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

    output = model(
        (
            image,
            text
        ),
        training=False
    )

    print(
        "Output shape:",
        output.shape
    )

    print(
        "\nVisionGPT v3.2 test successful"
    )
