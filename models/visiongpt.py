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
            embed_dim=embed_dim
        )


    def call(
        self,
        inputs,
        training=False
    ):

        image, text = inputs

        image_features = (
            self.vision_encoder.encode(
                image
            )
        )

        image_features = self.fusion_layer(
            image_features,
            training=training
        )

        output = self.answer_decoder(
            text,
            image_features,
            training=training
        )

        return output

if __name__ == "__main__":

    model = VisionGPT()

    image = tf.random.normal(
        (1, 224, 224, 3)
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
        ]
    )

    result = model(
        (
            image,
            text
        )
    )

    print(
        "VisionGPT output shape:",
        result.shape
    )