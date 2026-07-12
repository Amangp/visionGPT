import tensorflow as tf


class VisionEncoder:

    def __init__(
            self,
            embed_dim=256
    ):

        # --------------------------------------------------
        # EfficientNetB0 Backbone
        # --------------------------------------------------

        self.model = tf.keras.applications.EfficientNetB0(

            include_top=False,

            weights="imagenet",

            pooling=None

        )

        # --------------------------------------------------
        # Freeze EfficientNet
        # --------------------------------------------------

        self.model.trainable = False


    def encode(
            self,
            image
    ):

        # --------------------------------------------------
        # Extract Spatial Image Features
        #
        # Input:
        # (batch, 224, 224, 3)
        #
        # Output:
        # (batch, 7, 7, 1280)
        # --------------------------------------------------

        features = self.model(
            image,
            training=False
        )

        return features


if __name__ == "__main__":

    encoder = VisionEncoder()

    sample_image = tf.random.normal(
        (
            1,
            224,
            224,
            3
        )
    )

    output = encoder.encode(
        sample_image
    )

    print(
        "Vision Encoder output shape:",
        output.shape
    )