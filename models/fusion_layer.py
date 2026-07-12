import tensorflow as tf


class FusionLayer(tf.keras.layers.Layer):

    def __init__(
            self,
            embed_dim=256
    ):

        super().__init__()

        # --------------------------------------------------
        # Project EfficientNet features
        # 1280 -> 256
        # --------------------------------------------------

        self.image_projection = tf.keras.layers.Dense(
            embed_dim,
            activation="relu"
        )

        # --------------------------------------------------
        # Normalize projected features
        # --------------------------------------------------

        self.image_norm = (
            tf.keras.layers.LayerNormalization()
        )


    def call(
            self,
            image_features
    ):

        # --------------------------------------------------
        # image_features:
        # (batch, 7, 7, 1280)
        # --------------------------------------------------

        image_features = self.image_projection(
            image_features
        )

        # --------------------------------------------------
        # (batch, 7, 7, 256)
        # --------------------------------------------------

        image_features = self.image_norm(
            image_features
        )

        # --------------------------------------------------
        # Convert feature map into sequence
        #
        # (batch, 7, 7, 256)
        #        ↓
        # (batch, 49, 256)
        # --------------------------------------------------

        batch_size = tf.shape(
            image_features
        )[0]

        image_features = tf.reshape(

            image_features,

            (
                batch_size,
                -1,
                image_features.shape[-1]
            )

        )

        return image_features


if __name__ == "__main__":

    fusion = FusionLayer()

    image_features = tf.random.normal(
        (
            1,
            7,
            7,
            1280
        )
    )

    output = fusion(
        image_features
    )

    print(
        "Fusion output shape:",
        output.shape
    )