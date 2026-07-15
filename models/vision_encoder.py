import tensorflow as tf


class VisionEncoder(tf.keras.layers.Layer):

    def __init__(
        self,
        embed_dim=256,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.embed_dim = embed_dim

        self.model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            pooling=None
        )

        self.freeze_backbone()


    def freeze_backbone(self):

        self.model.trainable = False

        for layer in self.model.layers:
            layer.trainable = False


    def enable_fine_tuning(
        self,
        unfreeze_last_n=30
    ):

        self.model.trainable = True

        total_layers = len(self.model.layers)
        unfreeze_from = max(
            total_layers - unfreeze_last_n,
            0
        )

        for index, layer in enumerate(
            self.model.layers
        ):

            layer.trainable = (
                index >= unfreeze_from
            )

            if isinstance(
                layer,
                tf.keras.layers.BatchNormalization
            ):

                layer.trainable = False

        trainable_layers = sum(
            layer.trainable
            for layer in self.model.layers
        )

        print(
            "EfficientNet total layers:",
            total_layers
        )

        print(
            "EfficientNet trainable layers:",
            trainable_layers
        )


    def call(
        self,
        image,
        training=False
    ):

        return self.model(
            image,
            training=False
        )


    def encode(
        self,
        image,
        training=False
    ):

        return self(
            image,
            training=training
        )


    def get_config(self):

        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim
        })

        return config


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
        sample_image,
        training=False
    )

    print(
        "Vision Encoder output shape:",
        output.shape
    )

    encoder.enable_fine_tuning(
        unfreeze_last_n=30
    )
