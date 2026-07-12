import tensorflow as tf


class TextEncoder(tf.keras.layers.Layer):

    def __init__(
            self,
            vocab_size=10000,
            embedding_dim=256
    ):

        super().__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True
        )


    def call(
            self,
            tokens
    ):

        x = self.embedding(
            tokens
        )

        return x


    def encode(
            self,
            tokens
    ):

        return self(
            tokens
        )


if __name__ == "__main__":

    encoder = TextEncoder()

    sample_text = tf.constant(
        [
            [
                10,
                20,
                30,
                40,
                0,
                0
            ]
        ]
    )

    output = encoder(
        sample_text
    )

    print(
        "Text features shape:",
        output.shape
    )