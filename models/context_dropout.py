import tensorflow as tf


class ContextDropout(tf.keras.layers.Layer):

    def __init__(
        self,
        dropout_rate=0.20,
        mask_token_id=1,
        protected_token_ids=None,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )

        self.dropout_rate = dropout_rate

        self.mask_token_id = mask_token_id

        if protected_token_ids is None:

            protected_token_ids = [
                0,
                mask_token_id
            ]

        self.protected_token_ids = (
            protected_token_ids
        )


    # =====================================================
    # APPLY CONTEXT DROPOUT
    # =====================================================

    def call(
        self,
        tokens,
        training=False
    ):

        if not training:

            return tokens


        tokens = tf.convert_to_tensor(
            tokens
        )


        # =================================================
        # RANDOM DROPOUT MASK
        # =================================================

        random_values = tf.random.uniform(

            shape=tf.shape(tokens),

            minval=0.0,

            maxval=1.0,

            dtype=tf.float32

        )


        dropout_mask = (

            random_values

            <

            self.dropout_rate

        )


        # =================================================
        # PROTECT SPECIAL TOKENS
        # =================================================

        protected_mask = tf.zeros_like(

            tokens,

            dtype=tf.bool

        )


        for token_id in self.protected_token_ids:

            protected_mask = tf.logical_or(

                protected_mask,

                tf.equal(
                    tokens,
                    token_id
                )

            )


        dropout_mask = tf.logical_and(

            dropout_mask,

            tf.logical_not(
                protected_mask
            )

        )


        # =================================================
        # REPLACE TOKENS WITH MASK TOKEN
        # =================================================

        mask_tokens = tf.fill(

            tf.shape(tokens),

            tf.cast(
                self.mask_token_id,
                tokens.dtype
            )

        )


        output_tokens = tf.where(

            dropout_mask,

            mask_tokens,

            tokens

        )


        return output_tokens


    def get_config(
        self
    ):

        config = super().get_config()


        config.update({

            "dropout_rate":
                self.dropout_rate,

            "mask_token_id":
                self.mask_token_id,

            "protected_token_ids":
                self.protected_token_ids

        })


        return config


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    tf.random.set_seed(
        42
    )


    layer = ContextDropout(

        dropout_rate=0.50,

        mask_token_id=1,

        protected_token_ids=[
            0,
            1,
            3,
            4
        ]

    )


    tokens = tf.constant(

        [
            [
                3,
                10,
                20,
                30,
                40,
                4,
                0,
                0
            ],

            [
                3,
                50,
                60,
                70,
                80,
                4,
                0,
                0
            ]

        ],

        dtype=tf.int64

    )


    print(
        "Original tokens:"
    )


    print(
        tokens.numpy()
    )


    training_output = layer(

        tokens,

        training=True

    )


    print(
        "\nTraining output:"
    )


    print(
        training_output.numpy()
    )


    inference_output = layer(

        tokens,

        training=False

    )


    print(
        "\nInference output:"
    )


    print(
        inference_output.numpy()
    )


    assert tf.reduce_all(

        tf.equal(

            inference_output,

            tokens

        )

    )


    assert tf.reduce_all(

        tf.equal(

            training_output[:, 0],

            tokens[:, 0]

        )

    )


    print(
        "\nContextDropout test successful"
    )