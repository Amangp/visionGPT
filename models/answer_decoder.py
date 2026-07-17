import tensorflow as tf


# =========================================================
# TRANSFORMER DECODER BLOCK
# =========================================================

class TransformerDecoderBlock(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        dropout_rate=0.1
    ):

        super().__init__()


        # =================================================
        # MASKED SELF ATTENTION
        # =================================================

        self.self_attention = (
            tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads
            )
        )


        # =================================================
        # IMAGE CROSS ATTENTION
        # =================================================

        self.cross_attention = (
            tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads
            )
        )


        # =================================================
        # FEED FORWARD NETWORK
        # =================================================

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    ff_dim,
                    activation="gelu"
                ),

                tf.keras.layers.Dense(
                    embed_dim
                )
            ]
        )


        # =================================================
        # NORMALIZATION
        # =================================================

        self.norm1 = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6
            )
        )


        self.norm2 = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6
            )
        )


        self.norm3 = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6
            )
        )


        # =================================================
        # DROPOUT
        # =================================================

        self.dropout1 = (
            tf.keras.layers.Dropout(
                dropout_rate
            )
        )


        self.dropout2 = (
            tf.keras.layers.Dropout(
                dropout_rate
            )
        )


        self.dropout3 = (
            tf.keras.layers.Dropout(
                dropout_rate
            )
        )
        self.embedding_dropout = tf.keras.layers.Dropout(
            dropout_rate
        )

    # =====================================================
    # CALL
    # =====================================================

    def call(
        self,
        x,
        image_features,
        causal_mask=None,
        padding_mask=None,
        training=False
    ):


        # =================================================
        # MASKED SELF ATTENTION
        # =================================================

        self_attention_output = (
            self.self_attention(

                query=x,

                key=x,

                value=x,

                attention_mask=causal_mask,

                training=training

            )
        )


        self_attention_output = self.dropout1(

            self_attention_output,

            training=training

        )


        x = self.norm1(

            x + self_attention_output

        )


        # =================================================
        # IMAGE CROSS ATTENTION
        #
        # Text queries visual tokens.
        # =================================================

        cross_attention_output = (
            self.cross_attention(

                query=x,

                key=image_features,

                value=image_features,

                training=training

            )
        )


        cross_attention_output = self.dropout2(

            cross_attention_output,

            training=training

        )


        x = self.norm2(

            x + cross_attention_output

        )


        # =================================================
        # FEED FORWARD NETWORK
        # =================================================

        ffn_output = self.ffn(
            x
        )


        ffn_output = self.dropout3(

            ffn_output,

            training=training

        )


        x = self.norm3(

            x + ffn_output

        )


        return x


# =========================================================
# ANSWER DECODER
# =========================================================

class AnswerDecoder(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        vocab_size=10000,
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        dropout_rate=0.1,
        max_length=100,
        num_decoder_layers=3
    ):

        super().__init__()


        self.vocab_size = vocab_size

        self.embed_dim = embed_dim

        self.max_length = max_length

        self.num_decoder_layers = (
            num_decoder_layers
        )


        # =================================================
        # TOKEN EMBEDDING
        # =================================================

        self.embedding = (
            tf.keras.layers.Embedding(

                input_dim=vocab_size,

                output_dim=embed_dim,

                mask_zero=True

            )
        )


        # =================================================
        # POSITION EMBEDDING
        # =================================================

        self.position_embedding = (
            tf.keras.layers.Embedding(

                input_dim=max_length,

                output_dim=embed_dim

            )
        )

        self.embedding_dropout = (
            tf.keras.layers.Dropout(
                dropout_rate
            )
        )


        # =================================================
        # DECODER BLOCKS
        # =================================================

        self.decoder_blocks = [

            TransformerDecoderBlock(

                embed_dim=embed_dim,

                num_heads=num_heads,

                ff_dim=ff_dim,

                dropout_rate=dropout_rate

            )

            for _ in range(
                num_decoder_layers
            )

        ]


        # =================================================
        # FINAL NORMALIZATION
        # =================================================

        self.final_norm = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6
            )
        )


        # =================================================
        # OUTPUT VOCABULARY LAYER
        # =================================================

        self.output_layer = (
            tf.keras.layers.Dense(
                vocab_size,
                dtype="float32"
            )
        )


    # =====================================================
    # CREATE CAUSAL MASK
    # =====================================================

    def create_causal_mask(
        self,
        sequence_length
    ):

        mask = tf.linalg.band_part(

            tf.ones(

                (
                    sequence_length,
                    sequence_length
                ),

                dtype=tf.bool

            ),

            -1,

            0

        )


        return mask


    # =====================================================
    # CALL
    # =====================================================

    def call(
        self,
        text_tokens,
        image_features,
        training=False
    ):


        # =================================================
        # SEQUENCE LENGTH
        # =================================================

        sequence_length = tf.shape(
            text_tokens
        )[1]
        padding_mask = tf.not_equal(
            text_tokens,
            0
        )

        padding_mask = padding_mask[
            :,
            tf.newaxis,
            :
        ]

        # =================================================
        # TOKEN EMBEDDING
        # =================================================



        token_embeddings = self.embedding(text_tokens)

        scale = tf.cast(
            tf.math.sqrt(tf.cast(self.embed_dim, tf.float32)),
            token_embeddings.dtype,
        )

        token_embeddings = token_embeddings * scale

        positions = tf.range(
            start=0,
            limit=sequence_length,
            delta=1
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = (
            token_embeddings
            +
            position_embeddings
        )

        x = self.embedding_dropout(
            x,
            training=training
        )


        # =================================================
        # CAUSAL MASK
        # =================================================

        causal_mask = self.create_causal_mask(
            sequence_length
        )

        causal_mask = causal_mask[
            tf.newaxis,
            :,
            :
        ]

        attention_mask = tf.logical_and(
            causal_mask,
            padding_mask
        )


        # =================================================
        # RUN 3 TRANSFORMER DECODER BLOCKS
        # =================================================

        for decoder_block in self.decoder_blocks:

            x = decoder_block(
                x,
                image_features,
                attention_mask,
                training=training
            )


        # =================================================
        # FINAL NORMALIZATION
        # =================================================

        x = self.final_norm(
            x
        )


        # =================================================
        # VOCABULARY LOGITS
        #
        # IMPORTANT:
        # No softmax here.
        #
        # SparseCategoricalCrossentropy should use
        # from_logits=True.
        # =================================================

        output = self.output_layer(
            x
        )


        return output


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":


    print(
        "\n=========================================="
    )


    print(
        "Testing VisionGPT v2.5 AnswerDecoder"
    )


    print(
        "=========================================="
    )


    decoder = AnswerDecoder(

        vocab_size=10000,

        embed_dim=256,

        num_decoder_layers=3

    )


    sample_text = tf.constant(

        [
            [
                10,
                20,
                30,
                40,
                0
            ]
        ],

        dtype=tf.int64

    )


    sample_image = tf.random.normal(

        (
            1,
            49,
            256
        )

    )


    result = decoder(

        sample_text,

        sample_image,

        training=False

    )


    print(
        "\nDecoder layers:",
        decoder.num_decoder_layers
    )


    print(
        "Text shape:",
        sample_text.shape
    )


    print(
        "Image feature shape:",
        sample_image.shape
    )


    print(
        "Decoder output shape:",
        result.shape
    )


    print(
        "\nAnswerDecoder test successful"
    )