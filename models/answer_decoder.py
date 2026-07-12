import tensorflow as tf


class AnswerDecoder(tf.keras.layers.Layer):

    def __init__(
            self,
            vocab_size=10000,
            embed_dim=256,
            num_heads=4,
            ff_dim=512,
            dropout_rate=0.1,
            max_length=100
    ):

        super().__init__()

        # --------------------------------------------------
        # Token Embedding
        # --------------------------------------------------

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True
        )

        # --------------------------------------------------
        # Positional Embedding
        # --------------------------------------------------

        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=max_length,
            output_dim=embed_dim
        )

        # --------------------------------------------------
        # Self Attention
        # --------------------------------------------------

        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )

        # --------------------------------------------------
        # Cross Attention
        # --------------------------------------------------

        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )

        # --------------------------------------------------
        # Feed Forward Network
        # --------------------------------------------------

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(
                ff_dim,
                activation="relu"
            ),
            tf.keras.layers.Dense(
                embed_dim
            )
        ])

        # --------------------------------------------------
        # Layer Normalization
        # --------------------------------------------------

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()

        # --------------------------------------------------
        # Dropout
        # --------------------------------------------------

        self.dropout1 = tf.keras.layers.Dropout(
            dropout_rate
        )

        self.dropout2 = tf.keras.layers.Dropout(
            dropout_rate
        )

        self.dropout3 = tf.keras.layers.Dropout(
            dropout_rate
        )

        # --------------------------------------------------
        # Vocabulary Prediction
        # --------------------------------------------------

        self.output_layer = tf.keras.layers.Dense(
            vocab_size,
            activation="softmax"
        )

    # ======================================================
    # FORWARD PASS
    # ======================================================

    def call(
            self,
            text_tokens,
            image_features,
            training=False
    ):

        # -----------------------------------------------
        # Token Embedding
        # -----------------------------------------------

        x = self.embedding(
            text_tokens
        )

        # -----------------------------------------------
        # Positional Embedding
        # -----------------------------------------------

        seq_len = tf.shape(text_tokens)[1]

        positions = tf.range(
            seq_len
        )

        pos_embed = self.position_embedding(
            positions
        )

        x = x + pos_embed

        # -----------------------------------------------
        # Padding Mask
        # Shape -> (batch, seq_len)
        # -----------------------------------------------

        padding_mask = tf.not_equal(
            text_tokens,
            0
        )

        # Shape -> (batch,1,seq_len)

        padding_mask = padding_mask[
            :,
            tf.newaxis,
            :
        ]

        # -----------------------------------------------
        # Causal Mask
        # Shape -> (seq_len, seq_len)
        # -----------------------------------------------

        causal_mask = tf.linalg.band_part(

            tf.ones(
                (
                    seq_len,
                    seq_len
                ),
                dtype=tf.bool
            ),

            -1,

            0

        )

        causal_mask = causal_mask[
            tf.newaxis,
            :,
            :
        ]

        # -----------------------------------------------
        # Combine Masks
        # -----------------------------------------------

        attention_mask = tf.logical_and(
            causal_mask,
            padding_mask
        )

        # -----------------------------------------------
        # Self Attention
        # -----------------------------------------------

        attn = self.self_attention(

            query=x,

            value=x,

            key=x,

            attention_mask=attention_mask,

            training=training

        )

        attn = self.dropout1(
            attn,
            training=training
        )

        x = self.norm1(
            x + attn
        )

        # -----------------------------------------------
        # Cross Attention
        # -----------------------------------------------

        cross = self.cross_attention(

            query=x,

            value=image_features,

            key=image_features,

            training=training

        )

        cross = self.dropout2(
            cross,
            training=training
        )

        x = self.norm2(
            x + cross
        )

        # -----------------------------------------------
        # Feed Forward
        # -----------------------------------------------

        ffn = self.ffn(
            x
        )

        ffn = self.dropout3(
            ffn,
            training=training
        )

        x = self.norm3(
            x + ffn
        )

        # -----------------------------------------------
        # Vocabulary Prediction
        # -----------------------------------------------

        output = self.output_layer(
            x
        )

        return output


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    decoder = AnswerDecoder()

    sample_text = tf.constant([
        [10, 20, 30, 40, 0]
    ])

    sample_image = tf.random.normal(
        (
            1,
            49,
            256
        )
    )

    result = decoder(
        sample_text,
        sample_image
    )

    print("Decoder output shape:", result.shape)