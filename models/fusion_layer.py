import tensorflow as tf


class VisualTransformerBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        dropout_rate=0.1
    ):

        super().__init__()

        # ==========================================
        # VISUAL SELF ATTENTION
        # ==========================================

        self.self_attention = (
            tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads
            )
        )

        # ==========================================
        # FEED FORWARD NETWORK
        # ==========================================

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

        # ==========================================
        # NORMALIZATION
        # ==========================================

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

        # ==========================================
        # DROPOUT
        # ==========================================

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


    def call(
        self,
        inputs,
        training=False
    ):

        # ==========================================
        # VISUAL SELF ATTENTION
        # ==========================================

        attention_output = self.self_attention(

            query=inputs,

            key=inputs,

            value=inputs,

            training=training

        )

        attention_output = self.dropout1(

            attention_output,

            training=training

        )

        x = self.norm1(

            inputs + attention_output

        )

        # ==========================================
        # FEED FORWARD NETWORK
        # ==========================================

        ffn_output = self.ffn(
            x
        )

        ffn_output = self.dropout2(

            ffn_output,

            training=training

        )

        x = self.norm2(

            x + ffn_output

        )

        return x


class FusionLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        num_visual_layers=2,
        dropout_rate=0.1
    ):

        super().__init__()

        self.embed_dim = embed_dim

        # ==========================================
        # VISUAL FEATURE PROJECTION
        # ==========================================

        self.visual_projection = (
            tf.keras.layers.Dense(
                embed_dim
            )
        )

        # ==========================================
        # VISUAL NORMALIZATION
        # ==========================================

        self.visual_norm = (
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6
            )
        )

        # ==========================================
        # VISUAL TRANSFORMER BLOCKS
        # ==========================================

        self.visual_blocks = [

            VisualTransformerBlock(

                embed_dim=embed_dim,

                num_heads=num_heads,

                ff_dim=ff_dim,

                dropout_rate=dropout_rate

            )

            for _ in range(
                num_visual_layers
            )

        ]


    def call(
        self,
        image_features,
        training=False
    ):

        # ==========================================
        # HANDLE EFFICIENTNET FEATURE MAP
        #
        # Expected:
        #
        # (batch, 7, 7, 1280)
        #
        # Convert:
        #
        # (batch, 49, 1280)
        # ==========================================

        shape = tf.shape(
            image_features
        )

        batch_size = shape[0]

        feature_dim = shape[-1]


        image_features = tf.reshape(

            image_features,

            (
                batch_size,
                -1,
                feature_dim
            )

        )

        # ==========================================
        # PROJECT VISUAL FEATURES
        #
        # (batch, 49, 1280)
        #
        # →
        #
        # (batch, 49, 256)
        # ==========================================

        x = self.visual_projection(

            image_features

        )

        # ==========================================
        # NORMALIZE VISUAL TOKENS
        # ==========================================

        x = self.visual_norm(
            x
        )

        # ==========================================
        # VISUAL TRANSFORMER ENCODER
        # ==========================================

        for visual_block in self.visual_blocks:

            x = visual_block(

                x,

                training=training

            )

        return x


if __name__ == "__main__":

    fusion_layer = FusionLayer(

        embed_dim=256,

        num_visual_layers=2

    )

    sample_features = tf.random.normal(

        (
            1,
            7,
            7,
            1280
        )

    )

    result = fusion_layer(

        sample_features,

        training=False

    )

    print(
        "Input shape:",
        sample_features.shape
    )

    print(
        "Fusion output shape:",
        result.shape
    )