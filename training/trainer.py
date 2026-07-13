import tensorflow as tf


# =========================================================
# MASKED LOSS
# =========================================================

def masked_loss(
        y_true,
        y_pred
):

    loss_function = (
        tf.keras.losses
        .SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="none"
        )
    )


    loss = loss_function(
        y_true,
        y_pred
    )


    mask = tf.cast(

        tf.not_equal(
            y_true,
            0
        ),

        dtype=loss.dtype

    )


    loss = loss * mask


    return (

        tf.reduce_sum(loss)

        /

        tf.maximum(
            tf.reduce_sum(mask),
            1.0
        )

    )


# =========================================================
# MASKED ACCURACY
# =========================================================

def masked_accuracy(
        y_true,
        y_pred
):

    predicted_tokens = tf.argmax(

        y_pred,

        axis=-1

    )


    predicted_tokens = tf.cast(

        predicted_tokens,

        y_true.dtype

    )


    matches = tf.equal(

        y_true,

        predicted_tokens

    )


    mask = tf.not_equal(

        y_true,

        0

    )


    matches = tf.logical_and(

        matches,

        mask

    )


    matches = tf.cast(

        matches,

        tf.float32

    )


    mask = tf.cast(

        mask,

        tf.float32

    )


    return (

        tf.reduce_sum(matches)

        /

        tf.maximum(
            tf.reduce_sum(mask),
            1.0
        )

    )


# =========================================================
# TRAINER
# =========================================================

class Trainer:


    def __init__(
            self,
            model,
            learning_rate=0.0001
    ):


        self.model = model


        self.optimizer = (

            tf.keras.optimizers.Adam(
                learning_rate
            )

        )


    def compile(self):


        self.model.compile(

            optimizer=self.optimizer,

            loss=masked_loss,

            metrics=[
                masked_accuracy
            ]

        )


    def train(
            self,
            dataset,
            epochs=10
    ):


        history = self.model.fit(

            dataset,

            epochs=epochs

        )


        return history