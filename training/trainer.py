import tensorflow as tf


# =========================================================
# TOKEN LOSS
# =========================================================

def token_loss(
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


    return loss_function(
        y_true,
        y_pred
    )


# =========================================================
# TOKEN ACCURACY
# =========================================================

def token_accuracy(
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


    return tf.cast(

        matches,

        tf.float32

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

                learning_rate=learning_rate

            )

        )


    # =====================================================
    # COMPILE
    # =====================================================

    def compile(
        self
    ):


        self.model.compile(

            optimizer=self.optimizer,

            loss=token_loss,

            weighted_metrics=[

                token_accuracy

            ]

        )


    # =====================================================
    # TRAIN
    # =====================================================

    def train(
        self,
        dataset,
        validation_data=None,
        epochs=10,
        callbacks=None
    ):


        history = self.model.fit(

            dataset,

            validation_data=validation_data,

            epochs=epochs,

            callbacks=callbacks

        )


        return history