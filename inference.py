import tensorflow as tf

from models.visiongpt import VisionGPT

from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.text_preprocessor import TextPreprocessor


class VisionGPTInference:

    def __init__(
            self,
            model_path,
            vocab_path="vocab.json"
    ):

        # ====================================================
        # LOAD VOCAB FIRST
        # ====================================================

        self.text_processor = TextPreprocessor()

        self.text_processor.load_vocab(
            vocab_path
        )

        self.vocab = (
            self.text_processor
            .vectorizer
            .get_vocabulary()
        )

        self.vocab_size = len(
            self.vocab
        )

        print(
            f"Vocabulary size: {self.vocab_size}"
        )

        # ====================================================
        # FIND SPECIAL TOKENS
        # ====================================================

        if "startseq" not in self.vocab:
            raise ValueError(
                "startseq token not found."
            )

        if "endseq" not in self.vocab:
            raise ValueError(
                "endseq token not found."
            )

        self.start_token = self.vocab.index(
            "startseq"
        )

        self.end_token = self.vocab.index(
            "endseq"
        )

        # ====================================================
        # CREATE MODEL
        # ====================================================

        self.model = VisionGPT(
            vocab_size=self.vocab_size
        )

        # ====================================================
        # BUILD MODEL
        # ====================================================

        dummy_image = tf.random.normal(
            (
                1,
                224,
                224,
                3
            )
        )

        dummy_text = tf.ones(
            (
                1,
                10
            ),
            dtype=tf.int32
        )

        self.model(
            (
                dummy_image,
                dummy_text
            ),
            training=False
        )

        # ====================================================
        # LOAD WEIGHTS
        # ====================================================

        self.model.load_weights(
            model_path
        )

        print(
            "VisionGPT model loaded successfully."
        )

        print(
            f"Start Token: {self.start_token}"
        )

        print(
            f"End Token: {self.end_token}"
        )

        # ====================================================
        # IMAGE PROCESSOR
        # ====================================================

        self.image_processor = (
            ImagePreprocessor()
        )

    # ========================================================
    # GENERATE CAPTION
    # ========================================================

    def generate(
            self,
            image_path,
            max_length=30
    ):

        image = (
            self.image_processor.process(
                image_path
            )
        )

        image = tf.expand_dims(
            image,
            axis=0
        )

        generated = [
            self.start_token
        ]

        for _ in range(max_length):

            tokens = tf.constant(
                [
                    generated
                ],
                dtype=tf.int32
            )

            predictions = self.model(
                (
                    image,
                    tokens
                ),
                training=False
            )

            next_token = tf.argmax(
                predictions[0, -1]
            ).numpy()

            next_token = int(
                next_token
            )

            if (
                next_token == self.end_token
            ):
                break

            if next_token == 0:
                break

            generated.append(
                next_token
            )

        sentence = (
            self.text_processor.decode(
                generated
            )
        )

        return sentence


if __name__ == "__main__":

    bot = VisionGPTInference(

        model_path=(
            "checkpoints/"
            "visiongpt_v2_2026_07_12_21_08.weights.h5"
        ),

        vocab_path="vocab.json"

    )

    result = bot.generate(
        "test1.jpg"
    )

    print("\nGenerated Caption:\n")

    print(result)