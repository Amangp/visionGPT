import tensorflow as tf
import os

from models.visiongpt import VisionGPT

from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.text_preprocessor import TextPreprocessor


class VisionGPTInference:

    def __init__(
            self,
            model_path,
            vocab_path="vocab.json"
    ):

        print(
            "\n=========================================="
        )

        print(
            "Loading VisionGPT v2.4"
        )

        print(
            "=========================================="
        )


        # =================================================
        # 1. LOAD VOCABULARY
        # =================================================

        print(
            "\n[1/4] Loading vocabulary..."
        )

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
            "Vocabulary size:",
            self.vocab_size
        )


        # =================================================
        # 2. FIND SPECIAL TOKENS
        # =================================================

        print(
            "\n[2/4] Finding special tokens..."
        )


        if "startseq" not in self.vocab:

            raise ValueError(
                "startseq token not found"
            )


        if "endseq" not in self.vocab:

            raise ValueError(
                "endseq token not found"
            )


        self.start_token = self.vocab.index(
            "startseq"
        )


        self.end_token = self.vocab.index(
            "endseq"
        )


        print(
            "Start token:",
            self.start_token
        )


        print(
            "End token:",
            self.end_token
        )


        # =================================================
        # 3. CREATE MODEL
        # =================================================

        print(
            "\n[3/4] Creating VisionGPT..."
        )


        self.model = VisionGPT(

            vocab_size=self.vocab_size

        )


        # Build model variables using same sequence length
        # used during training

        dummy_image = tf.zeros(

            (
                1,
                224,
                224,
                3
            ),

            dtype=tf.float32

        )


        dummy_text = tf.ones(

            (
                1,
                29
            ),

            dtype=tf.int64

        )


        self.model(

            (
                dummy_image,
                dummy_text
            ),

            training=False

        )


        print(
            "VisionGPT architecture created"
        )


        # =================================================
        # 4. LOAD WEIGHTS
        # =================================================

        print(
            "\n[4/4] Loading trained weights..."
        )


        if not os.path.exists(
            model_path
        ):

            raise FileNotFoundError(

                f"Model checkpoint not found: "
                f"{model_path}"

            )


        self.model.load_weights(
            model_path
        )


        print(
            "Weights loaded successfully"
        )


        # =================================================
        # IMAGE PROCESSOR
        # =================================================

        self.image_processor = (
            ImagePreprocessor()
        )


        print(
            "\nVisionGPT v2.4 ready 🚀"
        )


    # =====================================================
    # TOKEN DECODER
    # =====================================================

    def decode_tokens(
            self,
            token_ids
    ):

        words = []


        for token_id in token_ids:

            token_id = int(
                token_id
            )


            if token_id <= 0:

                continue


            if token_id >= len(
                self.vocab
            ):

                continue


            word = self.vocab[
                token_id
            ]


            if word in [

                "startseq",
                "endseq"

            ]:

                continue


            words.append(
                word
            )


        return " ".join(
            words
        )


    # =====================================================
    # CAPTION GENERATION
    # =====================================================

    def generate(
            self,
            image_path,
            max_length=29
    ):

        if not os.path.exists(
            image_path
        ):

            raise FileNotFoundError(

                f"Image not found: "
                f"{image_path}"

            )


        # =================================================
        # PROCESS IMAGE
        # =================================================

        image = self.image_processor.process(
            image_path
        )


        image = tf.expand_dims(

            image,

            axis=0

        )


        # =================================================
        # START GENERATION
        # =================================================

        generated = [

            self.start_token

        ]


        # =================================================
        # AUTOREGRESSIVE GENERATION
        # =================================================

        for _ in range(
            max_length - 1
        ):


            tokens = tf.constant(

                [
                    generated
                ],

                dtype=tf.int64

            )


            predictions = self.model(

                (
                    image,
                    tokens
                ),

                training=False

            )


            next_token_probs = predictions[

                0,

                -1,

                :

            ]


            next_token = tf.argmax(

                next_token_probs,

                axis=-1

            )


            next_token = int(

                next_token.numpy()

            )


            # =============================================
            # END TOKEN
            # =============================================

            if next_token == self.end_token:

                break


            # =============================================
            # PADDING TOKEN
            # =============================================

            if next_token == 0:

                break


            generated.append(
                next_token
            )


        caption = self.decode_tokens(
            generated
        )


        return caption


    # =====================================================
    # MULTIPLE IMAGE TEST
    # =====================================================

    def test_images(
            self,
            image_paths
    ):

        print(
            "\n"
            "=========================================="
        )

        print(
            "VISIONGPT v2.4 IMAGE TEST"
        )

        print(
            "=========================================="
        )


        results = {}


        for image_path in image_paths:

            print(
                "\n------------------------------------------"
            )


            print(
                "Image:",
                image_path
            )


            try:

                caption = self.generate(
                    image_path
                )


                results[
                    image_path
                ] = caption


                print(
                    "Caption:"
                )


                print(
                    caption
                )


            except Exception as error:

                print(
                    "ERROR:"
                )

                print(
                    error
                )


        print(
            "\n"
            "=========================================="
        )

        print(
            "TEST COMPLETED"
        )

        print(
            "=========================================="
        )


        return results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":


    MODEL_PATH = (

        
        "checkpoints/visiongpt_v2_5_best_2026_07_13_11_06.weights.h5"

    )


    VOCAB_PATH = (
        "vocab.json"
    )


    bot = VisionGPTInference(

        model_path=MODEL_PATH,

        vocab_path=VOCAB_PATH

    )


    TEST_IMAGES = [

        "test_images/tree.jpg",

        "test_images/dinosaur.jpg",

        "test_images/person.jpg",

        "test_images/car.jpg",

        "test_images/dog.jpg"

    ]


    bot.test_images(
        TEST_IMAGES
    )