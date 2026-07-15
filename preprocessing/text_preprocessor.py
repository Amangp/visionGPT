import json
import random
import tensorflow as tf


class TextPreprocessor:

    START_TOKEN = "startseq"

    END_TOKEN = "endseq"

    ANSWER_TOKEN = "answerseq"

    TASK_TOKENS = {

        "caption": "taskcaption",

        "ocr": "taskocr",

        "vqa": "taskvqa",

        "reasoning": "taskreasoning",

        "visual_grounding": "taskgrounding"

    }


    def __init__(
        self,
        vocab_size=10000,
        max_length=50
    ):

        self.vocab_size = vocab_size

        self.max_length = max_length


        self.vectorizer = (

            tf.keras.layers.TextVectorization(

                max_tokens=vocab_size,

                output_sequence_length=max_length,

                standardize="lower_and_strip_punctuation"

            )

        )


    # =====================================================
    # BUILD TRAINING TEXT
    # =====================================================

    def build_training_text(
    self,
    question,
    answer,
    task
    ):

        task_token = self.TASK_TOKENS.get(
            task,
            "taskvqa"
        )

        if task == "caption":

            caption_prompts = [

                "describe the image",

                "what is happening",

                "what do you see",

                "describe this scene",

                "write a caption",

                "what is shown in the image",

                "describe everything visible",

                "describe the photograph"

            ]

            random.seed(
                hash(answer) & 0xffffffff
            )

            question = random.choice(
                caption_prompts
            )

        question = str(question).strip()

        answer = str(answer).strip()

        return (

            f"{self.START_TOKEN} "

            f"{task_token} "

            f"{question} "

            f"{self.ANSWER_TOKEN} "

            f"{answer} "

            f"{self.END_TOKEN}"

        )

    # =====================================================
    # BUILD DATASET TEXTS
    # =====================================================

    def build_dataset_texts(
        self,
        samples
    ):

        texts = []


        for sample in samples:

            text = self.build_training_text(

                question=sample.question,

                answer=sample.answer,

                task=sample.task

            )


            texts.append(
                text
            )


        return texts


    # =====================================================
    # BUILD VOCABULARY
    # =====================================================

    def build_vocabulary(
        self,
        texts
    ):

        print(
            "\nBuilding VisionGPT v4 vocabulary..."
        )


        self.vectorizer.adapt(
            texts
        )


        print(
            "Vocabulary size:",
            len(
                self.vectorizer.get_vocabulary()
            )
        )


        self._validate_special_tokens()


    # =====================================================
    # PROCESS TEXT
    # =====================================================

    def process(
        self,
        texts
    ):

        return self.vectorizer(
            texts
        )


    # =====================================================
    # TOKEN ID
    # =====================================================

    def get_token_id(
        self,
        token
    ):

        vocabulary = (

            self.vectorizer
            .get_vocabulary()

        )


        try:

            return vocabulary.index(
                token
            )


        except ValueError:

            raise ValueError(

                f"Token not found in vocabulary: "
                f"{token}"

            )


    # =====================================================
    # SPECIAL TOKEN IDS
    # =====================================================

    def get_special_token_ids(
        self
    ):

        return {

            "start_token_id":

                self.get_token_id(
                    self.START_TOKEN
                ),

            "end_token_id":

                self.get_token_id(
                    self.END_TOKEN
                ),

            "answer_token_id":

                self.get_token_id(
                    self.ANSWER_TOKEN
                ),

            "task_caption_token_id":

                self.get_token_id(
                    self.TASK_TOKENS["caption"]
                ),

            "task_ocr_token_id":

                self.get_token_id(
                    self.TASK_TOKENS["ocr"]
                )

        }


    # =====================================================
    # VALIDATE SPECIAL TOKENS
    # =====================================================

    def _validate_special_tokens(
        self
    ):

        required_tokens = [

            self.START_TOKEN,

            self.END_TOKEN,

            self.ANSWER_TOKEN,

            self.TASK_TOKENS["caption"],

            self.TASK_TOKENS["ocr"]

        ]


        vocabulary = (

            self.vectorizer
            .get_vocabulary()

        )


        missing_tokens = [

            token

            for token in required_tokens

            if token not in vocabulary

        ]


        if missing_tokens:

            raise ValueError(

                "Missing VisionGPT v4 special tokens: "

                +

                ", ".join(
                    missing_tokens
                )

            )


        print(
            "VisionGPT v4 special tokens verified"
        )


    # =====================================================
    # SAVE VOCABULARY
    # =====================================================

    def save_vocab(
        self,
        path="vocab_v4.json"
    ):

        vocab = (

            self.vectorizer
            .get_vocabulary()

        )


        with open(

            path,

            "w",

            encoding="utf-8"

        ) as file:


            json.dump(

                vocab,

                file,

                ensure_ascii=False,

                indent=2

            )


        print(
            "Vocabulary saved:",
            path
        )


    # =====================================================
    # LOAD VOCABULARY
    # =====================================================

    def load_vocab(
        self,
        path="vocab_v4.json"
    ):

        with open(

            path,

            "r",

            encoding="utf-8"

        ) as file:


            vocab = json.load(
                file
            )


        self.vectorizer.set_vocabulary(
            vocab
        )


        self._validate_special_tokens()


    # =====================================================
    # DECODE
    # =====================================================

    def decode(
        self,
        tokens
    ):

        vocab = (

            self.vectorizer
            .get_vocabulary()

        )


        words = []


        special_tokens = {

            "",

            "[UNK]",

            self.START_TOKEN,

            self.END_TOKEN,

            self.ANSWER_TOKEN,

            *self.TASK_TOKENS.values()

        }


        for token in tokens:

            token = int(
                token
            )


            if (

                token < 0

                or

                token >= len(vocab)

            ):

                continue


            word = vocab[
                token
            ]


            if word not in special_tokens:

                words.append(
                    word
                )


        return " ".join(
            words
        )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    processor = TextPreprocessor(

        vocab_size=10000,

        max_length=50

    )


    sample_texts = [

        processor.build_training_text(

            question="describe the image",

            answer=(
                "a dog standing on a beach"
            ),

            task="caption"

        ),

        processor.build_training_text(

            question=(
                "what is the brand of phone"
            ),

            answer="nokia",

            task="ocr"

        )

    ]


    print(
        "\nTraining samples:"
    )


    for text in sample_texts:

        print(
            text
        )


    processor.build_vocabulary(
        sample_texts
    )


    tokens = processor.process(
        sample_texts
    )


    print(
        "\nTokens:"
    )


    print(
        tokens.numpy()
    )


    print(
        "\nSpecial token IDs:"
    )


    print(
        processor.get_special_token_ids()
    )


    print(
        "\nDecoded:"
    )


    print(
        processor.decode(
            tokens[0].numpy()
        )
    )


    print(
        "\nTextPreprocessor v4 test successful"
    )