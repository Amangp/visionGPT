import tensorflow as tf
import json



class TextPreprocessor:


    def __init__(
            self,
            vocab_size=10000,
            max_length=30
    ):


        self.vocab_size = vocab_size

        self.max_length = max_length


        self.vectorizer = (
            tf.keras.layers.TextVectorization(
                max_tokens=vocab_size,
                output_sequence_length=max_length
            )
        )



    def build_vocabulary(
            self,
            texts
    ):


        self.vectorizer.adapt(
            texts
        )



    def process(
            self,
            texts
    ):


        return self.vectorizer(
            texts
        )



    def save_vocab(
            self,
            path="vocab.json"
    ):


        vocab = (
            self.vectorizer
            .get_vocabulary()
        )


        with open(
            path,
            "w"
        ) as file:


            json.dump(
                vocab,
                file
            )



    def load_vocab(
            self,
            path="vocab.json"
    ):


        with open(
            path,
            "r"
        ) as file:


            vocab = json.load(
                file
            )


        self.vectorizer.set_vocabulary(
            vocab
        )



    def decode(self, tokens):

        vocab = self.vectorizer.get_vocabulary()

        words = []

        special_tokens = {
            "",
            "[UNK]",
            "startseq",
            "endseq"
        }

        for token in tokens:

            token = int(token)

            if token < 0 or token >= len(vocab):
                continue

            word = vocab[token]

            if word not in special_tokens:
                words.append(word)

        return " ".join(words)