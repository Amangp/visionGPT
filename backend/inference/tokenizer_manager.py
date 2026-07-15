from pathlib import Path

import json

from core.errors import VisionGPTError


class TokenizerManager:

    def __init__(self):

        self.word_to_index = {}

        self.index_to_word = {}

        self.loaded = False


    def load(
        self,
        tokenizer_path: str
    ) -> None:

        path = Path(tokenizer_path)


        if not path.exists():

            raise VisionGPTError(

                message=(
                    f"Tokenizer not found: "
                    f"{tokenizer_path}"
                ),

                code="TOKENIZER_NOT_FOUND",

                status_code=503

            )


        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            tokenizer_data = json.load(file)


        self.word_to_index = (
            tokenizer_data.get(
                "word_to_index",
                {}
            )
        )


        self.index_to_word = {

            int(index): word

            for index, word
            in tokenizer_data.get(
                "index_to_word",
                {}
            ).items()

        }


        self.loaded = True


    def encode(
        self,
        text: str
    ) -> list[int]:

        if not self.loaded:

            raise VisionGPTError(

                message="Tokenizer is not loaded.",

                code="TOKENIZER_NOT_READY",

                status_code=503

            )


        words = text.lower().strip().split()


        token_ids = [

            self.word_to_index.get(
                word,
                self.word_to_index.get(
                    "<unk>",
                    0
                )
            )

            for word in words

        ]


        return token_ids


    def token_to_word(
        self,
        token_id: int
    ) -> str:

        return self.index_to_word.get(
            token_id,
            "<unk>"
        )


tokenizer_manager = TokenizerManager()