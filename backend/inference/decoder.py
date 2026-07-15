from inference.tokenizer_manager import (
    tokenizer_manager
)


class AnswerDecoder:

    SPECIAL_TOKENS = {

        "<pad>",
        "<start>",
        "<end>"

    }


    def decode(
        self,
        token_ids: list[int]
    ) -> str:


        words = []


        for token_id in token_ids:


            word = (
                tokenizer_manager
                .token_to_word(
                    int(token_id)
                )
            )


            if word == "<end>":

                break


            if word in self.SPECIAL_TOKENS:

                continue


            words.append(word)


        return " ".join(words).strip()


answer_decoder = AnswerDecoder()