from typing import Optional

from inference.preprocess import (
    image_preprocessor
)

from inference.tokenizer_manager import (
    tokenizer_manager
)

from inference.predictor import (
    visiongpt_predictor
)

from inference.decoder import (
    answer_decoder
)


class VisionGPTEngine:


    async def predict(

        self,

        image_bytes: bytes,

        question: Optional[str]

    ) -> str:


        image_tensor = (
            image_preprocessor.preprocess(
                image_bytes
            )
        )


        question_tokens = None


        if question:


            question_tokens = (
                tokenizer_manager.encode(
                    question
                )
            )


        predicted_tokens = (

            await visiongpt_predictor.predict(

                image_tensor=image_tensor,

                question_tokens=question_tokens

            )

        )


        answer = answer_decoder.decode(
            predicted_tokens
        )


        return answer


visiongpt_engine = VisionGPTEngine()