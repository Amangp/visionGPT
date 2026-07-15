from typing import Optional

from inference.engine import visiongpt_engine

from services.model_state import model_state


class VisionInferenceService:


    async def describe(

        self,

        image_bytes: bytes

    ) -> str:


        if not model_state.loaded:


            return (

                "Image description route is ready. "

                "The trained VisionGPT checkpoint "

                "is not connected yet."

            )


        return await self.infer(

            image_bytes=image_bytes,

            question=None

        )


    async def answer(

        self,

        image_bytes: bytes,

        question: str

    ) -> str:


        if not model_state.loaded:


            return (

                f'Visual Q&A route is ready for: '

                f'"{question}". '

                f'The trained VisionGPT checkpoint '

                f'is not connected yet.'

            )


        return await self.infer(

            image_bytes=image_bytes,

            question=question

        )


    async def infer(

        self,

        image_bytes: bytes,

        question: Optional[str]

    ) -> str:


        return await visiongpt_engine.predict(

            image_bytes=image_bytes,

            question=question

        )


vision_inference = VisionInferenceService()