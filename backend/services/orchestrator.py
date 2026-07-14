from typing import Optional

from services.inference_service import vision_inference

from services.model_state import model_state

from services.request_router import (
    RequestMode,
    resolve_mode
)

from services.text_service import (
    generate_text_response
)


async def process_request(
    image_bytes: Optional[bytes],
    question: Optional[str]
) -> dict:


    clean_question = (
        question.strip()
        if question
        else None
    )


    mode = resolve_mode(
        bool(image_bytes),
        clean_question
    )


    if mode == RequestMode.TEXT:

        answer = await generate_text_response(
            clean_question
        )


    elif mode == RequestMode.DESCRIBE:

        answer = await vision_inference.describe(
            image_bytes
        )


    else:

        answer = await vision_inference.answer(
            image_bytes,
            clean_question
        )


    return {

        "success": True,

        "mode": mode.value,

        "question": clean_question,

        "answer": answer,

        "model_ready": model_state.loaded

    }