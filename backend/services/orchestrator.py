from typing import Optional


from services.inference_service import (
    vision_inference
)

from services.model_state import (
    model_state
)

from services.request_router import (
    RequestMode,
    resolve_mode
)

from services.session_service import (
    session_service
)

from services.text_service import (
    generate_text_response
)


async def process_request(

    image_bytes: Optional[bytes],

    question: Optional[str],

    session_id: Optional[str]

) -> dict:


    clean_question = (

        question.strip()

        if question

        else None

    )


    session = (

        session_service
        .get_or_create_session(
            session_id
        )

    )


    if image_bytes:

        session_service.set_image(

            session=session,

            image_bytes=image_bytes

        )


    active_image = (
        session.image_bytes
    )


    mode = resolve_mode(

        bool(active_image),

        clean_question

    )


    if clean_question:

        session_service.add_message(

            session=session,

            role="user",

            content=clean_question

        )


    if mode == RequestMode.TEXT:

        answer = (

            await generate_text_response(

                clean_question

            )

        )


    elif mode == RequestMode.DESCRIBE:

        answer = (

            await vision_inference.describe(

                active_image

            )

        )


    else:

        answer = (

            await vision_inference.answer(

                active_image,

                clean_question

            )

        )


    session_service.add_message(

        session=session,

        role="assistant",

        content=answer

    )


    return {

        "success": True,

        "session_id": session.session_id,

        "mode": mode.value,

        "question": clean_question,

        "answer": answer,

        "model_ready": model_state.loaded,

        "image_in_session": (
            session.image_bytes
            is not None
        )

    }