from enum import Enum

from typing import Optional

from core.errors import VisionGPTError


class RequestMode(str, Enum):

    TEXT = "text"

    DESCRIBE = "describe"

    VQA = "vqa"


def resolve_mode(
    has_image: bool,
    question: Optional[str]
) -> RequestMode:

    has_question = bool(
        question and question.strip()
    )


    if has_image and has_question:

        return RequestMode.VQA


    if has_image:

        return RequestMode.DESCRIBE


    if has_question:

        return RequestMode.TEXT


    raise VisionGPTError(
        message="Add an image or enter a message.",
        code="EMPTY_REQUEST",
        status_code=400
    )