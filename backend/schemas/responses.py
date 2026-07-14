from typing import Literal, Optional

from pydantic import BaseModel


class VisionResponse(BaseModel):

    success: bool = True

    mode: Literal[
        "text",
        "describe",
        "vqa"
    ]

    question: Optional[str] = None

    answer: str

    model_ready: bool