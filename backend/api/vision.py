from typing import Optional


from fastapi import (
    APIRouter,
    File,
    Form,
    UploadFile
)


from core.config import settings

from core.errors import VisionGPTError

from schemas.responses import VisionResponse

from services.orchestrator import (
    process_request
)


router = APIRouter(
    tags=["VisionGPT"]
)


@router.post(
    "/vision",
    response_model=VisionResponse
)
async def vision(

    image: Optional[UploadFile] = File(
        default=None
    ),

    question: Optional[str] = Form(
        default=None
    ),

    session_id: Optional[str] = Form(
        default=None
    )

):


    image_bytes = None


    if image is not None:


        if (
            image.content_type
            not in settings.ALLOWED_IMAGE_TYPES
        ):

            raise VisionGPTError(

                message=(
                    "Use a JPG, PNG, "
                    "or WEBP image."
                ),

                code=(
                    "UNSUPPORTED_IMAGE_TYPE"
                ),

                status_code=415

            )


        image_bytes = await image.read()


        if (
            len(image_bytes)
            > settings.MAX_IMAGE_SIZE
        ):

            raise VisionGPTError(

                message=(
                    "Image must be "
                    "10 MB or smaller."
                ),

                code="IMAGE_TOO_LARGE",

                status_code=413

            )


        if not image_bytes:

            raise VisionGPTError(

                message=(
                    "The uploaded image "
                    "is empty."
                ),

                code="EMPTY_IMAGE",

                status_code=400

            )


    return await process_request(

        image_bytes=image_bytes,

        question=question,

        session_id=session_id

    )   