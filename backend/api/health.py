from fastapi import APIRouter

from services.model_state import model_state


router = APIRouter(
    tags=["System"]
)


@router.get("/health")
async def health():


    return {

        "status": "healthy",

        "model": {

            "name": model_state.model_name,

            "ready": model_state.loaded,

            "checkpoint": model_state.checkpoint,

            "error": model_state.error

        }

    }