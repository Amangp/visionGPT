from fastapi import APIRouter


from services.session_service import (
    session_service
)


router = APIRouter(
    prefix="/api/sessions",
    tags=["Sessions"]
)


@router.delete("/{session_id}")
async def delete_session(
    session_id: str
):


    deleted = (
        session_service.clear_session(
            session_id
        )
    )


    return {

        "success": deleted,

        "session_id": session_id

    }