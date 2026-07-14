from fastapi import FastAPI, Request

from fastapi.responses import JSONResponse


class VisionGPTError(Exception):

    def __init__(
        self,
        message: str,
        code: str = "VISIONGPT_ERROR",
        status_code: int = 500
    ):

        self.message = message

        self.code = code

        self.status_code = status_code

        super().__init__(message)


def register_exception_handlers(
    app: FastAPI
) -> None:

    @app.exception_handler(VisionGPTError)
    async def handler(
        request: Request,
        exc: VisionGPTError
    ):

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message
                }
            }
        )