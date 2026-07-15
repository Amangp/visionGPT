from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.sessions import router as sessions_router
from api.health import router as health_router
from api.vision import router as vision_router
from core.config import settings
from core.errors import register_exception_handlers


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


app.include_router(health_router)

app.include_router(
    vision_router,
    prefix="/api"
)
app.include_router(
    sessions_router
)

register_exception_handlers(app)


@app.get("/")
async def root():

    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs": "/docs"
    }