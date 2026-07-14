from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:

    APP_NAME: str = "VisionGPT API"

    APP_VERSION: str = "0.2.0"

    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024

    ALLOWED_IMAGE_TYPES: frozenset[str] = frozenset({
        "image/jpeg",
        "image/png",
        "image/webp"
    })

    CORS_ORIGINS: list[str] = field(
        default_factory=lambda: ["*"]
    )


settings = Settings()