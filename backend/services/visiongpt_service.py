from typing import Optional

async def run_visiongpt(image_bytes: Optional[bytes], question: Optional[str]) -> str:
    """
    Temporary service stub. Final model inference will be connected here.
    """
    if image_bytes and question:
        return (
            f'VisionGPT API is connected. I received your image and the question: '
            f'"{question}". Model inference will be plugged into this service.'
        )

    if image_bytes:
        return (
            "VisionGPT API is connected. I received the image. "
            "Automatic image description inference will be plugged into this service."
        )

    return (
        f'I received your text message: "{question}". '
        "Text-only routing is enabled and can later be connected to the conversational layer."
    )
