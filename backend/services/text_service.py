async def generate_text_response(
    message: str
) -> str:

    return (
        f'Text route active. '
        f'You said: "{message}" '
        f'The conversational layer '
        f'can be connected here.'
    )