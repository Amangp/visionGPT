from dataclasses import dataclass

from typing import Optional


@dataclass
class ModelState:

    loaded: bool = False

    model_name: str = "VisionGPT"

    checkpoint: Optional[str] = None

    error: Optional[str] = None


model_state = ModelState()