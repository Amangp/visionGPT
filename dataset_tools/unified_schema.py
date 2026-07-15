from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class VisionSample:
    image_id: str
    image_path: str
    question: str
    answer: str
    task: str
    dataset: str

    question_id: Optional[str] = None

    metadata: Optional[dict] = None


    def to_dict(self) -> dict:

        return asdict(self)