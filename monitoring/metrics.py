"""Training metrics definition and serialization.

Defines the EpochMetrics dataclass representing all logged information
for a single training epoch.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class EpochMetrics:
    """Dataclass holding all metrics recorded during a single training epoch.

    Supports training loss, validation loss, learning rate, system memory,
    evaluation scores (BLEU, METEOR, ROUGE-L, CIDEr, EM, Token Acc), and durations.
    """

    epoch: int
    loss: float
    validation_loss: Optional[float] = None
    learning_rate: Optional[float] = None

    # Evaluation metrics
    bleu_1: Optional[float] = None
    bleu_2: Optional[float] = None
    bleu_3: Optional[float] = None
    bleu_4: Optional[float] = None
    meteor: Optional[float] = None
    rouge_l: Optional[float] = None
    cider: Optional[float] = None
    exact_match: Optional[float] = None
    token_accuracy: Optional[float] = None

    # Performance and resources
    time_per_epoch: Optional[float] = None  # in seconds
    gpu_memory: Optional[float] = None  # VRAM peak in MB
    cpu_memory: Optional[float] = None  # RAM peak in MB

    def to_dict(self) -> Dict[str, Any]:
        """Convert the metrics dataclass to a dictionary.

        Excludes entries with None values to keep the serialized logs clean.

        Returns:
            A dictionary containing the metrics.
        """
        # Filter out None values to keep serialization tidy
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpochMetrics":
        """Reconstruct an EpochMetrics instance from a dictionary.

        Args:
            data: The dictionary containing metrics keys and values.

        Returns:
            An EpochMetrics instance.
        """
        return cls(
            epoch=int(data["epoch"]),
            loss=float(data["loss"]),
            validation_loss=data.get("validation_loss"),
            learning_rate=data.get("learning_rate"),
            bleu_1=data.get("bleu_1"),
            bleu_2=data.get("bleu_2"),
            bleu_3=data.get("bleu_3"),
            bleu_4=data.get("bleu_4"),
            meteor=data.get("meteor"),
            rouge_l=data.get("rouge_l"),
            cider=data.get("cider"),
            exact_match=data.get("exact_match"),
            token_accuracy=data.get("token_accuracy"),
            time_per_epoch=data.get("time_per_epoch"),
            gpu_memory=data.get("gpu_memory"),
            cpu_memory=data.get("cpu_memory"),
        )
