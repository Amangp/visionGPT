from abc import ABC, abstractmethod

from dataset_tools.unified_schema import VisionSample


class BaseDatasetAdapter(ABC):

    dataset_name: str

    task_name: str


    @abstractmethod
    def load(self) -> list[VisionSample]:

        raise NotImplementedError


    def validate_sample(
        self,
        sample: VisionSample
    ) -> bool:

        if not sample.image_id:

            return False


        if not sample.image_path:

            return False


        if not sample.question:

            return False


        if not sample.answer:

            return False


        if not sample.task:

            return False


        if not sample.dataset:

            return False


        return True