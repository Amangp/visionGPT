import json
import random

from pathlib import Path

from dataset_tools.base_adapter import (
    BaseDatasetAdapter
)

from dataset_tools.unified_schema import (
    VisionSample
)


class COCOAdapter(BaseDatasetAdapter):

    dataset_name = "coco"

    task_name = "caption"


    def __init__(
        self,
        annotation_path: str,
        image_directory: str
    ):

        self.annotation_path = Path(
            annotation_path
        )

        self.image_directory = Path(
            image_directory
        )


    def load(
        self,
        limit=None
    ) -> list[VisionSample]:

        self._validate_paths()

        with self.annotation_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            annotation_data = json.load(
                file
            )

        annotations = annotation_data.get(
            "annotations",
            []
        )

        if limit is not None:

            random.shuffle(
                annotations
            )

            annotations = annotations[
                :limit
            ]

            print(
                "Selected annotations:",
                len(annotations)
            )

        samples = []

        for annotation in annotations:

            sample = self._convert_annotation(
                annotation
            )

            if (
                sample is not None
                and self.validate_sample(sample)
            ):

                samples.append(
                    sample
                )

        return samples


    def _convert_annotation(
        self,
        annotation: dict
    ):

        image_id = str(
            annotation.get(
                "image_id",
                ""
            )
        )

        image_path = self._resolve_image_path(
            image_id
        )

        if image_path is None:

            return None

        return VisionSample(

            image_id=image_id,

            image_path=str(image_path),

            question="describe the image",

            answer=str(
                annotation.get(
                    "caption",
                    ""
                )
            ).strip(),

            task=self.task_name,

            dataset=self.dataset_name,

            question_id=str(
                annotation.get(
                    "id",
                    ""
                )
            ),

            metadata={

                "caption_id": annotation.get(
                    "id",
                    ""
                )

            }

        )


    def _resolve_image_path(
        self,
        image_id: str
    ):

        try:

            numeric_image_id = int(
                image_id
            )

        except ValueError:

            return None

        return (

            self.image_directory

            /

            f"{numeric_image_id:012d}.jpg"

        )


    def _validate_paths(
        self
    ):

        if not self.annotation_path.exists():

            raise FileNotFoundError(

                "COCO annotation file "
                f"not found: "
                f"{self.annotation_path}"

            )

        if not self.image_directory.exists():

            raise FileNotFoundError(

                "COCO image directory "
                f"not found: "
                f"{self.image_directory}"

            )