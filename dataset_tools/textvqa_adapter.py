import json

from collections import Counter

from pathlib import Path

from dataset_tools.base_adapter import (
    BaseDatasetAdapter
)

from dataset_tools.unified_schema import (
    VisionSample
)


class TextVQAAdapter(BaseDatasetAdapter):

    dataset_name = "textvqa"

    task_name = "ocr"


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


    def load(self) -> list[VisionSample]:

        self._validate_paths()


        with self.annotation_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            annotation_data = json.load(
                file
            )


        records = annotation_data.get(
            "data",
            []
        )


        samples = []


        for record in records:

            sample = self._convert_record(
                record
            )


            if (
                sample is not None
                and self.validate_sample(sample)
            ):

                samples.append(sample)


        return samples


    def _convert_record(
        self,
        record: dict
    ):

        image_id = str(
            record.get(
                "image_id",
                ""
            )
        )


        question = str(
            record.get(
                "question",
                ""
            )
        ).strip()


        question_id = str(
            record.get(
                "question_id",
                ""
            )
        )


        answers = record.get(
            "answers",
            []
        )


        answer = self._select_answer(
            answers
        )


        image_path = self._resolve_image_path(
            image_id
        )


        if image_path is None:

            return None


        return VisionSample(

            image_id=image_id,

            image_path=str(image_path),

            question=question,

            answer=answer,

            task=self.task_name,

            dataset=self.dataset_name,

            question_id=question_id,

            metadata={

                "all_answers": answers

            }

        )


    def _select_answer(
        self,
        answers: list
    ) -> str:

        normalized_answers = []


        for answer in answers:

            if isinstance(answer, dict):

                answer_text = answer.get(
                    "answer",
                    ""
                )

            else:

                answer_text = answer


            answer_text = str(
                answer_text
            ).strip()


            if answer_text:

                normalized_answers.append(
                    answer_text
                )


        if not normalized_answers:

            return ""


        answer_counts = Counter(
            normalized_answers
        )


        return answer_counts.most_common(
            1
        )[0][0]


    def _resolve_image_path(
        self,
        image_id: str
    ):

        supported_extensions = [

            ".jpg",

            ".jpeg",

            ".png",

            ".webp"

        ]


        for extension in supported_extensions:

            image_path = (

                self.image_directory
                / f"{image_id}{extension}"

            )


            if image_path.exists():

                return image_path


        return None


    def _validate_paths(self):

        if not self.annotation_path.exists():

            raise FileNotFoundError(

                "TextVQA annotation file "
                f"not found: "
                f"{self.annotation_path}"

            )


        if not self.image_directory.exists():

            raise FileNotFoundError(

                "TextVQA image directory "
                f"not found: "
                f"{self.image_directory}"

            )