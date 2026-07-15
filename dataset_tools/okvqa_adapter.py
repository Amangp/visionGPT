import json

from collections import Counter

from pathlib import Path

from dataset_tools.base_adapter import (
    BaseDatasetAdapter
)

from dataset_tools.unified_schema import (
    VisionSample
)


class OKVQAAdapter(BaseDatasetAdapter):

    dataset_name = "okvqa"

    task_name = "knowledge"


    def __init__(
        self,
        question_path: str,
        annotation_path: str,
        image_directory: str
    ):

        self.question_path = Path(
            question_path
        )

        self.annotation_path = Path(
            annotation_path
        )

        self.image_directory = Path(
            image_directory
        )


    def load(self) -> list[VisionSample]:

        self._validate_paths()


        question_data = self._load_json(
            self.question_path
        )


        annotation_data = self._load_json(
            self.annotation_path
        )


        questions = question_data.get(
            "questions",
            []
        )


        annotations = annotation_data.get(
            "annotations",
            []
        )


        annotation_map = {

            str(
                annotation.get(
                    "question_id"
                )
            ): annotation

            for annotation in annotations

        }


        samples = []


        for question_record in questions:

            sample = self._convert_record(

                question_record=question_record,

                annotation_map=annotation_map

            )


            if (
                sample is not None
                and self.validate_sample(sample)
            ):

                samples.append(sample)


        return samples


    def _load_json(
        self,
        path: Path
    ):

        with path.open(
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)


    def _convert_record(
        self,
        question_record: dict,
        annotation_map: dict
    ):

        question_id = str(

            question_record.get(
                "question_id",
                ""
            )

        )


        image_id = str(

            question_record.get(
                "image_id",
                ""
            )

        )


        question = str(

            question_record.get(
                "question",
                ""
            )

        ).strip()


        annotation = annotation_map.get(
            question_id
        )


        if annotation is None:

            return None


        answers = annotation.get(
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

            image_path=str(
                image_path
            ),

            question=question,

            answer=answer,

            task=self.task_name,

            dataset=self.dataset_name,

            question_id=question_id,

            metadata={

                "all_answers": answers,

                "answer_type": (
                    annotation.get(
                        "answer_type"
                    )
                )

            }

        )


    def _select_answer(
        self,
        answers: list
    ) -> str:

        normalized_answers = []


        for answer in answers:

            if isinstance(
                answer,
                dict
            ):

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

        try:

            numeric_image_id = int(
                image_id
            )


            padded_image_id = (
                f"{numeric_image_id:012d}"
            )

        except ValueError:

            padded_image_id = image_id


        candidate_names = [

            (
                "COCO_train2014_"
                f"{padded_image_id}.jpg"
            ),

            (
                "COCO_val2014_"
                f"{padded_image_id}.jpg"
            ),

            f"{image_id}.jpg",

            f"{padded_image_id}.jpg"

        ]


        for candidate_name in candidate_names:

            image_path = (

                self.image_directory
                / candidate_name

            )


            if image_path.exists():

                return image_path


        return None


    def _validate_paths(
        self
    ) -> None:

        if not self.question_path.exists():

            raise FileNotFoundError(

                "OK-VQA question file "
                "not found: "
                f"{self.question_path}"

            )


        if not self.annotation_path.exists():

            raise FileNotFoundError(

                "OK-VQA annotation file "
                "not found: "
                f"{self.annotation_path}"

            )


        if not self.image_directory.exists():

            raise FileNotFoundError(

                "OK-VQA image directory "
                "not found: "
                f"{self.image_directory}"

            )