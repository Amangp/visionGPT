import json

from collections import Counter

from pathlib import Path

from dataset_tools.base_adapter import (
    BaseDatasetAdapter
)

from dataset_tools.unified_schema import (
    VisionSample
)


class DocVQAAdapter(BaseDatasetAdapter):

    dataset_name = "docvqa"

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


        records = self._extract_records(
            annotation_data
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


    def _extract_records(
        self,
        annotation_data
    ) -> list:

        if isinstance(
            annotation_data,
            list
        ):

            return annotation_data


        if isinstance(
            annotation_data,
            dict
        ):

            for key in [
                "data",
                "dataset",
                "questions"
            ]:

                records = annotation_data.get(
                    key
                )


                if isinstance(
                    records,
                    list
                ):

                    return records


        return []


    def _convert_record(
        self,
        record: dict
    ):

        question = str(

            record.get(
                "question",
                ""
            )

        ).strip()


        question_id = str(

            record.get(
                "questionId",
                record.get(
                    "question_id",
                    ""
                )
            )

        )


        image_name = str(

            record.get(
                "image",
                record.get(
                    "image_name",
                    ""
                )
            )

        ).strip()


        image_id = str(

            record.get(
                "imageId",
                record.get(
                    "image_id",
                    image_name
                )
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

            image_name=image_name,

            image_id=image_id

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

                "source_image_name": (
                    image_name
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

                answer_text = (

                    answer.get(
                        "answer",
                        answer.get(
                            "text",
                            ""
                        )
                    )

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
        image_name: str,
        image_id: str
    ):

        if image_name:

            direct_path = (

                self.image_directory
                / image_name

            )


            if direct_path.exists():

                return direct_path


        supported_extensions = [

            ".jpg",

            ".jpeg",

            ".png",

            ".webp",

            ".tif",

            ".tiff"

        ]


        for candidate_name in [

            image_name,

            image_id

        ]:

            if not candidate_name:

                continue


            candidate_path = Path(
                candidate_name
            )


            candidate_stem = (
                candidate_path.stem
            )


            for extension in (
                supported_extensions
            ):

                image_path = (

                    self.image_directory
                    / (
                        candidate_stem
                        + extension
                    )

                )


                if image_path.exists():

                    return image_path


        return None


    def _validate_paths(
        self
    ) -> None:

        if not self.annotation_path.exists():

            raise FileNotFoundError(

                "DocVQA annotation file "
                f"not found: "
                f"{self.annotation_path}"

            )


        if not self.image_directory.exists():

            raise FileNotFoundError(

                "DocVQA image directory "
                f"not found: "
                f"{self.image_directory}"

            )