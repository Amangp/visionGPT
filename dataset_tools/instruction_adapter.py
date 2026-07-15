import json

from pathlib import Path

from dataset_tools.base_adapter import (
    BaseDatasetAdapter
)

from dataset_tools.unified_schema import (
    VisionSample
)


class InstructionAdapter(BaseDatasetAdapter):

    dataset_name = "instruction"

    task_name = "instruction"


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

            record_samples = (
                self._convert_record(
                    record
                )
            )


            samples.extend(
                record_samples
            )


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
                "conversations",
                "samples"
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
    ) -> list[VisionSample]:

        image_name = str(
            record.get(
                "image",
                ""
            )
        ).strip()


        image_id = str(
            record.get(
                "id",
                image_name
            )
        )


        conversations = record.get(
            "conversations",
            []
        )


        image_path = self._resolve_image_path(
            image_name
        )


        if image_path is None:

            return []


        samples = []


        pending_question = None

        turn_index = 0


        for conversation in conversations:

            role = str(
                conversation.get(
                    "from",
                    conversation.get(
                        "role",
                        ""
                    )
                )
            ).strip().lower()


            content = str(
                conversation.get(
                    "value",
                    conversation.get(
                        "content",
                        ""
                    )
                )
            ).strip()


            content = self._clean_content(
                content
            )


            if not content:

                continue


            if role in [
                "human",
                "user"
            ]:

                pending_question = content


            elif role in [
                "gpt",
                "assistant"
            ]:

                if not pending_question:

                    continue


                sample = VisionSample(

                    image_id=image_id,

                    image_path=str(
                        image_path
                    ),

                    question=pending_question,

                    answer=content,

                    task=self.task_name,

                    dataset=self.dataset_name,

                    question_id=(
                        f"{image_id}_"
                        f"{turn_index}"
                    ),

                    metadata={

                        "source_type": (
                            "conversation"
                        ),

                        "turn_index": (
                            turn_index
                        )

                    }

                )


                if self.validate_sample(
                    sample
                ):

                    samples.append(
                        sample
                    )


                pending_question = None

                turn_index += 1


        return samples


    def _clean_content(
        self,
        content: str
    ) -> str:

        image_tokens = [

            "<image>",

            "<Image>",

            "<IMAGE>"

        ]


        cleaned_content = content


        for image_token in image_tokens:

            cleaned_content = (
                cleaned_content.replace(
                    image_token,
                    ""
                )
            )


        return cleaned_content.strip()


    def _resolve_image_path(
        self,
        image_name: str
    ):

        if not image_name:

            return None


        direct_path = (
            self.image_directory
            / image_name
        )


        if direct_path.exists():

            return direct_path


        image_path = Path(
            image_name
        )


        stem = image_path.stem


        for extension in [

            ".jpg",

            ".jpeg",

            ".png",

            ".webp"

        ]:

            candidate_path = (

                self.image_directory
                / f"{stem}{extension}"

            )


            if candidate_path.exists():

                return candidate_path


        return None


    def _validate_paths(
        self
    ) -> None:

        if not self.annotation_path.exists():

            raise FileNotFoundError(

                "Instruction annotation "
                "file not found: "
                f"{self.annotation_path}"

            )


        if not self.image_directory.exists():

            raise FileNotFoundError(

                "Instruction image directory "
                "not found: "
                f"{self.image_directory}"

            )