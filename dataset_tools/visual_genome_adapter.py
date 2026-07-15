import json

from pathlib import Path

from dataset_tools.base_adapter import (
    BaseDatasetAdapter
)

from dataset_tools.unified_schema import (
    VisionSample
)


class VisualGenomeAdapter(BaseDatasetAdapter):

    dataset_name = "visual_genome"

    task_name = "reasoning"


    def __init__(
        self,
        region_annotation_path: str,
        relationship_annotation_path: str,
        image_directory: str
    ):

        self.region_annotation_path = Path(
            region_annotation_path
        )

        self.relationship_annotation_path = Path(
            relationship_annotation_path
        )

        self.image_directory = Path(
            image_directory
        )


    def load(self) -> list[VisionSample]:

        self._validate_paths()


        region_data = self._load_json(
            self.region_annotation_path
        )


        relationship_data = self._load_json(
            self.relationship_annotation_path
        )


        samples = []


        samples.extend(
            self._load_region_samples(
                region_data
            )
        )


        samples.extend(
            self._load_relationship_samples(
                relationship_data
            )
        )


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


    def _load_region_samples(
        self,
        region_data: list
    ) -> list[VisionSample]:

        samples = []


        for image_record in region_data:

            image_id = str(
                image_record.get(
                    "image_id",
                    ""
                )
            )


            image_path = (
                self._resolve_image_path(
                    image_id
                )
            )


            if image_path is None:

                continue


            regions = image_record.get(
                "regions",
                []
            )


            for region in regions:

                phrase = str(
                    region.get(
                        "phrase",
                        ""
                    )
                ).strip()


                if not phrase:

                    continue


                sample = VisionSample(

                    image_id=image_id,

                    image_path=str(
                        image_path
                    ),

                    question=(
                        "Describe a region "
                        "of this image."
                    ),

                    answer=phrase,

                    task="caption",

                    dataset=self.dataset_name,

                    question_id=str(
                        region.get(
                            "region_id",
                            ""
                        )
                    ),

                    metadata={

                        "source_type": "region",

                        "x": region.get("x"),

                        "y": region.get("y"),

                        "width": region.get(
                            "width"
                        ),

                        "height": region.get(
                            "height"
                        )

                    }

                )


                if self.validate_sample(
                    sample
                ):

                    samples.append(
                        sample
                    )


        return samples


    def _load_relationship_samples(
        self,
        relationship_data: list
    ) -> list[VisionSample]:

        samples = []


        for image_record in relationship_data:

            image_id = str(
                image_record.get(
                    "image_id",
                    ""
                )
            )


            image_path = (
                self._resolve_image_path(
                    image_id
                )
            )


            if image_path is None:

                continue


            relationships = (
                image_record.get(
                    "relationships",
                    []
                )
            )


            for relationship in relationships:

                subject = (
                    relationship.get(
                        "subject",
                        {}
                    )
                )


                obj = (
                    relationship.get(
                        "object",
                        {}
                    )
                )


                predicate = str(
                    relationship.get(
                        "predicate",
                        ""
                    )
                ).strip()


                subject_name = (
                    self._extract_object_name(
                        subject
                    )
                )


                object_name = (
                    self._extract_object_name(
                        obj
                    )
                )


                if (
                    not subject_name
                    or not object_name
                    or not predicate
                ):

                    continue


                question = (

                    f"What is the relationship "
                    f"between {subject_name} "
                    f"and {object_name}?"

                )


                answer = predicate


                sample = VisionSample(

                    image_id=image_id,

                    image_path=str(
                        image_path
                    ),

                    question=question,

                    answer=answer,

                    task=self.task_name,

                    dataset=self.dataset_name,

                    question_id=str(
                        relationship.get(
                            "relationship_id",
                            ""
                        )
                    ),

                    metadata={

                        "source_type": (
                            "relationship"
                        ),

                        "subject": subject_name,

                        "object": object_name

                    }

                )


                if self.validate_sample(
                    sample
                ):

                    samples.append(
                        sample
                    )


        return samples


    def _extract_object_name(
        self,
        object_data: dict
    ) -> str:

        names = object_data.get(
            "names",
            []
        )


        if names:

            return str(
                names[0]
            ).strip()


        return str(
            object_data.get(
                "name",
                ""
            )
        ).strip()


    def _resolve_image_path(
        self,
        image_id: str
    ):

        supported_extensions = [

            ".jpg",

            ".jpeg",

            ".png"

        ]


        for extension in supported_extensions:

            image_path = (

                self.image_directory
                / f"{image_id}{extension}"

            )


            if image_path.exists():

                return image_path


        return None


    def _validate_paths(
        self
    ) -> None:

        if (
            not self.region_annotation_path.exists()
        ):

            raise FileNotFoundError(

                "Visual Genome region "
                "annotations not found: "
                f"{self.region_annotation_path}"

            )


        if (
            not self.relationship_annotation_path.exists()
        ):

            raise FileNotFoundError(

                "Visual Genome relationship "
                "annotations not found: "
                f"{self.relationship_annotation_path}"

            )


        if not self.image_directory.exists():

            raise FileNotFoundError(

                "Visual Genome image directory "
                "not found: "
                f"{self.image_directory}"

            )