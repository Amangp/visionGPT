from collections import Counter

from dataset_tools.dataset_registry import (
    DatasetRegistry,
    dataset_registry
)

from dataset_tools.unified_schema import (
    VisionSample
)


class DatasetManager:

    def __init__(
        self,
        registry: DatasetRegistry = dataset_registry
    ):

        self.registry = registry


    def load_dataset(
        self,
        dataset_name: str
    ) -> list[VisionSample]:

        adapter = self.registry.get(
            dataset_name
        )

        samples = adapter.load()

        print(
            f"[{dataset_name}] "
            f"Loaded samples: "
            f"{len(samples):,}"
        )

        return samples


    def load_all(
        self
    ) -> list[VisionSample]:

        all_samples = []

        for dataset_name in (
            self.registry.list_datasets()
        ):

            samples = self.load_dataset(
                dataset_name
            )

            all_samples.extend(
                samples
            )

        return all_samples


    def audit(
        self,
        samples: list[VisionSample]
    ) -> dict:

        dataset_counts = Counter(
            sample.dataset
            for sample in samples
        )

        task_counts = Counter(
            sample.task
            for sample in samples
        )

        unique_images = {
            (
                sample.dataset,
                sample.image_id
            )
            for sample in samples
        }

        empty_questions = sum(
            1
            for sample in samples
            if not sample.question.strip()
        )

        empty_answers = sum(
            1
            for sample in samples
            if not sample.answer.strip()
        )

        duplicate_samples = (
            len(samples)
            - len(
                {
                    (
                        sample.dataset,
                        sample.image_id,
                        sample.question,
                        sample.answer
                    )
                    for sample in samples
                }
            )
        )

        return {

            "total_samples": len(samples),

            "unique_images": len(
                unique_images
            ),

            "dataset_counts": dict(
                dataset_counts
            ),

            "task_counts": dict(
                task_counts
            ),

            "empty_questions": (
                empty_questions
            ),

            "empty_answers": (
                empty_answers
            ),

            "duplicate_samples": (
                duplicate_samples
            )

        }


    def print_audit(
        self,
        samples: list[VisionSample]
    ) -> None:

        result = self.audit(
            samples
        )

        print()
        print("=" * 60)
        print("VISIONGPT FUTURE DATASET AUDIT")
        print("=" * 60)

        print(
            "Total samples:",
            result["total_samples"]
        )

        print(
            "Unique images:",
            result["unique_images"]
        )

        print(
            "Empty questions:",
            result["empty_questions"]
        )

        print(
            "Empty answers:",
            result["empty_answers"]
        )

        print(
            "Duplicate samples:",
            result["duplicate_samples"]
        )

        print()

        print("Dataset distribution:")

        for dataset_name, count in (
            result["dataset_counts"].items()
        ):

            print(
                f"  {dataset_name}: "
                f"{count:,}"
            )

        print()

        print("Task distribution:")

        for task_name, count in (
            result["task_counts"].items()
        ):

            print(
                f"  {task_name}: "
                f"{count:,}"
            )

        print("=" * 60)


dataset_manager = DatasetManager()