from dataset_tools.base_adapter import (
    BaseDatasetAdapter
)


class DatasetRegistry:

    def __init__(self):

        self._adapters: dict[
            str,
            BaseDatasetAdapter
        ] = {}


    def register(
        self,
        adapter: BaseDatasetAdapter
    ) -> None:

        dataset_name = (
            adapter.dataset_name
            .strip()
            .lower()
        )


        if not dataset_name:

            raise ValueError(
                "Dataset adapter must "
                "define dataset_name."
            )


        if dataset_name in self._adapters:

            raise ValueError(
                f"Dataset already registered: "
                f"{dataset_name}"
            )


        self._adapters[
            dataset_name
        ] = adapter


    def get(
        self,
        dataset_name: str
    ) -> BaseDatasetAdapter:

        normalized_name = (
            dataset_name
            .strip()
            .lower()
        )


        if normalized_name not in self._adapters:

            raise KeyError(
                f"Dataset is not registered: "
                f"{normalized_name}"
            )


        return self._adapters[
            normalized_name
        ]


    def list_datasets(
        self
    ) -> list[str]:

        return list(
            self._adapters.keys()
        )


    def is_registered(
        self,
        dataset_name: str
    ) -> bool:

        return (
            dataset_name
            .strip()
            .lower()
            in self._adapters
        )


dataset_registry = DatasetRegistry()