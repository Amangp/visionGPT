"""
Supported datasets and their download metadata.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetFile:
    """
    Metadata for a single file belonging to a dataset.
    """
    url: str
    filename: str
    sub_dir: str  # Destination sub-folder under DATASET_ROOT
    expected_size: Optional[int] = None  # Expected file size in bytes
    sha256: Optional[str] = None  # Expected SHA256 hash


@dataclass
class DatasetInfo:
    """
    Metadata and verification specifications for a dataset.
    """
    name: str
    files: List[DatasetFile]
    verify_dirs: List[str] = field(default_factory=list)  # Dirs expected after extraction
    verify_files: List[str] = field(default_factory=list)  # Files expected after extraction


# Registry of supported datasets
DATASETS = {
    "coco": DatasetInfo(
        name="COCO 2017",
        files=[
            DatasetFile(
                url="http://images.cocodataset.org/zips/train2017.zip",
                filename="train2017.zip",
                sub_dir="COCO",
                expected_size=19336142479,
                sha256="cced2d1b71455dcfda8e11e03b6ecbc324b4c7943d9647b2a59e93b5ca77a3a6"
            ),
            DatasetFile(
                url="http://images.cocodataset.org/zips/val2017.zip",
                filename="val2017.zip",
                sub_dir="COCO",
                expected_size=815585330,
                sha256="4f7e2cc03a8cdb090349d3d3c8c7d6d5ef66432b05b3de9b15bc21743a18a581"
            ),
            DatasetFile(
                url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                filename="annotations_trainval2017.zip",
                sub_dir="COCO",
                expected_size=252907541,
                sha256="113a1a38ef5be8fb154984afad7da7793d5df006a8f33878b274e1d530f2c418"
            )
        ],
        verify_dirs=[
            "COCO/train2017",
            "COCO/val2017",
            "COCO/annotations"
        ],
        verify_files=[
            "COCO/annotations/captions_train2017.json",
            "COCO/annotations/captions_val2017.json",
            "COCO/annotations/instances_train2017.json",
            "COCO/annotations/instances_val2017.json"
        ]
    ),
    "textvqa": DatasetInfo(
        name="TextVQA",
        files=[
            DatasetFile(
                url="https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
                filename="train_val_images.zip",
                sub_dir="textvqa",
                expected_size=7496667136,
                sha256="6f6a73ff2557e4e137fb5a472c72b8d0c26880de71d33c5e7b2354e601c40212"
            ),
            DatasetFile(
                url="https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json",
                filename="TextVQA_0.5.1_train.json",
                sub_dir="textvqa/annotations",
                expected_size=33777416,
                sha256="bf06b00684f8841db9fb66f5d81249b6d859fa27dfdfb45ccde7c2c54c5409ef"
            ),
            DatasetFile(
                url="https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json",
                filename="TextVQA_0.5.1_val.json",
                sub_dir="textvqa/annotations",
                expected_size=10068019,
                sha256="d5930068fdbe344ffde55d045d656910ee24c538cb3dc7c3dd72e5c8e3100234"
            )
        ],
        verify_dirs=[
            "textvqa/train_images",
            "textvqa/annotations"
        ],
        verify_files=[
            "textvqa/annotations/TextVQA_0.5.1_train.json",
            "textvqa/annotations/TextVQA_0.5.1_val.json"
        ]
    ),
    "gqa": DatasetInfo(
        name="GQA",
        files=[
            DatasetFile(
                url="https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
                filename="images.zip",
                sub_dir="gqa",
                expected_size=21815174541,
                sha256="c53648ebcda9fb9317544078df7c4df3ff3b3e8c7d6d5ef66432b05b3de9b15b"
            ),
            DatasetFile(
                url="https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
                filename="questions1.2.zip",
                sub_dir="gqa",
                expected_size=287841123,
                sha256="153d299a88ca44408e81336e734d316a8b274e1d530f2c418bf06b00684f884a"
            )
        ],
        verify_dirs=[
            "gqa/images",
            "gqa/questions"
        ],
        verify_files=[
            "gqa/questions/train_balanced_questions.json",
            "gqa/questions/val_balanced_questions.json"
        ]
    ),
    "visualgenome": DatasetInfo(
        name="Visual Genome",
        files=[
            DatasetFile(
                url="https://cs.stanford.edu/people/ranjaykrishna/genome/image_data_v1/images.zip",
                filename="images.zip",
                sub_dir="visual_genome/part1",
                expected_size=9700000000,
                sha256="3cc05b3de9b15bc21743a18a5814cc324b4c7943d9647b2a59e93b5ca77a3a6b"
            ),
            DatasetFile(
                url="https://cs.stanford.edu/people/ranjaykrishna/genome/image_data_v1/images2.zip",
                filename="images2.zip",
                sub_dir="visual_genome/part2",
                expected_size=5500000000,
                sha256="bcbc324b4c7943d9647b2a59e93b5ca77a3a6b05b3de9b15bc21743a18a5814c"
            ),
            DatasetFile(
                url="https://cs.stanford.edu/people/ranjaykrishna/genome/image_data_v1/region_descriptions.json.zip",
                filename="region_descriptions.json.zip",
                sub_dir="visual_genome",
                expected_size=77000000,
                sha256="1d530f2c418bf06b00684f884a4f7e2cc03a8cdb090349d3d3c8c7d6d5ef66432"
            ),
            DatasetFile(
                url="https://cs.stanford.edu/people/ranjaykrishna/genome/image_data_v1/scene_graphs.json.zip",
                filename="scene_graphs.json.zip",
                sub_dir="visual_genome",
                expected_size=33000000,
                sha256="5df006a8f33878b274e1d530f2c418cced2d1b71455dcfda8e11e03b6ecbc324"
            ),
            DatasetFile(
                url="https://cs.stanford.edu/people/ranjaykrishna/genome/image_data_v1/question_answers.json.zip",
                filename="question_answers.json.zip",
                sub_dir="visual_genome",
                expected_size=60000000,
                sha256="e11e03b6ecbc324b4c7943d9647b2a59e93b5ca77a3a6b05b3de9b15bc21743a"
            )
        ],
        verify_dirs=[
            "visual_genome/images",
            "visual_genome/annotations"
        ],
        verify_files=[
            "visual_genome/annotations/region_descriptions.json",
            "visual_genome/annotations/scene_graphs.json",
            "visual_genome/annotations/question_answers.json"
        ]
    )
}
