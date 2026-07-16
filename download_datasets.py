import os
import requests
from pathlib import Path
from tqdm import tqdm

# ==========================================================
# VisionGPT Dataset Downloader
# ==========================================================

ROOT = Path("Dataset")

DATASETS = {

    "COCO": {

        "folder": ROOT / "COCO",

        "files": [

            (
                "train2017.zip",
                "https://images.cocodataset.org/zips/train2017.zip"
            ),

            (
                "val2017.zip",
                "https://images.cocodataset.org/zips/val2017.zip"
            ),

            (
                "annotations_trainval2017.zip",
                "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            )

        ]

    },

    "VisualGenome": {

        "folder": ROOT / "VisualGenome",

        "files": [

            (
                "images_part1.zip",
                "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
            ),

            (
                "images_part2.zip",
                "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
            ),

            (
                "region_descriptions.json.zip",
                "https://homes.cs.washington.edu/~ranjay/visualgenome/api/v0/data/region_descriptions.json.zip"
            ),

            (
                "question_answers.json.zip",
                "https://homes.cs.washington.edu/~ranjay/visualgenome/api/v0/data/question_answers.json.zip"
            ),

            (
                "scene_graphs.json.zip",
                "https://homes.cs.washington.edu/~ranjay/visualgenome/api/v0/data/scene_graphs.json.zip"
            )

        ]

    },

    "GQA": {

        "folder": ROOT / "GQA",

        "files": [

            (
                "images.zip",
                "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
            ),

            (
                "train_balanced_questions.json",
                "https://downloads.cs.stanford.edu/nlp/data/gqa/train_balanced_questions.json"
            ),

            (
                "val_balanced_questions.json",
                "https://downloads.cs.stanford.edu/nlp/data/gqa/val_balanced_questions.json"
            )

        ]

    },

    "TextVQA": {

        "folder": ROOT / "TextVQA",

        "files": [

            (
                "TextVQA_0.5.1_train.json",
                "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json"
            ),

            (
                "TextVQA_0.5.1_val.json",
                "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json"
            ),

            (
                "train_val_images.zip",
                "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
            )

        ]

    }

}


# ==========================================================
# DOWNLOAD
# ==========================================================

def download(url, destination):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if destination.exists():

        print(f"✓ Already exists: {destination.name}")

        return

    response = requests.get(
        url,
        stream=True
    )

    response.raise_for_status()

    total = int(
        response.headers.get(
            "content-length",
            0
        )
    )

    with open(destination, "wb") as file:

        with tqdm(

            total=total,

            unit="B",

            unit_scale=True,

            desc=destination.name

        ) as progress:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    file.write(chunk)

                    progress.update(
                        len(chunk)
                    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("\nVisionGPT Dataset Downloader\n")

    for dataset_name, dataset in DATASETS.items():

        print("=" * 60)

        print(dataset_name)

        print("=" * 60)

        folder = dataset["folder"]

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        for filename, url in dataset["files"]:

            try:

                download(

                    url,

                    folder / filename

                )

            except Exception as error:

                print(

                    f"\nFailed: {filename}"

                )

                print(error)

    print("\nAll downloads completed.") 