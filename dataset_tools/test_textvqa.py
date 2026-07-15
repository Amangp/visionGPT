from dataset_tools.textvqa_adapter import (
    TextVQAAdapter
)


ANNOTATION_PATH = (
    "future_datasets/"
    "textvqa/"
    "annotations/"
    "TextVQA_0.5.1_train.json"
)


IMAGE_DIRECTORY = (
    "future_datasets/"
    "textvqa/"
    "images"
)


adapter = TextVQAAdapter(

    annotation_path=ANNOTATION_PATH,

    image_directory=IMAGE_DIRECTORY

)


samples = adapter.load()


print(
    "Total TextVQA samples:",
    len(samples)
)


print()


for sample in samples[:5]:

    print(
        sample.to_dict()
    )

    print(
        "-" * 80
    )