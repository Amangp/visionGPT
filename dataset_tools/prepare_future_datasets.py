from dataset_tools.docvqa_adapter import (
    DocVQAAdapter
)

from dataset_tools.instruction_adapter import (
    InstructionAdapter
)

from dataset_tools.okvqa_adapter import (
    OKVQAAdapter
)

from dataset_tools.dataset_manager import (
    dataset_manager
)

from dataset_tools.dataset_registry import (
    dataset_registry
)

from dataset_tools.textvqa_adapter import (
    TextVQAAdapter
)
from dataset_tools.visual_genome_adapter import (
    VisualGenomeAdapter
)

TEXTVQA_ANNOTATIONS = (
    "future_datasets/"
    "textvqa/"
    "annotations/"
    "TextVQA_0.5.1_train.json"
)


TEXTVQA_IMAGES = (
    "future_datasets/"
    "textvqa/"
    "images"
)

VISUAL_GENOME_REGIONS = (
    "future_datasets/"
    "visual_genome/"
    "annotations/"
    "region_descriptions.json"
)


VISUAL_GENOME_RELATIONSHIPS = (
    "future_datasets/"
    "visual_genome/"
    "annotations/"
    "relationships.json"
)


VISUAL_GENOME_IMAGES = (
    "future_datasets/"
    "visual_genome/"
    "images"
)

DOCVQA_ANNOTATIONS = (
    "future_datasets/"
    "docvqa/"
    "annotations/"
    "train_v1.0.json"
)


DOCVQA_IMAGES = (
    "future_datasets/"
    "docvqa/"
    "images"
)

OKVQA_QUESTIONS = (
    "future_datasets/"
    "okvqa/"
    "annotations/"
    "OpenEnded_mscoco_train2014_questions.json"
)


OKVQA_ANNOTATIONS = (
    "future_datasets/"
    "okvqa/"
    "annotations/"
    "mscoco_train2014_annotations.json"
)


OKVQA_IMAGES = (
    "future_datasets/"
    "okvqa/"
    "images"
)

INSTRUCTION_ANNOTATIONS = (
    "future_datasets/"
    "instruction/"
    "annotations/"
    "llava_instruct_150k.json"
)


INSTRUCTION_IMAGES = (
    "future_datasets/"
    "instruction/"
    "images"
)

def register_datasets() -> None:

    textvqa_adapter = TextVQAAdapter(

        annotation_path=(
            TEXTVQA_ANNOTATIONS
        ),

        image_directory=(
            TEXTVQA_IMAGES
        )

    )
    docvqa_adapter = DocVQAAdapter(

        annotation_path=(
            DOCVQA_ANNOTATIONS
        ),

        image_directory=(
            DOCVQA_IMAGES
        )

    )
    okvqa_adapter = OKVQAAdapter(

        question_path=(
            OKVQA_QUESTIONS
        ),

        annotation_path=(
            OKVQA_ANNOTATIONS
        ),

        image_directory=(
            OKVQA_IMAGES
        )

    )
    instruction_adapter = InstructionAdapter(

        annotation_path=(
            INSTRUCTION_ANNOTATIONS
        ),

        image_directory=(
            INSTRUCTION_IMAGES
        )

    )


    dataset_registry.register(
        instruction_adapter
    )


    dataset_registry.register(
        okvqa_adapter
    )

    dataset_registry.register(
        docvqa_adapter
    )


    dataset_registry.register(
        textvqa_adapter
    )
    
    visual_genome_adapter = (
        VisualGenomeAdapter(

            region_annotation_path=(
                VISUAL_GENOME_REGIONS
            ),

            relationship_annotation_path=(
                VISUAL_GENOME_RELATIONSHIPS
            ),

            image_directory=(
                VISUAL_GENOME_IMAGES
            )

        )
    )


    dataset_registry.register(
        visual_genome_adapter
    )


def main() -> None:

    register_datasets()


    print(
        "Registered future datasets:",
        dataset_registry.list_datasets()
    )


    samples = dataset_manager.load_all()


    dataset_manager.print_audit(
        samples
    )


if __name__ == "__main__":

    main()