import json
from pathlib import Path


class COCOReader:


    def __init__(self, annotation_file):

        self.annotation_file = Path(annotation_file)

        self.captions = {}


    def load_annotations(self):

        with open(
            self.annotation_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)


        for annotation in data["annotations"]:

            image_id = annotation["image_id"]

            caption = annotation["caption"]


            if image_id not in self.captions:

                self.captions[image_id] = []


            self.captions[image_id].append(
                caption
            )


        print(
            f"Loaded captions for {len(self.captions)} images"
        )


    def get_captions(self, image_id):

        return self.captions.get(
            image_id,
            []
        )



if __name__ == "__main__":


    reader = COCOReader(
        r"C:\Users\1243a\OneDrive\Desktop\visionGPT\Dataset\COCO\annotations_trainval2017\annotations\captions_train2017.json"
    )

    reader.load_annotations()

    print(
        reader.get_captions(391895)
    )