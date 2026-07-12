import json
from pathlib import Path
import random


class COCOTrainingLoader:


    def __init__(
            self,
            image_folder,
            caption_file,
            limit=1000
    ):

        self.image_folder = Path(image_folder)
        self.caption_file = Path(caption_file)

        self.limit = limit



    def load(self):


        with open(
            self.caption_file,
            "r"
        ) as file:

            data = json.load(file)


        # -------------------
        # Create image lookup
        # -------------------

        images = {}

        for img in data["images"]:


            images[
                img["id"]
            ] = img["file_name"]



        pairs = []


        # -------------------
        # Match image-caption
        # -------------------

        for ann in data["annotations"]:


            image_id = ann["image_id"]


            if image_id in images:


                image_path = (

                    self.image_folder
                    /
                    images[image_id]

                )


                caption = ann["caption"]


                pairs.append(
                    (
                        str(image_path),
                        caption
                    )
                )


        random.seed(42)

        random.shuffle(
            pairs
        )

        selected_pairs = pairs[:self.limit]

        unique_images = set(
            image_path
            for image_path, caption in selected_pairs
        )

        print(
            f"Selected caption pairs: {len(selected_pairs)}"
        )

        print(
            f"Unique images: {len(unique_images)}"
        )

        print(
            f"Repeated caption-image pairs: "
            f"{len(selected_pairs) - len(unique_images)}"
        )

        return selected_pairs