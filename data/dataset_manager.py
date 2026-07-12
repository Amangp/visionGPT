from utils.image_registry import ImageRegistry
from utils.coco_reader import COCOReader


class DatasetManager:


    def __init__(
            self,
            image_folder,
            annotation_file
    ):


        self.image_registry = ImageRegistry(
            image_folder
        )


        self.coco_reader = COCOReader(
            annotation_file
        )


    def prepare(self):


        self.image_registry.register_images()


        self.coco_reader.load_annotations()


    def get_sample(self, image_id):


        image = self.image_registry.get_image(
            image_id
        )


        captions = self.coco_reader.get_captions(
            image_id
        )


        return {
            "image": image,
            "captions": captions
        }



if __name__ == "__main__":


    manager = DatasetManager(

        image_folder=
        r"C:\Users\1243a\OneDrive\Desktop\visionGPT\Dataset\COCO\train2017\train2017",


        annotation_file=
        r"C:\Users\1243a\OneDrive\Desktop\visionGPT\Dataset\COCO\annotations_trainval2017\annotations\captions_train2017.json"

    )


    manager.prepare()


    sample = manager.get_sample(
        391895
    )


    print(sample)