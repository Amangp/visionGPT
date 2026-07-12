from pathlib import Path


class ImageRegistry:


    def __init__(self, image_folder):

        self.image_folder = Path(image_folder)

        self.images = {}


    def register_images(self):

        for image in self.image_folder.iterdir():

            image_id = int(
                image.stem.split("_")[-1]
            )

            self.images[image_id] = image


        print(
            f"Registered {len(self.images)} images"
        )


    def get_image(self, image_id):

        if image_id not in self.images:

            return None


        return self.images[image_id]



if __name__ == "__main__":


    registry = ImageRegistry(
        r"C:\Users\1243a\OneDrive\Desktop\visionGPT\Dataset\COCO\train2017\train2017"
    )


    registry.register_images()


    print(
        registry.get_image(391895)
    )