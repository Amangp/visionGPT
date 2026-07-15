from io import BytesIO

import numpy as np

from PIL import Image


IMAGE_SIZE = (224, 224)


class ImagePreprocessor:

    def __init__(
        self,
        image_size=IMAGE_SIZE
    ):

        self.image_size = image_size


    def preprocess(
        self,
        image_bytes: bytes
    ) -> np.ndarray:

        image = Image.open(
            BytesIO(image_bytes)
        )

        image = image.convert("RGB")

        image = image.resize(
            self.image_size
        )

        image = np.asarray(
            image,
            dtype=np.float32
        )

        image = image / 255.0

        image = np.expand_dims(
            image,
            axis=0
        )

        return image


image_preprocessor = ImagePreprocessor()