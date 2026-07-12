import tensorflow as tf


class ImagePreprocessor:

    def __init__(
            self,
            image_size=(224, 224)
    ):

        self.image_size = image_size


    def process(
            self,
            image_path
    ):

        # --------------------------------------------------
        # Read image
        # --------------------------------------------------

        image = tf.io.read_file(
            image_path
        )


        # --------------------------------------------------
        # Decode JPEG
        # --------------------------------------------------

        image = tf.image.decode_jpeg(
            image,
            channels=3
        )


        # --------------------------------------------------
        # Resize image
        # --------------------------------------------------

        image = tf.image.resize(
            image,
            self.image_size
        )


        # --------------------------------------------------
        # Convert to float32
        #
        # IMPORTANT:
        # EfficientNetB0 expects pixel values in [0, 255]
        #
        # DO NOT divide by 255
        # --------------------------------------------------

        image = tf.cast(
            image,
            tf.float32
        )


        return image


if __name__ == "__main__":

    processor = ImagePreprocessor()

    img = processor.process(
        r"C:\Users\1243a\OneDrive\Desktop\visionGPT\Dataset\COCO\train2017\train2017\000000391895.jpg"
    )

    print(
        "Image shape:",
        img.shape
    )

    print(
        "Minimum pixel value:",
        tf.reduce_min(img).numpy()
    )

    print(
        "Maximum pixel value:",
        tf.reduce_max(img).numpy()
    )