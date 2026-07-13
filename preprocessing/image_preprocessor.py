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

        # ==========================================
        # READ IMAGE
        # ==========================================

        image = tf.io.read_file(
            image_path
        )


        # ==========================================
        # AUTO-DECODE JPEG / PNG / GIF / BMP
        # ==========================================

        image = tf.io.decode_image(

            image,

            channels=3,

            expand_animations=False

        )


        # ==========================================
        # CONVERT TYPE
        # ==========================================

        image = tf.cast(

            image,

            tf.float32

        )


        # ==========================================
        # RESIZE
        # ==========================================

        image = tf.image.resize(

            image,

            self.image_size

        )


        # IMPORTANT:
        # DO NOT divide by 255
        #
        # EfficientNet preprocessing is handled
        # by the model architecture.


        return image



if __name__ == "__main__":

    processor = ImagePreprocessor()


    img = processor.process(

        "test_images/person.jpg"

    )


    print(
        "Image shape:",
        img.shape
    )


    print(
        "Minimum value:",
        tf.reduce_min(img).numpy()
    )


    print(
        "Maximum value:",
        tf.reduce_max(img).numpy()
    )