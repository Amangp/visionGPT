import tensorflow as tf

from models.vision_encoder import VisionEncoder
from preprocessing.image_preprocessor import ImagePreprocessor


def compare_images(
        image_path_1,
        image_path_2
):

    print("\nCreating image processor...")

    image_processor = ImagePreprocessor()

    print("Creating VisionEncoder...")

    encoder = VisionEncoder()


    # =====================================================
    # PROCESS IMAGE 1
    # =====================================================

    image_1 = image_processor.process(
        image_path_1
    )

    image_1 = tf.expand_dims(
        image_1,
        axis=0
    )


    # =====================================================
    # PROCESS IMAGE 2
    # =====================================================

    image_2 = image_processor.process(
        image_path_2
    )

    image_2 = tf.expand_dims(
        image_2,
        axis=0
    )


    # =====================================================
    # ENCODE IMAGES
    # =====================================================

    features_1 = encoder.encode(
        image_1
    )

    features_2 = encoder.encode(
        image_2
    )


    # =====================================================
    # FEATURE STATISTICS
    # =====================================================

    print("\n===================================")

    print("IMAGE 1")

    print("Feature shape:", features_1.shape)

    print(
        "Mean:",
        tf.reduce_mean(
            features_1
        ).numpy()
    )

    print(
        "Std:",
        tf.math.reduce_std(
            features_1
        ).numpy()
    )


    print("\n===================================")

    print("IMAGE 2")

    print("Feature shape:", features_2.shape)

    print(
        "Mean:",
        tf.reduce_mean(
            features_2
        ).numpy()
    )

    print(
        "Std:",
        tf.math.reduce_std(
            features_2
        ).numpy()
    )


    # =====================================================
    # FEATURE DIFFERENCE
    # =====================================================

    difference = tf.reduce_mean(
        tf.abs(
            features_1
            -
            features_2
        )
    )


    print("\n===================================")

    print(
        "MEAN ABSOLUTE FEATURE DIFFERENCE:"
    )

    print(
        difference.numpy()
    )


    # =====================================================
    # COSINE SIMILARITY
    # =====================================================

    flat_features_1 = tf.reshape(
        features_1,
        (
            1,
            -1
        )
    )

    flat_features_2 = tf.reshape(
        features_2,
        (
            1,
            -1
        )
    )


    flat_features_1 = tf.math.l2_normalize(
        flat_features_1,
        axis=1
    )

    flat_features_2 = tf.math.l2_normalize(
        flat_features_2,
        axis=1
    )


    cosine_similarity = tf.reduce_sum(
        flat_features_1
        *
        flat_features_2,
        axis=1
    )


    print("\nCOSINE SIMILARITY:")

    print(
        cosine_similarity.numpy()
    )

    print("===================================\n")


if __name__ == "__main__":

    compare_images(

        "test.jpg",

        "test1.jpg"

    )