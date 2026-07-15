from models.visiongpt import VisionGPT


class ModelFactory:

    @staticmethod
    def create():

        model = VisionGPT()

        return model


model_factory = ModelFactory()