from models.visiongpt import VisionGPT


class ModelFactory:

    @staticmethod
    def create():

        model = VisionGPT(
            vocab_size=10000,
            embed_dim=256,
            context_dropout_rate=0.10,
            mask_token_id=1,
            start_token_id=3,
            end_token_id=4,
            answer_token_id=5,
            task_caption_token_id=7,
            task_ocr_token_id=14
        )

        return model


model_factory = ModelFactory()