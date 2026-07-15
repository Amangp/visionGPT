from typing import Optional

import numpy as np

from core.errors import VisionGPTError

from backend.inference.checkpoint_loader import visiongpt_loader


class VisionGPTPredictor:


    async def predict(

        self,

        image_tensor: np.ndarray,

        question_tokens: Optional[list[int]]

    ) -> list[int]:


        model = visiongpt_loader.model


        if model is None:


            raise VisionGPTError(

                message=(
                    "VisionGPT model "
                    "is not loaded."
                ),

                code="MODEL_NOT_READY",

                status_code=503

            )


        # =====================================
        # FINAL VISIONGPT INFERENCE
        # =====================================

        # prediction = model(
        #     image_tensor,
        #     question_tokens,
        #     training=False
        # )
        #
        # token_ids = ...
        #
        # return token_ids


        raise VisionGPTError(

            message=(
                "VisionGPT prediction "
                "logic is waiting for the "
                "final v3 architecture."
            ),

            code="PREDICTION_NOT_IMPLEMENTED",

            status_code=503

        )


visiongpt_predictor = VisionGPTPredictor()