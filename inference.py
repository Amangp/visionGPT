import tensorflow as tf
import os

from models.visiongpt import VisionGPT

from preprocessing.image_preprocessor import ImagePreprocessor
from preprocessing.text_preprocessor import TextPreprocessor


class VisionGPTInference:

    def __init__(
            self,
            model_path,
            vocab_path="vocab_v4.json"
    ):

        print(
            "\n=========================================="
        )

        print(
            "Loading VisionGPT v4"
        )

        print(
            "=========================================="
        )


        # =================================================
        # 1. LOAD VOCABULARY
        # =================================================

        print(
            "\n[1/4] Loading vocabulary..."
        )

        self.text_processor = TextPreprocessor()

        self.text_processor.load_vocab(
            vocab_path
        )


        self.vocab = (
            self.text_processor
            .vectorizer
            .get_vocabulary()
        )


        self.vocab_size = len(
            self.vocab
        )


        print(
            "Vocabulary size:",
            self.vocab_size
        )


        # =================================================
        # 2. FIND SPECIAL TOKENS
        # =================================================

        print(
            "\n[2/4] Finding special tokens..."
        )


        if "startseq" not in self.vocab:

            raise ValueError(
                "startseq token not found"
            )


        if "endseq" not in self.vocab:

            raise ValueError(
                "endseq token not found"
            )


        self.start_token = self.vocab.index(
            "startseq"
        )

        self.end_token = self.vocab.index(
            "endseq"
        )

        self.answer_token = self.vocab.index(
            "answerseq"
        )

        self.task_caption_token = self.vocab.index(
            "taskcaption"
        )

        self.task_ocr_token = self.vocab.index(
            "taskocr"
        )

        print(
            "Start token:",
            self.start_token
        )

        print(
            "End token:",
            self.end_token
        )

        print(
            "Answer token:",
            self.answer_token
        )

        print(
            "Task caption token:",
            self.task_caption_token
        )

        print(
            "Task OCR token:",
            self.task_ocr_token
        )

        # =================================================
        # 3. CREATE MODEL
        # =================================================

        print(
            "\n[3/4] Creating VisionGPT..."
        )

        self.mask_token_id = self.vocab.index("[UNK]")

        self.model = VisionGPT(
            vocab_size=self.vocab_size,
            mask_token_id=self.mask_token_id,
            start_token_id=self.start_token,
            end_token_id=self.end_token,
            answer_token_id=self.answer_token,
            task_caption_token_id=self.task_caption_token,
            task_ocr_token_id=self.task_ocr_token
        )


        # Build model variables using same sequence length
        # used during training

        dummy_image = tf.zeros(

            (
                1,
                224,
                224,
                3
            ),

            dtype=tf.float32

        )


        dummy_text = tf.ones(

            (
                1,
                49
            ),

            dtype=tf.int64

        )


        self.model(

            (
                dummy_image,
                dummy_text
            ),

            training=False

        )


        print(
            "VisionGPT architecture created"
        )


        # =================================================
        # 4. LOAD WEIGHTS
        # =================================================

        print(
            "\n[4/4] Loading trained weights..."
        )


        if not os.path.exists(
            model_path
        ):

            raise FileNotFoundError(

                f"Model checkpoint not found: "
                f"{model_path}"

            )


        self.model.load_weights(
            model_path
        )


        print(
            "Weights loaded successfully"
        )


        # =================================================
        # IMAGE PROCESSOR
        # =================================================

        self.image_processor = (
            ImagePreprocessor()
        )


        print(
            "\nVisionGPT v4 ready"
        )


    # =====================================================
    # TOKEN DECODER
    # =====================================================

    def decode_tokens(
            self,
            token_ids
    ):

        words = []


        for token_id in token_ids:

            token_id = int(
                token_id
            )


            if token_id <= 0:

                continue


            if token_id >= len(
                self.vocab
            ):

                continue


            word = self.vocab[
                token_id
            ]


            if word in [

                "startseq",
                "endseq",
                "answerseq",
                "taskcaption",
                "taskocr"

            ]:

                continue


            words.append(
                word
            )


        return " ".join(
            words
        )


    # =====================================================
    # BEAM SEARCH HELPERS
    # =====================================================

    def _creates_repeated_ngram(
        self,
        sequence,
        candidate_token,
        ngram_size=3
    ):

        if ngram_size <= 1:
            return candidate_token in sequence

        candidate_sequence = (
            list(sequence)
            +
            [int(candidate_token)]
        )

        if len(candidate_sequence) < ngram_size:
            return False

        new_ngram = tuple(
            candidate_sequence[-ngram_size:]
        )

        previous_ngrams = {

            tuple(
                candidate_sequence[
                    index:index + ngram_size
                ]
            )

            for index in range(
                len(candidate_sequence)
                -
                ngram_size
            )

        }

        return new_ngram in previous_ngrams


    def _apply_repetition_penalty(
        self,
        logits,
        generated_tokens,
        repetition_penalty=1.2
    ):

        if repetition_penalty <= 1.0:
            return logits

        logits = tf.identity(
            logits
        )

        try:
            answer_idx = list(generated_tokens).index(self.answer_token)
            actual_generated = list(generated_tokens)[answer_idx + 1:]
        except ValueError:
            actual_generated = list(generated_tokens)

        repeated_tokens = set(
            int(token)
            for token in actual_generated
            if int(token) > 0
        )

        if not repeated_tokens:
            return logits

        token_indices = tf.constant(
            sorted(repeated_tokens),
            dtype=tf.int32
        )

        token_logits = tf.gather(
            logits,
            token_indices
        )

        penalized_logits = tf.where(

            token_logits < 0,

            token_logits
            *
            repetition_penalty,

            token_logits
            /
            repetition_penalty

        )

        return tf.tensor_scatter_nd_update(

            logits,

            tf.expand_dims(
                token_indices,
                axis=1
            ),

            penalized_logits

        )


    # =====================================================
    # CAPTION GENERATION - BEAM SEARCH
    # =====================================================
    def generate(
        self,
        image_path,
        max_length=29,
        method="beam",
        beam_width=5,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        length_penalty=0.7
    ):
        """Generate text caption or VQA answer for an input image.

        Supports both Beam Search and Temperature/Top-K/Top-P Sampling.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 1. Preprocess Image
        image = self.image_processor.process(image_path)
        image = tf.expand_dims(image, axis=0)

        # 2. Build Initial Sequence
        prompt = "describe the image"
        prompt_tokens = self.text_processor.process([prompt]).numpy()[0]
        prompt_tokens = [int(token) for token in prompt_tokens if token > 0]

        initial_tokens = [
            self.start_token,
            self.task_caption_token,
            *prompt_tokens,
            self.answer_token
        ]

        if method == "beam":
            # =================================================
            # BEAM SEARCH
            # =================================================
            beams = [(initial_tokens, 0.0, False)]

            for _ in range(max_length - 1):
                candidates = []
                all_finished = True

                for generated, score, finished in beams:
                    if finished:
                        candidates.append((generated, score, True))
                        continue

                    all_finished = False
                    tokens = tf.constant([generated], dtype=tf.int64)

                    predictions = self.model((image, tokens), training=False)
                    next_token_logits = predictions[0, -1, :]

                    next_token_logits = self._apply_repetition_penalty(
                        next_token_logits,
                        generated,
                        repetition_penalty
                    )

                    # Block special tokens
                    blocked_indices = tf.constant([
                        0,
                        self.mask_token_id,
                        self.start_token,
                        self.answer_token,
                        self.task_caption_token,    
                        self.task_ocr_token
                    ], dtype=tf.int32)

                    next_token_logits = tf.tensor_scatter_nd_update(
                        next_token_logits,
                        tf.expand_dims(blocked_indices, axis=1),
                        tf.fill(
                            tf.shape(blocked_indices),
                            tf.cast(-1e9, next_token_logits.dtype)
                        )
                    )

                    log_probs = tf.nn.log_softmax(next_token_logits)
                    search_width = min(
                        self.vocab_size,
                        max(beam_width * 4, beam_width)
                    )

                    top_values, top_indices = tf.math.top_k(log_probs, k=search_width)
                    top_values = top_values.numpy()
                    top_indices = top_indices.numpy()

                    accepted = 0
                    for token_log_prob, token_id in zip(top_values, top_indices):
                        token_id = int(token_id)

                        if (token_id != self.end_token and 
                            self._creates_repeated_ngram(
                                generated,
                                token_id,
                                no_repeat_ngram_size
                            )
                        ):
                            continue

                        new_generated = generated + [token_id]
                        new_score = score + float(token_log_prob)
                        new_finished = (token_id == self.end_token)

                        candidates.append((new_generated, new_score, new_finished))
                        accepted += 1

                        if accepted >= beam_width:
                            break

                if all_finished or not candidates:
                    break

                def normalized_score(beam):
                    generated, score, _ = beam
                    try:
                        answer_idx = list(generated).index(self.answer_token)
                        generated_length = max(len(generated) - (answer_idx + 1), 1)
                    except ValueError:
                        generated_length = max(len(generated) - 1, 1)

                    return score / (generated_length ** length_penalty)

                candidates.sort(key=normalized_score, reverse=True)
                beams = candidates[:beam_width]

                if all(finished for _, _, finished in beams):
                    break

            # Select Best Beam
            finished_beams = [beam for beam in beams if beam[2]]
            final_beams = finished_beams if finished_beams else beams

            best_generated, _, _ = max(final_beams, key=normalized_score)
            try:
                answer_idx = best_generated.index(self.answer_token)
                caption_tokens = best_generated[answer_idx + 1:]
            except ValueError:
                caption_tokens = best_generated

            caption = self.decode_tokens(caption_tokens)
            return caption

        else:
            # =================================================
            # TEMPERATURE / TOP-K / TOP-P SAMPLING
            # =================================================
            generated = list(initial_tokens)

            for _ in range(max_length - 1):
                tokens = tf.constant([generated], dtype=tf.int64)

                predictions = self.model((image, tokens), training=False)
                next_token_logits = predictions[0, -1, :]

                # Apply repetition penalty
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits,
                    generated,
                    repetition_penalty
                )

                # N-gram blocking
                if no_repeat_ngram_size > 0:
                    blocked_by_ngram = []
                    for token_id in range(self.vocab_size):
                        if self._creates_repeated_ngram(generated, token_id, no_repeat_ngram_size):
                            blocked_by_ngram.append(token_id)
                    if blocked_by_ngram:
                        blocked_by_ngram_t = tf.constant(blocked_by_ngram, dtype=tf.int32)
                        next_token_logits = tf.tensor_scatter_nd_update(
                            next_token_logits,
                            tf.expand_dims(blocked_by_ngram_t, axis=1),
                            tf.fill(tf.shape(blocked_by_ngram_t), tf.cast(-1e9, next_token_logits.dtype))
                        )

                # Block special tokens
                blocked_indices = tf.constant([
                    0,
                    self.mask_token_id,
                    self.start_token,
                    self.answer_token,
                    self.task_caption_token,
                    self.task_ocr_token
                ], dtype=tf.int32)

                next_token_logits = tf.tensor_scatter_nd_update(
                    next_token_logits,
                    tf.expand_dims(blocked_indices, axis=1),
                    tf.fill(
                        tf.shape(blocked_indices),
                        tf.cast(-1e9, next_token_logits.dtype)
                    )
                )

                # Temperature scaling
                if temperature > 0.0:
                    next_token_logits = next_token_logits / temperature

                    # Top-K filtering
                    if top_k > 0:
                        top_values, _ = tf.math.top_k(next_token_logits, k=top_k)
                        min_value = top_values[-1]
                        next_token_logits = tf.where(
                            next_token_logits < min_value,
                            tf.cast(-1e9, next_token_logits.dtype),
                            next_token_logits
                        )

                    # Top-P (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits = tf.sort(next_token_logits, direction="DESCENDING")
                        sorted_probs = tf.nn.softmax(sorted_logits)
                        cumulative_probs = tf.cumsum(sorted_probs, axis=-1)

                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove = tf.concat(
                            [tf.zeros((1,), dtype=tf.bool), sorted_indices_to_remove[:-1]],
                            axis=0
                        )

                        cutoff_index = tf.reduce_sum(tf.cast(tf.logical_not(sorted_indices_to_remove), tf.int32)) - 1
                        cutoff_value = sorted_logits[cutoff_index]
                        next_token_logits = tf.where(
                            next_token_logits < cutoff_value,
                            tf.cast(-1e9, next_token_logits.dtype),
                            next_token_logits
                        )

                    # Sample next token
                    probs = tf.nn.softmax(next_token_logits)
                    sampled_token = tf.random.categorical(
                        tf.expand_dims(tf.math.log(probs + 1e-10), axis=0),
                        num_samples=1
                    )[0, 0]
                    next_token = int(sampled_token.numpy())
                else:
                    # Greedy search if temperature is 0
                    next_token = int(tf.argmax(next_token_logits).numpy())

                generated.append(next_token)

                if next_token == self.end_token:
                    break

            try:
                answer_idx = generated.index(self.answer_token)
                caption_tokens = generated[answer_idx + 1:]
            except ValueError:
                caption_tokens = generated

            caption = self.decode_tokens(caption_tokens)
            return caption


    # =====================================================
    # MULTIPLE IMAGE TEST
    # =====================================================

    def test_images(
            self,
            image_paths
    ):

        print(
            "\n"
            "=========================================="
        )

        print(
            "VISIONGPT v3.1 IMAGE TEST"
        )

        print(
            "=========================================="
        )


        results = {}


        for image_path in image_paths:

            print(
                "\n------------------------------------------"
            )


            print(
                "Image:",
                image_path
            )


            try:

                caption = self.generate(
                    image_path
                )


                results[
                    image_path
                ] = caption


                print(
                    "Caption:"
                )


                print(
                    caption
                )


            except Exception as error:

                print(
                    "ERROR:"
                )

                print(
                    error
                )


        print(
            "\n"
            "=========================================="
        )

        print(
            "TEST COMPLETED"
        )

        print(
            "=========================================="
        )


        return results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import os
    import glob
    VOCAB_PATH = "vocab_v4.json"
    
    # Find the latest saved weights file in checkpoints/
    weight_files = glob.glob("checkpoints/visiongpt_v4_best_*.weights.h5")
    if weight_files:
        # Sort by modification time to get the newest
        MODEL_PATH = max(weight_files, key=os.path.getmtime)
        print(f"\n[Diagnostic] Loading latest trained weights: {MODEL_PATH}")
    else:
        MODEL_PATH = "checkpoints/visiongpt_v5_temp.weights.h5"
        if not os.path.exists(MODEL_PATH):
            import json
            with open(VOCAB_PATH, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
                v_size = len(vocab_data)
            print(f"\n[Diagnostic] No weights file found at {MODEL_PATH}. Building model with vocab_size={v_size} and saving temporary weights...")
            os.makedirs("checkpoints", exist_ok=True)
            from models.visiongpt import VisionGPT
            import tensorflow as tf
            model = VisionGPT(vocab_size=v_size)
            dummy_img = tf.zeros((1, 224, 224, 3))
            dummy_txt = tf.zeros((1, 5), dtype=tf.int64)
            _ = model((dummy_img, dummy_txt), training=False)
            model.save_weights(MODEL_PATH)
            print("[Diagnostic] Temporary random weights saved.")

    bot = VisionGPTInference(
        model_path=MODEL_PATH,
        vocab_path=VOCAB_PATH
    )
    # =========================================================
    # VISUAL CONDITIONING DIAGNOSTIC
    # =========================================================

    import numpy as np
    import tensorflow as tf


    def cosine_similarity(a, b):

        a = np.asarray(a).reshape(-1)
        b = np.asarray(b).reshape(-1)

        denominator = (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )

        if denominator == 0:

            return 0.0

        return float(
            np.dot(a, b)
            /
            denominator
        )


    def run_visual_conditioning_test(bot):

        image_paths = {
            "tree": "test_images/tree.jpg",
            "bike": "test_images/bike.jpg",
            "person": "test_images/person.jpg",
            "car": "test_images/car.jpg",
            "dog": "test_images/dog.jpg"
        }

        logits_by_image = {}
        vocabulary = bot.vocab

        prompt = "describe the image"

        prompt_tokens = (
            bot.text_processor.process(
                [prompt]
            )
            .numpy()[0]
        )

        prompt_tokens = [

            int(token)

            for token in prompt_tokens

            if token > 0

        ]

        task_caption_token = bot.task_caption_token

        answer_token = bot.answer_token

        decoder_tokens = [

            bot.start_token,

            task_caption_token,

            *prompt_tokens,

            answer_token

        ]
        print("\nDiagnostic prompt:")

        print(
            bot.text_processor.decode(
                decoder_tokens
            )
        )

        text_input = tf.constant(

            [
                decoder_tokens
            ],

            dtype=tf.int64

        )


        # =====================================================
        # RUN EVERY IMAGE
        # =====================================================

        for image_name, image_path in image_paths.items():

            print(
                "\nProcessing:",
                image_name
            )


            image = bot.image_processor.process(
                image_path
            )


            image = tf.expand_dims(
                image,
                axis=0
            )


            logits = bot.model(

                (
                    image,
                    text_input
                ),

                training=False

            )


            # Output:
            #
            # (batch, sequence, vocabulary)
            #
            # We inspect first generated token logits.

            first_token_logits = logits[
                0,
                -1,
                :
            ]


            first_token_logits = (
                first_token_logits
                .numpy()
            )


            logits_by_image[
                image_name
            ] = first_token_logits


            probabilities = tf.nn.softmax(first_token_logits).numpy()

            top_indices = np.argsort(probabilities)[-10:][::-1]

            print("\n========================================")
            print("FIRST ANSWER TOKEN DIAGNOSTIC")
            print("========================================")

            print("Expected prompt:")
            print("startseq taskcaption describe the image answerseq")

            print("\nTop 10 predictions:\n")

            for rank, index in enumerate(top_indices, 1):
                print(
                    f"{rank:2d}. "
                    f"ID={index:4d} "
                    f"WORD={vocabulary[index]:20s} "
                    f"PROB={probabilities[index]:.6f}"
                )

            print("========================================\n")


        # =====================================================
        # COSINE SIMILARITY MATRIX
        # =====================================================

        names = list(
            logits_by_image.keys()
        )


        print(
            "\n=========================================="
        )

        print(
            "FIRST TOKEN LOGIT COSINE SIMILARITY"
        )

        print(
            "=========================================="
        )


        print(
            f"{'':12s}",
            end=""
        )


        for name in names:

            print(
                f"{name:12s}",
                end=""
            )


        print()


        for name_a in names:

            print(
                f"{name_a:12s}",
                end=""
            )


            for name_b in names:

                similarity = cosine_similarity(

                    logits_by_image[name_a],

                    logits_by_image[name_b]

                )


                print(

                    f"{similarity:<12.6f}",

                    end=""

                )


            print()


        # =====================================================
        # MEAN ABSOLUTE LOGIT DIFFERENCE
        # =====================================================

        print(
            "\n=========================================="
        )

        print(
            "PAIRWISE MEAN ABSOLUTE LOGIT DIFFERENCE"
        )

        print(
            "=========================================="
        )


        for i in range(
            len(names)
        ):

            for j in range(
                i + 1,
                len(names)
            ):

                name_a = names[i]

                name_b = names[j]


                difference = np.mean(

                    np.abs(

                        logits_by_image[name_a]

                        -

                        logits_by_image[name_b]

                    )

                )


                print(

                    f"{name_a:10s} vs "
                    f"{name_b:10s}: "
                    f"{difference:.8f}"

                )


        print(
            "\n=========================================="
        )

        print(
            "VISUAL CONDITIONING TEST COMPLETED"
        )

        print(
            "=========================================="
        )


    # =========================================================
    # RUN DIAGNOSTIC
    # =========================================================

    run_visual_conditioning_test(
        bot
    )


    TEST_IMAGES = [

        "test_images/tree.jpg",

        "test_images/bike.jpg",

        "test_images/person.jpg",

        "test_images/car.jpg",

        "test_images/dog.jpg"

    ]


    bot.test_images(
        TEST_IMAGES
    )