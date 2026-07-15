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


        print(
            "Start token:",
            self.start_token
        )


        print(
            "End token:",
            self.end_token
        )
        self.answer_token = self.vocab.index(
            "answerseq"
        )

        self.task_caption_token = self.vocab.index(
            "taskcaption"
        )

        # =================================================
        # 3. CREATE MODEL
        # =================================================

        print(
            "\n[3/4] Creating VisionGPT..."
        )


        self.model = VisionGPT(

            vocab_size=self.vocab_size,
            mask_token_id=self.vocab.index("[UNK]"),

            start_token_id=self.start_token,

            end_token_id=self.end_token
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
            "\nVisionGPT v4 ready 🚀"
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

        repeated_tokens = set(
            int(token)
            for token in generated_tokens
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
        beam_width=5,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        length_penalty=0.7
    ):

        if not os.path.exists(
            image_path
        ):

            raise FileNotFoundError(

                f"Image not found: "
                f"{image_path}"

            )


        # =================================================
        # PROCESS IMAGE
        # =================================================

        image = self.image_processor.process(
            image_path
        )

        image = tf.expand_dims(

            image,

            axis=0

        )


        # =================================================
        # INITIAL BEAM
        #
        # beam:
        # (
        #     generated token IDs,
        #     cumulative log probability,
        #     finished
        # )
        # =================================================

        prompt = (
            "describe the image"
        )

        prompt_tokens = self.text_processor.process(
            [prompt]
        ).numpy()[0]

        prompt_tokens = [
            int(token)
            for token in prompt_tokens
            if token > 0
        ]

        initial_tokens = [

            self.start_token,

            self.task_caption_token,

            *prompt_tokens,

            self.answer_token

        ]

        beams = [

            (
                initial_tokens,
                0.0,
                False
            )

        ]


        # =================================================
        # AUTOREGRESSIVE BEAM SEARCH
        # =================================================

        for _ in range(
            max_length - 1
        ):

            candidates = []

            all_finished = True


            for generated, score, finished in beams:

                if finished:

                    candidates.append(

                        (
                            generated,
                            score,
                            True
                        )

                    )

                    continue


                all_finished = False


                tokens = tf.constant(

                    [
                        generated
                    ],

                    dtype=tf.int64

                )


                predictions = self.model(

                    (
                        image,
                        tokens
                    ),

                    training=False

                )


                next_token_logits = predictions[

                    0,

                    -1,

                    :

                ]


                next_token_logits = (
                    self._apply_repetition_penalty(

                        next_token_logits,

                        generated,

                        repetition_penalty

                    )
                )


                # Never generate padding or start token.

                blocked_indices = tf.constant(
                    [
                        0,
                        self.start_token,
                        self.answer_token,
                        self.task_caption_token,    
                        self.vocab.index("taskocr")
                    ],
                    dtype=tf.int32
                )


                next_token_logits = (
                    tf.tensor_scatter_nd_update(

                        next_token_logits,

                        tf.expand_dims(
                            blocked_indices,
                            axis=1
                        ),

                        tf.fill(
                            tf.shape(blocked_indices),
                            tf.cast(
                                -1e9,
                                next_token_logits.dtype
                            )
                        )

                    )
                )


                log_probs = tf.nn.log_softmax(
                    next_token_logits
                )


                search_width = min(

                    self.vocab_size,

                    max(
                        beam_width * 4,
                        beam_width
                    )

                )


                top_values, top_indices = tf.math.top_k(

                    log_probs,

                    k=search_width

                )


                top_values = top_values.numpy()

                top_indices = top_indices.numpy()


                accepted = 0


                for token_log_prob, token_id in zip(

                    top_values,

                    top_indices

                ):

                    token_id = int(
                        token_id
                    )


                    if (

                        token_id != self.end_token

                        and

                        self._creates_repeated_ngram(

                            generated,

                            token_id,

                            no_repeat_ngram_size

                        )

                    ):

                        continue


                    new_generated = (
                        generated
                        +
                        [token_id]
                    )


                    new_score = (

                        score

                        +

                        float(
                            token_log_prob
                        )

                    )


                    new_finished = (
                        token_id
                        ==
                        self.end_token
                    )


                    candidates.append(

                        (
                            new_generated,
                            new_score,
                            new_finished
                        )

                    )


                    accepted += 1


                    if accepted >= beam_width:
                        break


            if all_finished:
                break


            if not candidates:
                break


            def normalized_score(beam):

                generated, score, _ = beam

                generated_length = max(

                    len(generated) - 1,

                    1

                )

                return (

                    score

                    /

                    (
                        generated_length
                        **
                        length_penalty
                    )

                )


            candidates.sort(

                key=normalized_score,

                reverse=True

            )


            beams = candidates[
                :beam_width
            ]


            if all(
                finished
                for _, _, finished in beams
            ):

                break


        # =================================================
        # SELECT BEST BEAM
        # =================================================

        finished_beams = [

            beam

            for beam in beams

            if beam[2]

        ]


        final_beams = (

            finished_beams

            if finished_beams

            else beams

        )


        def final_score(beam):

            generated, score, _ = beam

            generated_length = max(

                len(generated) - 1,

                1

            )

            return (

                score

                /

                (
                    generated_length
                    **
                    length_penalty
                )

            )


        best_generated, _, _ = max(

            final_beams,

            key=final_score

        )


        caption = self.decode_tokens(
            best_generated
        )


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


    MODEL_PATH = (

        
        "checkpoints//visiongpt_v4_best_2026_07_15_19_01.weights.h5"
        

    )


    VOCAB_PATH = (
        "vocab_v4.json"
    )


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

        task_caption_token = word_to_index[
            "taskcaption"
        ]

        answer_token = word_to_index[
            "answerseq"
        ]

        decoder_tokens = [

            start_token,

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


            probabilities = tf.nn.softmax(
                first_token_logits
            ).numpy()


            top_indices = np.argsort(
                probabilities
            )[-10:][::-1]


            print(
                "Top first-token predictions:"
            )


            for index in top_indices:

                print(

                    f"{vocabulary[index]:20s} "

                    f"{probabilities[index]:.6f}"

                )


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

        "test_images/dinosaur.jpg",

        "test_images/person.jpg",

        "test_images/car.jpg",

        "test_images/dog.jpg"

    ]


    bot.test_images(
        TEST_IMAGES
    )