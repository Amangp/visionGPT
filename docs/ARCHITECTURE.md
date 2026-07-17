# Model Architecture & Dataflow

This document describes the design, layers, parameters, and computational execution flows of the VisionGPT v3 model.

---

## 1. Core Model Components

The VisionGPT model comprises three primary layers coordinate under `models/visiongpt.py`:

```
┌────────────────────────────────────────────────────────┐
│                      VisionGPT                         │
│                                                        │
│   ┌────────────────────────────────────────────────┐   │
│   │                 Vision Encoder                 │   │
│   └───────────────────────┬────────────────────────┘   │
│                           │ (None, 7, 7, 1280)         │
│                           ▼                            │
│   ┌────────────────────────────────────────────────┐   │
│   │                  Fusion Layer                  │   │
│   └───────────────────────┬────────────────────────┘   │
│                           │ (None, 49, 256)            │
│                           ▼                            │
│   ┌────────────────────────────────────────────────┐   │
│   │              Transformer Decoder               │   │
│   └────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

### A. Vision Encoder
* **Class File:** `models/vision_encoder.py`
* **Backbone:** EfficientNet-B0.
* **Input Resolution:** `(None, 224, 224, 3)` (RGB).
* **Output Tensor:** `(None, 7, 7, 1280)`.
* **Description:** Extracts raw spatial grid features from images. By default, the encoder parameters are **frozen** during VQA training to retain pre-trained ImageNet priors and prevent catastrophic forgetting.

### B. Fusion Layer
* **Class File:** `models/fusion_layer.py`
* **Input Tensor:** `(None, 7, 7, 1280)`.
* **Output Tensor:** `(None, 49, 256)`.
* **Description:** Flattens the $7 \times 7$ grid into a 49-element sequence and projects the 1280-dimensional features through a linear layer with a trainable weight matrix of size `(1280, 256)`. This aligns the visual features with the embedding dimension ($256$) of the Transformer.

### C. Transformer Decoder (Answer Decoder)
* **Class File:** `models/answer_decoder.py`
* **Layers:** 3 Decoder Blocks, 1 Dense Output layer.
* **Embed Dim:** 256.
* **Feed Forward Dim:** 512.
* **Attention Heads:** 4.
* **Causal Masking:** Applied over text tokens to prevent the model from attending to future tokens during training.
* **Description:** Autoregressively generates predictions. The text inputs are embedded and attend to the projected image sequence (`(None, 49, 256)`) via cross-attention.

---

## 2. Text Tokenizer

* **Vocabulary File:** `vocab.json`
* **Vocabulary Size:** 10,000 tokens.
* **Control Tokens:**
  * `0`: Padding token.
  * `1`: Mask token.
  * `3`: Start-of-Sequence token.
  * `4`: End-of-Sequence token.
  * `5`: Answer-marker token.

---

## 3. Computational Dataflow

### A. Training Flow (Forward & Backward Pass)
1. **Data Load:** Batches of `(images, text_tokens)` are loaded from disk.
2. **Forward Pass:**
   * Images are processed by the Vision Encoder to yield features.
   * Features are mapped by the Fusion Layer.
   * Text tokens are masked and processed by the Decoder.
   * Cross-attention attends to the image features.
   * Logits are projected by the final classifier.
3. **Loss Computation:** Categorical cross-entropy loss is computed against targets using a target mask.
4. **Gradient Extraction:** Gradients are calculated via `tf.GradientTape()` only for trainable layers (Fusion Layer and Answer Decoder).
5. **Optimization:** Trainable variables are updated using Adam.

### B. Inference Flow (Autoregressive Generation)
1. **Image Encoding:** The input image is preprocessed and encoded to a static feature sequence.
2. **First Token:** The text sequence is initialized with the start token `[3]`.
3. **Autoregressive Loop:**
   * Run the model on the current sequence: `logits = model((image, current_sequence))`
   * Extract the last step logits: `next_token_id = argmax(logits[:, -1, :])`
   * Append `next_token_id` to the sequence.
   * Repeat until `next_token_id == 4` (end token) or maximum sequence length is reached.
4. **Decoding:** Map token IDs to words to produce the final string.
