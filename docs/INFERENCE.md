# Inference execution & Deployment

This document describes how to execute inference using VisionGPT v3 models. It covers command-line execution, Python API calls, batch inputs evaluation, interactive modes, and output formatting.

---

## 1. Command-Line Interface (CLI)

We provide a general inference script `inference.py` at the workspace root directory.

### A. Run Single Image Captioning
Generates a caption describing the contents of a target image.
```bash
python inference.py --image test_images/cat.jpg --task caption
```

### B. Run Visual Question Answering (VQA)
Answers a natural language question about an input image.
```bash
python inference.py --image test_images/street.jpg --task vqa --question "What color is the bus?"
```

### C. Run OCR-aware Question Answering
Extracts text present in the image to answer a target question.
```bash
python inference.py --image test_images/receipt.jpg --task ocr --question "What is the total price?"
```

### D. Run Interactive VQA Console Mode
Starts an interactive terminal session where you can query an input image repeatedly:
```bash
python inference.py --image test_images/living_room.jpg --interactive
```
*Console prompt:*
```text
Loaded image: test_images/living_room.jpg
[Query]: What is on the table?
[Answer]: a laptop and a coffee mug
[Query]: Is there a cat in the room?
[Answer]: yes, sleeping on the couch
```

---

## 2. Python API Usage

To call inference programmatically inside your scripts, instantiate the model and load its weights:

```python
import tensorflow as tf
from models.visiongpt import VisionGPT
from preprocessing.image import load_and_preprocess_image

# 1. Instantiate the model and load weights
model = VisionGPT(vocab_size=10000)
# Build model weights
dummy_img = tf.zeros((1, 224, 224, 3))
dummy_txt = tf.zeros((1, 5), dtype=tf.int64)
_ = model((dummy_img, dummy_txt), training=False)

model.load_weights("checkpoints/visiongpt_coco.keras")

# 2. Preprocess input image and question tokens
image = load_and_preprocess_image("test_images/cat.jpg")  # shape: (1, 224, 224, 3)

# Tokens: Start token (3), question tokens, answer marker (5)
# In real applications, use a text tokenizer to encode text into integer tokens
text_tokens = tf.constant([[3, 105, 42, 80, 5]], dtype=tf.int64)

# 3. Call the model forward pass
logits = model((image, text_tokens), training=False)  # shape: (1, 5, 10000)

# 4. Extract token IDs and decode
predicted_ids = tf.argmax(logits, axis=-1).numpy()[0]
print("Predicted Token IDs:", predicted_ids)
```

---

## 3. Batch Inference

For high-throughput evaluations, execute batch inference:

```python
import tensorflow as tf
import numpy as np

# Prepare batch of 4 images and text tokens
batch_images = tf.random.normal((4, 224, 224, 3))
batch_tokens = tf.constant([[3, 10, 20, 5]] * 4, dtype=tf.int64)

# Run batch model call
batch_logits = model((batch_images, batch_tokens), training=False)

# Shape: (4, 4, 10000)
print("Batch Logits Shape:", batch_logits.shape)
```

---

## 4. Output Logits Token Decoding

To convert the raw logits output of the model back into readable text:
1. Apply `argmax` over the last dimension to get the most likely token ID for each step.
2. Filter out padding and start/end control tokens (e.g. `start_token_id=3`, `end_token_id=4`).
3. Load the vocabulary dictionary (`vocab.json`) and map the integer token IDs back to word strings.
4. Join the words with spaces to construct the final text string response.
5. In autoregressive sampling, feed the predicted token back into the input sequence for the next step calculation until the end token is generated or the maximum sequence length is reached.
