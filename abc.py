# check_oov.py — run in your visionGPT env
from preprocessing.text_preprocessor import TextPreprocessor
from dataset_tools.coco_adapter import COCOAdapter
from dataset_tools.textvqa_adapter import TextVQAAdapter

coco = COCOAdapter(annotation_path="Dataset/COCO/annotations/captions_train2017.json",
                    image_directory="Dataset/COCO/train2017").load(limit=100000)
textvqa = TextVQAAdapter(annotation_path="future_datasets/textvqa/annotations/TextVQA_0.5.1_train.json",
                          image_directory="future_datasets/textvqa/train_images/train_images").load()

tp = TextPreprocessor(vocab_size=10000, max_length=50)
texts = tp.build_dataset_texts(coco + textvqa)
tp.build_vocabulary(texts)

tokens = tp.process(texts)
unk_id = tp.get_token_id("[UNK]")
total = int((tokens != 0).numpy().sum())          # non-pad tokens
unk   = int((tokens == unk_id).numpy().sum())
print(f"OOV rate: {unk/total:.2%}  ({unk}/{total} tokens)")