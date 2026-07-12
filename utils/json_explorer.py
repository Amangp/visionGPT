import json
from pathlib import Path

json_path = Path(r"C:\Users\1243a\OneDrive\Desktop\visionGPT\Dataset\COCO\annotations_trainval2017\annotations\captions_train2017.json")

with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)

print(type(data))
print(data.keys())
