import os 
import json
import random 
from pathlib import Path


class dataexplorer:
    def __init__(self):
        self.base_path=Path(r"C:\Users\1243a\OneDrive\Desktop\visionGPT\Dataset")
        self.coco_path=self.base_path/"COCO"
        self.gqa_path=self.base_path/"GQA"
        self.vqa_path=self.base_path/"VQA"

        
    def check_directories(self):
        dataset={
            "COCO":self.coco_path,
            "GQA":self.gqa_path,
            "VQA":self.vqa_path
        }
        for name , path in dataset.items():
            if path.exists():
                print("Found")
            else:
                print("Not Found")
                
if __name__=="__main__":
    explorer=dataexplorer()
    explorer.check_directories()
