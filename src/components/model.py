import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import MODEL_PATH, LABELS

class VideoMAEClassifier:
    def __init__(self, num_classes=3, pretrained="MCG-NJU/videomae-base", custom_head=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = VideoMAEImageProcessor.from_pretrained(pretrained)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            pretrained,
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # allows head reshaping
        ).to(self.device)

    def preprocess(self, frames):
        # frames: list of PIL images or numpy RGB arrays
        inputs = self.processor(frames, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict(self, frames):
        self.model.eval()
        inputs = self.preprocess(frames)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs.cpu().numpy()
    
    def save(self, path=MODEL_PATH):
        torch.save(self.model.state_dict(), path)

    def load(self, path, pretrained="MCG-NJU/videomae-base"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        return self.model
