from torch.utils.data import Dataset
from torchvision.io import read_video
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import EXTRACTED_VIDEOS_DIR, PROCESSED_VIDEOS_DIR, NUM_FRAMES, FRAME_SIZE, LABELS

from src.loggingInfo.loggingFile import logging

class GymVideoDatasetTorch(Dataset):
    def __init__(self, preprocessor):
        self.data = []
        self.labels = LABELS
        self.preprocessor = preprocessor

        for label_idx, label_name in enumerate(self.labels):
            class_dir = os.path.join(PROCESSED_VIDEOS_DIR, label_name)
            if not os.path.exists(class_dir):
                logging.warning(f"Missing folder: {class_dir}")
                continue

            for file in os.listdir(class_dir):
                if file.endswith(".mp4"):
                    self.data.append((os.path.join(class_dir, file), label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        frames = self.preprocessor.extract_frames(video_path)
        return frames, torch.tensor(label)
