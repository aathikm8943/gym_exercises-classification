import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import EXTRACTED_VIDEOS_DIR, PROCESSED_VIDEOS_DIR, NUM_FRAMES, FRAME_SIZE, LABELS

import cv2
import os
from PIL import Image
from typing import Tuple
from src.loggingInfo.loggingFile import logging

class VideoPreprocessor:
    def __init__(self, num_frames: int = NUM_FRAMES, frame_size: Tuple[int, int] = FRAME_SIZE):
        self.num_frames = num_frames
        self.frame_size = frame_size
        os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)

    def preprocess_all_videos(self):
        """
        Iterates through extracted videos, preprocesses them, and saves to processed folder.
        """
        for label in LABELS:
            class_dir = os.path.join(EXTRACTED_VIDEOS_DIR, label)
            output_dir = os.path.join(PROCESSED_VIDEOS_DIR, label)
            os.makedirs(output_dir, exist_ok=True)

            if not os.path.exists(class_dir):
                logging.warning(f"Class folder not found: {class_dir}")
                continue

            for filename in os.listdir(class_dir):
                if not filename.endswith(".mp4"):
                    continue
                input_path = os.path.join(class_dir, filename)
                output_path = os.path.join(output_dir, filename)
                self._extract_video_clip(input_path, output_path)
                logging.info(f"Processed video: {output_path}")

    def _extract_video_clip(self, video_path: str, save_path: str):
        """
        Load, resize, and write trimmed video with fixed frames.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < self.num_frames:
            logging.warning(f"Skipping {video_path} — not enough frames")
            return

        step = total_frames // self.num_frames
        selected_frames = []

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i % step == 0 and len(selected_frames) < self.num_frames:
                frame = cv2.resize(frame, self.frame_size)
                selected_frames.append(frame)

        cap.release()

        # Save trimmed video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 5.0, self.frame_size)
        for f in selected_frames:
            out.write(f)
        out.release()

    def extract_frames(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        step = max(1, total_frames // self.num_frames)
        selected = []

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i % step == 0 and len(selected) < self.num_frames:
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                selected.append(Image.fromarray(frame))

        cap.release()
        return selected

if __name__ == "__main__":
    processor = VideoPreprocessor()
    processor.preprocess_all_videos()