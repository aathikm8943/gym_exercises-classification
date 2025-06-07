import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from src.components.model import VideoMAEClassifier
from config import LABELS, FRAME_SIZE, NUM_FRAMES, MODEL_PATH, FIRST_ITERATION_MODEL_PATH
from src.loggingInfo.loggingFile import logging

class VideoPredictor:
    def __init__(self, model_path: str = MODEL_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        classifer = VideoMAEClassifier(num_classes=len(LABELS))
        classifer.load(model_path)
        self.model = classifer.model

        self.transform = transforms.Compose([
            transforms.Resize(FRAME_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_frames(self, video_path: str, num_frames: int = NUM_FRAMES):
        logging.info(f"Extracting {num_frames} frames from {video_path}")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=np.int32)

        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
                frames.append(self.transform(pil_img))
        cap.release()

        if len(frames) != num_frames:
            logging.warning(f"Only {len(frames)} frames were extracted instead of {num_frames}. Padding with last frame.")
            while len(frames) < num_frames:
                frames.append(frames[-1].clone())

        return torch.stack(frames)  # Shape: [T, C, H, W]

    def predict(self, video_path: str):
        try:
            print(f"Predicting video: {video_path}")
            frames_tensor = self.extract_frames(video_path, num_frames=NUM_FRAMES)
            print(f"Extracted frames shape: {frames_tensor.shape}")
            frames_tensor = frames_tensor.unsqueeze(0).to(self.device)  # [1, T, C, H, W]

            # If model expects input shape: [B, C, T, H, W]
            # frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4)
            print(f"Input shape for model: {frames_tensor.shape}")

            with torch.no_grad():
                print("Running inference...")
                outputs = self.model(frames_tensor)
                print(f"Model outputs shape: {outputs}")
                logits = outputs.logits
                print(f"Model outputs shape: {logits.shape}")
                probs = torch.nn.functional.softmax(logits, dim=1)
                # probs = torch.nn.functional.softmax(outputs, dim=1)
                print(f"Probabilities: {probs}")
                pred_class = torch.argmax(probs, dim=1).item()
                print(f"Predicted class index: {pred_class}")
                class_name = LABELS[pred_class]
                print(f"Predicted class name: {class_name}")

            return {
                "predicted_class": class_name,
                "confidence": round(float(probs[0][pred_class]), 4),
                "probabilities": {LABELS[i]: round(float(p), 4) for i, p in enumerate(probs[0])}
            }

        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return {"error": str(e)}
        
if __name__ == "__main__":
    output_video_path = r"C:\Users\Aathi K M\Documents\JobAssessments\smartenFitTech_AI_Assessment\data\extracted_videos\squat\squat_19.mp4"
    predictor = VideoPredictor()
    result = predictor.predict(output_video_path)
    print(result)
