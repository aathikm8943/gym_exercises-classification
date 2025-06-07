# Input and Output file paths and configurations for a video classification task
INPUT_DIR = "data/input_dir"
EXTRACTED_VIDEOS_DIR = "data/extracted_videos"
PROCESSED_VIDEOS_DIR = "data/processed_videos"
OUTPUT_DIR = "data/output_videos"

# Configuration for a video classification task using a pre-trained model
LABELS = ["Bicep Curl", "Lateral Raise", "Squat"]
NUM_FRAMES = 16
FRAME_SIZE = (224, 224)
MODEL_PATH = "models/videomae_model_updated1.pt"
FIRST_ITERATION_MODEL_PATH = "models/videomae_model.pth"

VALIDATION_SPLIT = 0.2
NUM_EPOCHS = 8
BATCH_SIZE = 4
LEARNING_RATE = 0.0001

