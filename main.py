from src.components.model import VideoMAEClassifier
from src.components.preprocessing import VideoPreprocessor
from src.components.dataloader import GymVideoDatasetTorch
from src.components.evaluate import evaluate_model

from src.helpers.extract_zip import ZipExtractor
# from src.helpers.preprocess_video import VideoPreprocessor

from src.loggingInfo.loggingFile import logging

from config import INPUT_DIR, EXTRACTED_VIDEOS_DIR, MODEL_PATH

def extracting_zip_files():
    # Step 1: Extract
    extractor = ZipExtractor(INPUT_DIR, EXTRACTED_VIDEOS_DIR)
    extractor.extract_all()

def main():
    print("Loading model and data...")
    logging.info("Loading model and data...")
    preprocessor = VideoPreprocessor()
    dataset = GymVideoDatasetTorch(preprocessor)

    model_obj = VideoMAEClassifier(num_classes=3)
    model_obj.load(MODEL_PATH)

    logging.info("Model and data loaded successfully.")
    print("Evaluating model...")
    evaluate_model(model_obj, dataset)

if __name__ == "__main__":
    # extracting_zip_files()
    main()
