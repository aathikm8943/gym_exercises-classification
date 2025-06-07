# VideoMAE-Based Exercise Action Classification

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Folder Structure](#folder-structure)   
- [Install dependencies](#installation)  
- [Usage](#usage)  
  - [Training Pipeline](#training-pipeline)  
  - [Prediction Pipeline](#prediction-pipeline)  
  - [Streamlit App](#streamlit-app)  
- [Model & Performance](#model--performance)  
- [Customization](#customization)  

---

## Project Overview

This repository contains a complete implementation for human exercise classification from video data using a VideoMAE-based model architecture.

The system performs:

- **Video frame extraction and preprocessing**  
- **Training and validation of a VideoMAE classifier**  
- **Inference/prediction on unseen video files**  
- **Interactive visualization using Streamlit with MediaPipe overlays**

The main goal is to classify exercises such as push-ups, lateral raises, and squats based on video input with high accuracy.

---

## Folder Structure

├── config.py # Global config constants (paths, labels, hyperparams)
├── requirements.txt # Python dependencies
├── README.md # This documentation file
├── train_pipeline.py # Entrypoint for model training
│
├── src/
│ ├── components/
│ │ ├── model.py # VideoMAEClassifier model and utilities
│ │ ├── dataloader.py # Custom PyTorch Dataset and DataLoader
│ │ ├── preprocessing.py # Video frame extraction and transform pipeline
│ │
│ ├── helpers/
│ │ ├── media_pipe.py # MediaPipe pose estimation overlay utilities
│ │ └── zip_extraction.py # Dataset zip extraction helper functions
│ │
│ ├── pipelines/
│ │ ├── train_pipeline.py # Training loop with logging and validation
│ │ └── prediction_pipeline.py # Video prediction/inference pipeline
│ │
│ ├── loggingInfo/
│ │ └── loggingFile.py # Logger configuration
│ │
│ └── app/
│ └── streamlit_app.py # Streamlit UI for video upload and real-time prediction

## Install dependencies

```
pip install -r requirements.txt
```

## Usage

**Training Pipeline**

Run model training with your dataset.

```python src/pipelines/train_pipeline.py```

Performs train/validation split internally.

Logs training progress with loss values.

Saves the trained model weights at MODEL_PATH.

**Prediction Pipeline**
Predict class label on a .mp4 video file:
```
python src/pipelines/prediction_pipeline.py
```
Modify video_path in the script or adapt it for CLI usage.

Returns predicted class, confidence score, and probabilities for all classes.

**Streamlit App**

Launch interactive web app for video upload and real-time predictions:
```
streamlit run src/app/streamlit_app.py
```
Upload .mp4 videos through the browser.

See class predictions and confidence scores instantly.

## Model & Performance

**Model:** VideoMAE-based classifier fine-tuned for 3 exercise classes.

**Input:** Video frames resized to 224x224, sampled uniformly with NUM_FRAMES=16.

**Training:** Cross-entropy loss optimized with AdamW and linear LR scheduler.

**Metrics (example from validation):**

Accuracy: ~98%
Average Loss: ~0.0254 after 6 epochs

## Customization

**Add new action classes:** Update LABELS and prepare corresponding dataset folders.

**Change input size or frames:** Modify FRAME_SIZE and NUM_FRAMES in config.py.

**Adjust training params:** Tune LEARNING_RATE, BATCH_SIZE, and NUM_EPOCHS for your hardware/data.

**MediaPipe overlays:** Enable/disable pose skeleton visualization in Streamlit via media_pipe.py.
