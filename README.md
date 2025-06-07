# VideoMAE-Based Exercise Action Classification

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Folder Structure](#folder-structure)   
- [How to Run](#how-to-run)  
- [Data Sources](#data-sources)  
- [Model & Performance](#model--performance)  
- [Notes](#notes)

---

## Project Overview

A production-grade deep learning pipeline to classify human exercise movements (e.g., Bicep Curl, Lateral Raise, Squat) from video clips using . The project includes preprocessing, video-to-tensor conversion, a **fine-tuned VideoMAE** model for classification, training pipeline with evaluation logic, confusion matrices, prediction pipeline and streamlit application.

## Objective

To build a robust and modular video classification pipeline using VideoMAE (Transformer-based model) that can automatically identify types of gym exercises from raw video data.

---

## Folder Structure

```
gym_exercises_classification/
├── src/
│   ├── app/
│   │   └── streamlit_app.py
│   ├── components/
│   │   ├── dataloader.py
│   │   ├── preprocessing.py
│   │   ├── evaluate.py
│   │   └── model.py
│   ├── helpers/
│   │   └── extract_zip.py           # Extracting videos from the zip files
│   ├── loggingInfo/
│   │   └── loggingFile.py
│   └── pipelines/
│       ├── train_pipeline.py        # Main training logic with evaluation
│       └── prediction_pipeline.py   # Prediction pipeline logic
├── config.py                        # Config variables (e.g., learning rate, batch size)
├── main.py
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── template.py                      # Created the file structure from this file
```

## How to Run

**1. Setup Python Environment**
Make sure you have Python 3.11.13 installed. Then create and activate a virtual environment:
```
pip install -r requirements.txt
```

**2. Extract Raw Videos into Processed Clips**

**Step:** Download the raw gym exercise videos from the provided [resource link] and place them into the following directory:
```
data/input_dir/
```

**Run the extraction script** to convert raw videos into 16-frame .mp4 clips:
```
python src/helpers/extract_zip.py
```

This will generate short video clips categorized by exercise type in:
```
extracted_videos/
├── Bicep Curl/
├── Lateral Raise/
└── Squat/

```

**3. Train the Model**
Run the training pipeline to preprocess the data, split it, and train a VideoMAE transformer model:
```
python src/pipelines/training_pipeline.py
```
During training, the best model (based on validation accuracy) is saved to best_model.pth

**4. Run Predictions on New Video Clips**
To classify a new processed video clip (with 16 frames), use:
```
python src/pipelines/prediction_pipeline.py
```
This prints the predicted class label, confidence for all the classes of the input video.

**5. Launch the Streamlit App (Optional UI)**
To interact with the model using a web interface:
```
streamlit run app/streamlit_app.py
```
This launches a browser UI where users can upload video clips and get exercise predictions in real-time.

## Data Sources
This project uses publicly available gym workout videos for building and training the classification model. The raw video data includes three core exercise categories:

1. Bicep Curl
2. Lateral Raise
3. Squat

These videos were sourced from: [Kaggle](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video)

**Output Videos**: [link](https://drive.google.com/drive/folders/1dIZhvXTddDKmCw_ae08CLYH6LmMg-ek_?usp=sharing)

## Model Performance
The model used is a fine-tuned VideoMAE Transformer with a custom classification head. It was trained using the processed video clips over NUM_EPOCHS epochs with early stopping.

**Classification Report (Validation Set)**
After training, the best model achieved the following results:

```
              precision    recall  f1-score   support

   Bicep Curl       1.00      1.00      1.00        13
Lateral Raise       0.86      1.00      0.92         6
        Squat       1.00      0.80      0.89         5

     accuracy                           0.96        24
    macro avg       0.95      0.93      0.94        24
 weighted avg       0.96      0.96      0.96        24
```

## Notes

1. The model uses VideoMAE from HuggingFace transformers, designed for video input.
2. Input videos should be preprocessed into uniform-length short clips.
3. You can swap or extend label categories via the LABELS list in config.py.