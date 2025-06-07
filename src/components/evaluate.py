from sklearn.metrics import classification_report
import numpy as np
import torch

def evaluate_model(model_obj, dataset):
    y_true = []
    y_pred = []

    for frames, label in dataset:
        probs = model_obj.predict(frames)  # shape: (1, num_classes)
        pred_class = np.argmax(probs)
        y_true.append(label)
        y_pred.append(pred_class)

    report = classification_report(y_true, y_pred, target_names=["Bicep Curl", "Lateral Raise", "Squat"])
    print("🔍 Classification Report:\n")
    print(report)
