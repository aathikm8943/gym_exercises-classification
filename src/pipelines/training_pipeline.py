import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.components.model import VideoMAEClassifier
from src.components.preprocessing import VideoPreprocessor
from src.components.dataloader import GymVideoDatasetTorch

from src.loggingInfo.loggingFile import logging

from config import MODEL_PATH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, LABELS

def video_collate_fn(batch):
    frames, labels = zip(*batch)
    return list(frames), torch.tensor(labels, dtype=torch.long)

class VideoTrainer:
    def __init__(self, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._setup_data()
        self._setup_model()

    def _setup_data(self):
        logging.info("Initializing dataset and preprocessing...")
        preprocessor = VideoPreprocessor()
        dataset = GymVideoDatasetTorch(preprocessor)
        labels = [int(dataset[i][1]) for i in range(len(dataset))]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset = Subset(dataset, val_idx)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=video_collate_fn, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=video_collate_fn)

        logging.info(f"Dataset loaded: Total={len(dataset)}, Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")

    def _setup_model(self):
        logging.info("Initializing model...")
        model_obj = VideoMAEClassifier(num_classes=len(LABELS), custom_head=True)
        self.model = model_obj.model
        self.processor = model_obj.processor
        self.device = model_obj.device

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        num_training_steps = self.num_epochs * len(self.train_loader)
        self.lr_scheduler = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    def train(self):
        logging.info("Starting training loop...")
        best_val_acc = 0
        patience = 3
        counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for frames, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                labels = labels.to(self.device)
                inputs = self.processor(frames, return_tensors="pt")
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)

                outputs = self.model(**inputs)
                logits = outputs.logits
                loss = self.loss_fn(logits, labels)

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)

            val_loss, val_acc = self.evaluate()

            logging.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                torch.save(self.model.state_dict(), MODEL_PATH)
                logging.info(f"✅ Model saved at epoch {epoch+1} with Val Accuracy: {val_acc:.4f}")
            else:
                counter += 1
                if counter >= patience:
                    logging.info("⏹️ Early stopping triggered.")
                    break

    def evaluate(self):
        self.model.eval()
        all_labels = []
        all_preds = []
        total_loss = 0

        with torch.no_grad():
            for frames, labels in self.val_loader:
                labels = labels.to(self.device)
                inputs = self.processor(frames, return_tensors="pt")
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)

                outputs = self.model(**inputs)
                logits = outputs.logits
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        acc = np.mean(np.array(all_preds) == np.array(all_labels))

        logging.info("\n🔍 Classification Report:\n" + classification_report(all_labels, all_preds, target_names=LABELS))

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        return avg_loss, acc

if __name__ == "__main__":
    trainer = VideoTrainer()
    trainer.train()

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# import torch
# from torch.utils.data import DataLoader, random_split
# from torch.optim import AdamW
# from transformers import get_scheduler
# from tqdm import tqdm

# from src.components.model import VideoMAEClassifier
# from src.components.preprocessing import VideoPreprocessor
# from src.components.dataloader import GymVideoDatasetTorch
# from config import MODEL_PATH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, LABELS, VALIDATION_SPLIT
# from src.loggingInfo.loggingFile import logging


# def video_collate_fn(batch):
#     frames, labels = zip(*batch)
#     return list(frames), torch.tensor(labels)


# def evaluate(model, processor, dataloader, device):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for frames, labels in dataloader:
#             labels = labels.to(device)
#             inputs = processor(frames, return_tensors="pt")
#             for key in inputs:
#                 inputs[key] = inputs[key].to(device)

#             outputs = model(**inputs)
#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     accuracy = correct / total if total > 0 else 0
#     model.train()
#     return accuracy


# def train_pipeline():
#     try:
#         logging.info("Starting training pipeline...")

#         # Prepare dataset and split
#         preprocessor = VideoPreprocessor()
#         full_dataset = GymVideoDatasetTorch(preprocessor)
#         total_size = len(full_dataset)
#         val_size = int(VALIDATION_SPLIT * total_size)
#         train_size = total_size - val_size

#         train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
#         train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=video_collate_fn, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=video_collate_fn)

#         logging.info(f"Dataset loaded: Total={total_size}, Train={train_size}, Val={val_size}")

#         # Initialize model
#         model_obj = VideoMAEClassifier(num_classes=len(LABELS))
#         model = model_obj.model
#         processor = model_obj.processor
#         device = model_obj.device

#         optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
#         num_training_steps = NUM_EPOCHS * len(train_loader)
#         lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
#         loss_fn = torch.nn.CrossEntropyLoss()

#         for epoch in range(NUM_EPOCHS):
#             model.train()
#             total_loss = 0

#             for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
#                 labels = labels.to(device)
#                 inputs = processor(frames, return_tensors="pt")
#                 for key in inputs:
#                     inputs[key] = inputs[key].to(device)

#                 outputs = model(**inputs)
#                 logits = outputs.logits
#                 loss = loss_fn(logits, labels)

#                 loss.backward()
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()

#                 total_loss += loss.item()

#             avg_loss = total_loss / len(train_loader)
#             val_acc = evaluate(model, processor, val_loader, device)

#             logging.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Val Accuracy = {val_acc:.4f}")

#         # Save model
#         model_obj.save(MODEL_PATH)
#         logging.info(f"Training complete. Model saved at: {MODEL_PATH}")

#     except Exception as e:
#         logging.error(f"Training failed due to error: {str(e)}", exc_info=True)


# if __name__ == "__main__":
#     train_pipeline()
