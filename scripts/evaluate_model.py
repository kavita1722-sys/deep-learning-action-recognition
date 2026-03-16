"""
evaluate_model.py

This script is used to evaluate the performance of my trained CNN-LSTM
Human Action Recognition model. I wrote this code to understand how a
trained deep learning model can be tested on a dataset and how its
performance can be analyzed using different evaluation metrics.

Main idea of this script:

1. Loading the Dataset
   I use the custom ActionDataset class that I implemented in
   dataset_loader.py. This loader prepares sequences of video frames
   that represent different human actions.

2. Model Architecture
   The model combines two important components:

   - CNN (ResNet18):
     I use a pretrained ResNet18 model as the feature extractor.
     The final classification layer is removed so that the CNN only
     extracts spatial features from each frame.

   - LSTM:
     Since video data has a temporal dimension, I use an LSTM layer
     to learn the sequence relationship between consecutive frames.

   - Fully Connected Layer:
     The final layer converts the LSTM output into predictions for
     the six action classes.

3. Loading Trained Weights
   The trained model parameters are loaded from:
       models/action_model.pth

   This allows the script to evaluate the already trained model
   without retraining it.

4. Prediction Process
   The dataset is passed through the model in batches using a
   PyTorch DataLoader. For each batch:
       • Frames are processed by the CNN
       • Temporal features are learned by the LSTM
       • Final predictions are generated

5. Evaluation Metrics
   To analyze model performance, I calculate:

       • Accuracy Score
       • Classification Report (precision, recall, F1-score)
       • Confusion Matrix

   These metrics help understand how well the model performs for
   each action category.

6. Confusion Matrix Visualization
   I generate a heatmap of the confusion matrix using Seaborn and
   Matplotlib. This visualization helps me see which actions are
   correctly predicted and which ones are commonly misclassified.

   The confusion matrix is also saved to:
       results/confusion_matrix.png

Purpose of this script:

This implementation helped me understand:
    • How trained deep learning models are evaluated
    • How predictions and ground truth labels are compared
    • How classification metrics are calculated
    • How confusion matrices help interpret model performance

This evaluation step is an important part of my project
"Explainable Human Action Recognition with CNN-LSTM".
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from dataset_loader import ActionDataset
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Device
device = torch.device("cpu")

actions = [
    "Basketball",
    "CricketBowling",
    "JumpingJack",
    "Punch",
    "PushUps",
    "TaiChi"
]

# CNN backbone
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()

# Model
class ActionModel(nn.Module):

    def __init__(self, cnn, hidden_size=256, num_classes=6):
        super(ActionModel, self).__init__()

        self.cnn = cnn
        self.lstm = nn.LSTM(512, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        batch, seq, c, h, w = x.size()

        x = x.view(batch * seq, c, h, w)

        features = self.cnn(x)

        features = features.view(batch, seq, -1)

        lstm_out, _ = self.lstm(features)

        out = self.fc(lstm_out[:, -1, :])

        return out


# Load dataset
dataset = ActionDataset("frames")

loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Load model
model = ActionModel(resnet)

model.load_state_dict(
    torch.load("models/action_model.pth", map_location=device)
)

model.to(device)
model.eval()

all_preds = []
all_labels = []

total_batches = len(loader)

print("Evaluating model...")
print("Total batches:", total_batches)

with torch.no_grad():

    for i, (images, labels) in enumerate(loader):

        if i % 100 == 0:
            print(f"Processed {i}/{total_batches} batches")

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# Convert to numpy
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Accuracy
accuracy = accuracy_score(all_labels, all_preds)

print("\nModel Accuracy:", accuracy * 100, "%")

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=actions))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=actions,
    yticklabels=actions
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("results/confusion_matrix.png")

plt.show()