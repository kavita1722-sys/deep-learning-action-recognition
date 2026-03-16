"""
train_model.py

This script is responsible for training the CNN-LSTM model used in my
project "Explainable Human Action Recognition with CNN-LSTM".

I wrote this script to understand the complete deep learning training
pipeline for video-based action recognition. The model learns to
recognize human activities from sequences of video frames that were
extracted earlier from the dataset.

Overview of how this training script works:

1. Dataset Loading
   The training data consists of frames extracted from videos in the
   UCF101 dataset. These frames are organized by action category and
   video sequence.

   I use my custom ActionDataset class (implemented in dataset_loader.py)
   to load sequences of frames. Each sample contains:
       • A sequence of frames (16 frames per video)
       • The corresponding action label

   A PyTorch DataLoader is then used to load the dataset in batches
   during training.

2. Device Configuration
   The script automatically checks whether a GPU is available and uses
   CUDA if possible. Otherwise, it falls back to CPU.

3. CNN Feature Extraction
   I use a pretrained ResNet18 model as the CNN backbone. The final
   classification layer is removed so that the network acts as a
   feature extractor for each video frame.

   This allows the model to learn spatial features such as body
   posture, motion cues, and object interactions.

4. LSTM for Temporal Learning
   Since actions occur over time, I use an LSTM layer to capture the
   temporal relationship between consecutive frames in the sequence.

   The pipeline works as follows:
       Frames → CNN (spatial features) → LSTM (temporal learning)
       → Fully Connected Layer → Action Prediction

5. Loss Function and Optimizer
   I use CrossEntropyLoss because this is a multi-class classification
   problem. The optimizer used is Adam with a small learning rate to
   ensure stable training.

6. Training Loop
   The model is trained for multiple epochs. During each epoch:
       • Batches of video frame sequences are passed through the model
       • Predictions are compared with true labels
       • Loss is computed
       • Gradients are calculated using backpropagation
       • Model weights are updated using the optimizer

   The total loss for each epoch is printed to monitor training progress.

7. Saving the Trained Model
   After training completes, the trained model weights are saved to:

       models/action_model.pth

   This file is later used during evaluation and real-time prediction.

Purpose of this implementation:

This training script helped me understand:
    • How deep learning models are trained using PyTorch
    • How CNN and LSTM architectures can be combined for video analysis
    • How frame sequences represent temporal information
    • How loss functions and optimizers update model parameters
    • How trained models are saved for later inference

This script forms the core training component of my Human Action
Recognition system.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader

# import the single dataset loader used across the project
from dataset_loader import ActionDataset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Path to frames dataset
frames_path = "C:/Users/anko1/Desktop/DeepLearning_ActionRecognition/frames"


# Dataset and DataLoader

print("Loading dataset...")

dataset = ActionDataset(frames_path, sequence_length=16)

print("Dataset loaded")
print("Total samples:", len(dataset))

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0   # safer on Windows
)


#  CNN Backbone 

resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()


# Action Recognition Model 

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


# Initialize model 

model = ActionModel(resnet).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)


#  Training settings 

epochs = 25


# Training loop 

print("Starting training...\n")

for epoch in range(epochs):

    total_loss = 0

    for videos, labels in dataloader:

        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


#  Save model 

os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/action_model.pth")

print("\nTraining Complete")