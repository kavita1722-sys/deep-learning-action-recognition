"""
predict_action.py

This script is responsible for performing action prediction using the
trained CNN-LSTM model in my project "Explainable Human Action Recognition
with CNN-LSTM".

I implemented this module to understand how a trained deep learning model
can be used for real-time inference on video data, webcam streams, or
frame sequences. The script processes frames, extracts spatial features
using a CNN, learns temporal relationships using an LSTM, and finally
predicts the human action being performed.

Main components of this script:

1. Action Classes
   I define a list of six action categories that my model was trained on:
       • Basketball
       • CricketBowling
       • JumpingJack
       • Punch
       • PushUps
       • TaiChi

   These labels correspond to the dataset classes used during training.

2. Action Explanations
   To make the system more explainable, I created a dictionary that maps
   each predicted action to a short human-readable explanation. This
   helps users understand what the detected action represents.

3. Image Preprocessing
   Frames are transformed using the same preprocessing steps used during
   training:
       • Resize to 224 × 224
       • Convert to tensor
       • Normalize using ImageNet statistics

   This ensures consistency between training and inference.

4. CNN-LSTM Model
   The architecture used in this script is the same model used during
   training:

       • CNN (ResNet18)
         Extracts spatial features from each frame.

       • LSTM
         Learns temporal relationships between consecutive frames.

       • Fully Connected Layer
         Produces the final action classification output.

5. Model Loading
   The trained weights are loaded from:

       models/action_model.pth

   This allows the system to perform predictions without retraining.

6. Prediction Function
   The predict() function receives a sequence of frames and:
       • Applies preprocessing
       • Passes frames through the CNN
       • Uses LSTM to learn temporal patterns
       • Calculates class probabilities using Softmax
       • Returns predicted action, confidence score, and explanation

7. Frame Buffer Prediction
   The system uses a buffer of 16 frames to perform predictions.
   This sequence length matches the configuration used during training.

8. Video File Mode
   The run_video() function processes frames from a video file and
   periodically performs predictions using the frame buffer.

9. Webcam Mode
   The run_webcam() function enables real-time action recognition
   using a webcam. Predictions are displayed directly on the video
   stream along with:
       • Action label
       • Confidence percentage
       • Short explanation

Purpose of this script:

This implementation helped me understand:
    • How trained deep learning models perform inference
    • How CNN and LSTM models can be combined for video analysis
    • How real-time prediction works using frame buffers
    • How to integrate computer vision with OpenCV for live streams
    • How explainability can be added to AI systems

This module is a key component of my project that connects the trained
model with real-time applications such as webcam and video analysis.
"""
import torch
import torch.nn as nn
import cv2
from torchvision import models, transforms
from PIL import Image
from collections import deque
import torch.nn.functional as F

device = torch.device("cpu")

actions = [
    "Basketball",
    "CricketBowling",
    "JumpingJack",
    "Punch",
    "PushUps",
    "TaiChi"
]

# Explanations 
EXPLANATIONS = {
    "PushUps": "Push-ups strengthen the chest, shoulders, and arms.",
    "JumpingJack": "Jumping jacks are a full-body aerobic exercise.",
    "Basketball": "Basketball involves running, jumping, and throwing a ball into a hoop.",
    "Punch": "Punching involves striking forward with the fist.",
    "CricketBowling": "Cricket bowling is delivering the ball to the batsman.",
    "TaiChi": "Tai Chi is a slow martial art focused on balance and control."
}

# Correct Transform for ResNet
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# CNN Backbone
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Identity()


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


# Load Model
model = ActionModel(resnet)

model.load_state_dict(
    torch.load("models/action_model.pth", map_location=device, weights_only=True)
)

model.to(device)
model.eval()


# Prediction 
def predict(frames):

    images = []

    for frame in frames:
        img = Image.fromarray(frame)
        img = transform(img)
        images.append(img)

    images = torch.stack(images).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(images)

        probs = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    action = actions[predicted.item()]
    confidence = confidence.item() * 100

    explanation = EXPLANATIONS.get(action, "Human physical activity detected.")

    return action, confidence, explanation


#  Frame Buffer Mode (Used by Flask Webcam)
def run_frames(frame_buffer):

    try:
        return predict(frame_buffer)

    except Exception:
        return "Processing...", 0, ""


#  Video File Mode 
def run_video(path):

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("ERROR: Could not open video file.")
        return None, None, None

    frame_buffer = deque(maxlen=16)
    frame_count = 0

    action = "Detecting..."
    confidence = 0
    explanation = ""

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))

        small = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        frame_buffer.append(rgb)
        frame_count += 1

        # Predict every 8 frames to reduce lag
        if frame_count % 12 == 0 and len(frame_buffer) == 16:

            action, confidence, explanation = predict(list(frame_buffer))

    cap.release()

    return action, confidence, explanation


# Webcam Mode (Standalone Testing)
def run_webcam():

    print("Trying to open webcam...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: Webcam not detected.")
        return

    cv2.namedWindow("Action Recognition - Webcam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Action Recognition - Webcam", 900, 700)

    frame_buffer = deque(maxlen=16)
    frame_count = 0

    action = "Detecting..."
    confidence = 0
    explanation = ""

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        display_frame = cv2.resize(frame, (900, 700))

        small = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        frame_buffer.append(rgb)
        frame_count += 1

        if frame_count % 12 == 0 and len(frame_buffer) == 16:

            action, confidence, explanation = predict(list(frame_buffer))

        # Draw Labels on Frame

        cv2.putText(
            display_frame,
            f"Action: {action}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.putText(
            display_frame,
            f"Confidence: {confidence:.1f}%",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            explanation[:60],
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Action Recognition - Webcam", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main
if __name__ == "__main__":

    mode = input("Enter mode (webcam/video): ").strip().lower()

    if mode == "webcam":
        run_webcam()

    elif mode == "video":

        video_path = input("Enter video path: ")

        action, confidence, explanation = run_video(video_path)

        print("\nPredicted Action:", action)
        print("Confidence:", confidence)
        print("Explanation:", explanation)

    else:
        print("Invalid mode. Choose 'webcam' or 'video'.")