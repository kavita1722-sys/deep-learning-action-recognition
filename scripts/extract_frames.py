"""
extract_frames.py

This script is used to convert videos from the UCF101 dataset into
individual image frames. I wrote this script to understand how raw
video data can be prepared for deep learning models, especially for
Human Action Recognition tasks.

Since my CNN-LSTM model cannot process videos directly, I first
extract frames from each video and store them as image sequences.
These frames are later used as input to the dataset loader during
training and evaluation.

Dataset Structure (Original UCF101 videos):

    dataset/UCF101/UCF-101/
        Basketball/
            video1.avi
            video2.avi
        CricketBowling/
        JumpingJack/
        ...

Frame Extraction Output Structure:

    frames/
        Basketball/
            video1/
                frame_0000.jpg
                frame_0001.jpg
                ...
        CricketBowling/
        JumpingJack/
        ...

How this script works:

1. It reads all action category folders inside the dataset directory.
2. For each action class, it loops through every video file.
3. Each video is opened using OpenCV's VideoCapture.
4. Frames are read sequentially from the video.
5. Every frame is saved as a JPEG image inside a folder corresponding
   to that specific video.
6. Frame files are named in order (frame_0000.jpg, frame_0001.jpg, etc.)
   so that the temporal sequence of the video is preserved.

Important Notes:

- Preserving frame order is important because the LSTM later learns
  temporal patterns from these frame sequences.
- The extracted frames form the dataset that will be loaded by
  dataset_loader.py during model training.

Purpose of writing this script:

This implementation helped me understand:
    • How video files can be processed using OpenCV
    • How datasets must be structured for deep learning pipelines
    • How temporal data (videos) can be converted into frame sequences
    • How preprocessing is an important step before training a model

This script is part of my project:
"Explainable Human Action Recognition with CNN-LSTM".
"""
import os
import cv2

# Path to dataset
dataset_path = "C:/Users/anko1/Desktop/DeepLearning_ActionRecognition/dataset/UCF101/UCF-101"

# Path to save frames
frames_path = "C:/Users/anko1/Desktop/DeepLearning_ActionRecognition/frames"

os.makedirs(frames_path, exist_ok=True)

for action in os.listdir(dataset_path):
    action_path = os.path.join(dataset_path, action)

    if not os.path.isdir(action_path):
        continue

    save_action_path = os.path.join(frames_path, action)
    os.makedirs(save_action_path, exist_ok=True)

    for video in os.listdir(action_path):
        video_path = os.path.join(action_path, video)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        video_name = os.path.splitext(video)[0]
        video_folder = os.path.join(save_action_path, video_name)
        os.makedirs(video_folder, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_file = os.path.join(video_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_file, frame)

            frame_count += 1

        cap.release()

print("Frame extraction completed!")