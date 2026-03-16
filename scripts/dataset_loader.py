"""
dataset_loader.py

This file contains the custom PyTorch Dataset class that I implemented to load
the Human Action Recognition dataset for training my CNN-LSTM model.

I created this class mainly to understand how video datasets can be converted
into frame sequences that deep learning models can process efficiently.
Instead of feeding the entire video directly, each video is represented as a
sequence of frames.

Key ideas I implemented here:

1. Dataset Organization
   The dataset is expected to be organized in the following structure:

       dataset/
           Action1/
               video1/
                   frame1.jpg
                   frame2.jpg
                   ...
               video2/
           Action2/
               video1/
               video2/

   Each action folder represents a class label.

2. Label Generation
   I automatically generate numerical labels by enumerating the action folders.

3. Frame Sampling
   Since videos may contain many frames, I sample a fixed number of frames
   (sequence_length = 16) from each video. This ensures that all sequences
   have equal length for training the CNN-LSTM model.

4. Image Preprocessing
   Each frame is processed using torchvision transforms:
       • Resize to 224×224
       • Convert to tensor
       • Normalize using ImageNet mean and standard deviation

   This helps the CNN backbone learn features effectively.

5. Tensor Formation
   Frames from each video are stacked into a tensor of shape:

       (sequence_length, 3, 224, 224)

   which represents a temporal sequence of images.

This implementation helped me understand:
    • How PyTorch Dataset and DataLoader work
    • How video data can be converted into frame sequences
    • How preprocessing and normalization are applied
    • How labels and samples are prepared for training

This loader is used during the training stage of my project
"Explainable Human Action Recognition with CNN-LSTM".
"""
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ActionDataset(Dataset):

    def __init__(self, root_dir, sequence_length=16):

        self.root_dir = root_dir
        self.sequence_length = sequence_length

        self.actions = sorted(os.listdir(root_dir))

        self.data = []

        for label, action in enumerate(self.actions):

            action_path = os.path.join(root_dir, action)

            if not os.path.isdir(action_path):
                continue

            for video in os.listdir(action_path):

                video_path = os.path.join(action_path, video)

                frames = sorted(os.listdir(video_path))

                if len(frames) < self.sequence_length:
                    continue

                # -------- SAMPLE FRAMES ACROSS VIDEO --------

                step = max(1, len(frames) // self.sequence_length)

                sampled_frames = frames[::step][:self.sequence_length]

                sequence_paths = [
                    os.path.join(video_path, f)
                    for f in sampled_frames
                ]

                self.data.append((sequence_paths, label))


        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])


    def __len__(self):

        return len(self.data)


    def __getitem__(self, idx):

        frame_paths, label = self.data[idx]

        images = []

        for frame in frame_paths:

            img = Image.open(frame).convert("RGB")

            img = self.transform(img)

            images.append(img)

        images = torch.stack(images)

        return images, label