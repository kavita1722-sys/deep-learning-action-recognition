"""
test_dataset.py

This script is a small utility that I created to verify whether my
custom dataset loader (ActionDataset) is working correctly before
training the model.

While developing the Human Action Recognition pipeline, I wanted to
make sure that the dataset structure, frame loading, and batch
generation were functioning properly. Instead of directly using the
dataset in training, I wrote this test script to inspect the output
of the DataLoader.

What this script does:

1. Import Custom Dataset
   It imports the ActionDataset class from dataset_loader.py, which
   loads frame sequences extracted from the videos.

2. Create Dataset Instance
   The dataset is initialized using the "frames" directory, which
   contains the extracted video frames organized by action class.

3. Check Dataset Size
   The script prints the total number of samples in the dataset so
   that I can confirm the loader is reading all video sequences
   correctly.

4. Create DataLoader
   A PyTorch DataLoader is used to load the dataset in small batches.
   I set the batch size to 2 just for testing purposes.

5. Inspect Batch Output
   The script prints:
       • The shape of the image batch
       • The corresponding labels

   This helps verify that:
       - Frames are correctly stacked into tensors
       - The sequence dimension is correct
       - Labels are assigned properly

6. Stop After First Batch
   The loop breaks after the first batch because the purpose of this
   script is only to check the dataset format, not to iterate through
   the entire dataset.

Purpose of this script:

This test helped me understand:
    • How PyTorch Dataset and DataLoader work together
    • How video frames are converted into tensor batches
    • Whether the dataset structure is compatible with my model

I mainly wrote this script for debugging and verifying my data
pipeline before starting the model training process in my project
"Explainable Human Action Recognition with CNN-LSTM".
"""
from dataset_loader import ActionDataset
from torch.utils.data import DataLoader

dataset = ActionDataset("frames")

print("Total samples:", len(dataset))

loader = DataLoader(dataset, batch_size=2, shuffle=True)

for images, labels in loader:

    print("Batch shape:", images.shape)
    print("Labels:", labels)

    break