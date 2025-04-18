import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(42)


# Defining the Custom Pytorch Dataset 
class inaturalist_dataset(Dataset):
    def __init__(self, data_path, transform=None):
        
        self.root_dir = Path(data_path)
        self.transform = transform

        # Get all image paths and corresponding labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # Assign class index
        for idx, class_dir in enumerate(sorted(os.listdir(self.root_dir))):
            class_path = self.root_dir / class_dir
            if os.path.isdir(class_path):
                self.class_to_idx[class_dir] = idx
                for img_file in os.listdir(class_path):
                    if img_file.endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(class_path / img_file)
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    
