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
import pandas as pd
import wandb
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
import io
from data_preprocessing.custom_dataset import inaturalist_dataset
from custom_model.model import Custom_CNN

# Initialize wandb
wandb.init(project="DA6401_Assignment_2",entity="ch21b033-iit-madras",name= "Predictions_Grid")

np.random.seed(42)

# Best set of hyperparameters 
batch_size = 64
filter1,filter2, filter3, filter4, filter5 = [16,32,48,64,96]
activation_func = 'relu'
dropout_rate = 0.2

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Defining the dataset path
data_path_test  = "inaturalist_12K/test"

# Create train, val and test datasets
test_dataset = inaturalist_dataset(data_path_test, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Custom_CNN(filters1=filter1,filters2=filter2,filters3 = filter3,
             filters4=filter4,filters5=filter5,kernel_size_conv=3,
             kernel_size_maxpool=2,stride=2,neurons=256,
             dropout_rate=dropout_rate,activation_func = activation_func)

# Loading the weights of the best model
model.load_state_dict(torch.load('best_model_metrics_and_path/Custom_best_model.pth', 
                                 map_location=device))
model.to(device)

# Assuming train_dataset.class_to_idx exists
idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}


# Find one sample per class from the test set
samples = {}
for img, label in test_dataset:
    if label not in samples:
        samples[label] = img
    if len(samples) == 10:
        break

# Sort by label index
samples = dict(sorted(samples.items()))

# Prepare wandb Table
columns = ["Class Name", "Predicted Label", "Image"]
table = wandb.Table(columns=columns)

model.eval()
with torch.no_grad():
    for label_idx, img in samples.items():
        img_input = img.unsqueeze(0).to(device)
        output = model(img_input)
        _, predicted_idx = torch.max(output, 1)
        
        class_name = idx_to_class[label_idx]
        pred_class = idx_to_class[predicted_idx.item()]
        
        # Convert tensor image to PIL for wandb logging
        image_pil = transforms.ToPILImage()(img.cpu())
        table.add_data(class_name, pred_class, wandb.Image(image_pil))

# Log the table to wandb
wandb.log({"Prediction Grid": table})



