import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
from torchvision import models
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F
from data_preprocessing.custom_dataset import inaturalist_dataset

np.random.seed(42)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Defining the dataset path
data_path_train  = "inaturalist_12K/train"
data_path_val = "inaturalist_12K/val"
data_path_test = "inaturalist_12K/test"

# Create train, val and test datasets
train_dataset = inaturalist_dataset(data_path_train, transform=transform)
val_dataset = inaturalist_dataset(data_path_val, transform=transform)
test_dataset = inaturalist_dataset(data_path_test, transform=transform)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained GoogLeNet
model = models.googlenet(pretrained=True)

# Replacing the final fully connected layer suitable for 10 classes.
model.fc = nn.Linear(in_features=1024, out_features=10, bias=True)

# Freezing all the layers of the model
for param in model.parameters():
    param.requires_grad = False

# Unfreezing the final fc layer for finetuning of the model
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

# Training loop
for epoch in range(epochs): 

    running_loss = 0
    total = 0
    correct = 0
    
    for images, labels in train_loader:
        
        model.train()
        
        # get the inputs; data is a list of [inputs, labels]
        images, labels = images.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()*images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_loss = running_loss/total
    train_acc = 100*correct/total
    
    with torch.no_grad():
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        for images, labels in val_loader:
            
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
       
    print(f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
            f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

print('Finished Training')
torch.save(model.state_dict(), 'best_model_metrics_and_path/GoogLeNet_best_model.pth')


with torch.no_grad():
        
    # Testing
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    for images, labels in test_loader:
        
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

    test_loss /= test_total
    test_acc = 100. * test_correct / test_total

print(f'Test accuracy obtained using Pre-trained GoogLeNet : {test_acc:.2f}')


df = pd.DataFrame([{'Train_accuracy' : train_acc,
                    'Validation accuracy': val_acc,
                   'Test_accuracy': test_acc}])

df.to_csv('best_model_metrics_and_path/Metrics_GoogLeNet.csv',index=False)