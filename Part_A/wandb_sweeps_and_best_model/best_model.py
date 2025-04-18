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
import sys

# Add Part_A to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.custom_dataset import inaturalist_dataset
from custom_model.model import Custom_CNN
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

# Get the absolute path relative to this script
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path_train = os.path.join(base_dir, "inaturalist_12K", "train")
data_path_val = os.path.join(base_dir, "inaturalist_12K", "val")
data_path_test = os.path.join(base_dir, "inaturalist_12K", "test")

# Create train, val and test datasets
train_dataset = inaturalist_dataset(data_path_train, transform=transform)
val_dataset = inaturalist_dataset(data_path_val, transform=transform)
test_dataset = inaturalist_dataset(data_path_test, transform=transform)

print('Path found')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Custom_CNN(filters1=filter1,filters2=filter2,filters3 = filter3,
             filters4=filter4,filters5=filter5,kernel_size_conv=3,
             kernel_size_maxpool=2,stride=2,neurons=256,
             dropout_rate=dropout_rate,activation_func = activation_func).to(device)

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
save_path_model  = os.path.join(base_dir, 'best_model_metrics_and_path','Custom_best_model.pth')
torch.save(model.state_dict(), save_path_model)


# Testing the model
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

print(f'Test accuracy obtained using Custom Model : {test_acc:.2f}')


df = pd.DataFrame([{'Train_accuracy' : train_acc,
                    'Validatoin accuracy': val_acc,
                   'Test_accuracy': test_acc}])

save_path  = os.path.join(base_dir, 'best_model_metrics_and_path','Metrics_Custom_Model.csv')
df.to_csv(save_path,index=False)