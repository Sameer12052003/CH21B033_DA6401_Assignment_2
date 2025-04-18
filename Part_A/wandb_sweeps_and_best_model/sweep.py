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
import wandb

# Add Part_A to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.custom_dataset import inaturalist_dataset
from custom_model.model import Custom_CNN
np.random.seed(42)

sweep_config = {
    
    'method': 'bayes',
    
    'name': 'bayes-cnn-hyperparam-search1',
    'metric': {
        'name': 'Val_Acc',  # or 'val_loss', depending on what you're optimizing
        'goal': 'maximize'
    },

    'parameters': {
        'filter_config': {
            'values': 
                [
                [32, 32, 32, 32, 32],
                [16, 32, 64, 128, 256],
                [128, 64, 32, 16, 8],
                [16, 32, 48, 64, 96],
                [8, 16, 32, 64, 64],
                ]

        },
        'activation_func': {
            'values': ['relu', 'sigmoid', 'tanh']
        },
        
        'dropout_rate': {
            'values': [0.2, 0.3]
            
        },
        
        'batch_size': {
            'values': [32, 64]
        },
        
        'epochs' : {
            'values': [5]
        },
    }
}

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Get the absolute path relative to this script
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

data_path_train = os.path.join(base_dir, "inaturalist_12K", "train")
data_path_val = os.path.join(base_dir, "inaturalist_12K", "val")

# Create train and val datasets
train_dataset = inaturalist_dataset(data_path_train, transform=transform)
val_dataset = inaturalist_dataset(data_path_val, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    
    wandb.init(project="DA6401_Assignment_2",entity="ch21b033-iit-madras",
               name= "Hyperparam_Search_1"
)
    
    config = wandb.config
    
    
    # Re-assign just to ensure it reflects in plots/legends
    wandb.run.name = f"""DL_sweep_1_filters_{config.filter_config}_act_{config.activation_func}
            _drop_{config.dropout_rate}_bs_{config.batch_size}_ep_{config.epochs}"""
            
    wandb.run.save()  # ensures name update is logged
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    filter1,filter2, filter3, filter4, filter5 = config.filter_config
        
    model = Custom_CNN(filters1=filter1,filters2=filter2,filters3 = filter3,
                 filters4=filter4,filters5=filter5,kernel_size_conv=3,
                 kernel_size_maxpool=2,stride=2,neurons=256,
                 dropout_rate=config.dropout_rate,activation_func = config.activation_func).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = config.epochs

    for epoch in range(epochs):  # loop over the dataset multiple times

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

        
        # Logging to wandb 
        wandb.log({
            'epoch':epoch,
            "Train_loss" : train_loss,
            "Val_loss" : val_loss,
            'Train_accuracy': train_acc,
            "Val_Acc" : val_acc
        }) 
        
           
        print(f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
                f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    print('Finished Training')

    wandb.finish()
    
    
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment_2",entity="ch21b033-iit-madras")
    wandb.agent(sweep_id, function=main, count=30)