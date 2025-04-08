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
    
# Defining the Custom CNN model
class Custom_CNN(nn.Module):
    
    def __init__(self,input_dim=224,filters1=16,filters2=32,filters3 = 64,
                 filters4=128,filters5=256,kernel_size_conv=3,
                 kernel_size_maxpool=2,stride=2,neurons=256,dropout_rate=0.2,activation_func = 'relu'):
        
        super().__init__()
        print('Model Invoked')
        
        self.activation = activation_func
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=filters1, kernel_size=kernel_size_conv)
        self.bn1 = nn.BatchNorm2d(filters1)
        self.drop1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(in_channels=filters1, out_channels=filters2, kernel_size=kernel_size_conv)
        self.bn2 = nn.BatchNorm2d(filters2)
        self.drop2 = nn.Dropout2d(dropout_rate)
        
        self.conv3 = nn.Conv2d(in_channels=filters2,out_channels=filters3,kernel_size=kernel_size_conv)
        self.bn3 = nn.BatchNorm2d(filters3)
        self.drop3 = nn.Dropout2d(dropout_rate)
        
        self.conv4 = nn.Conv2d(in_channels=filters3,out_channels=filters4,kernel_size=kernel_size_conv)
        self.bn4 = nn.BatchNorm2d(filters4)
        self.drop4 = nn.Dropout2d(dropout_rate)

        self.conv5 = nn.Conv2d(in_channels=filters4,out_channels=filters5,kernel_size=kernel_size_conv)
        self.bn5 = nn.BatchNorm2d(filters5)
        self.drop5 = nn.Dropout2d(dropout_rate)

        self.pool = nn.MaxPool2d(kernel_size=kernel_size_maxpool, stride=stride)

        # Automatically compute output dimension after conv and pooling
        dummy_input = torch.zeros(1, 3, input_dim, input_dim)
        x = self.pool(F.relu(self.drop1(self.bn1(self.conv1(dummy_input)))))
        x = self.pool(F.relu(self.drop2(self.bn2(self.conv2(x)))))
        x = self.pool(F.relu(self.drop3(self.bn3(self.conv3(x)))))
        x = self.pool(F.relu(self.drop4(self.bn4(self.conv4(x)))))
        x = self.pool(F.relu(self.drop5(self.bn5(self.conv5(x)))))

        self.flatten_dim = x.view(1, -1).size(1)  # total features going into fc
        
        self.dropout_fc = nn.Dropout2d(dropout_rate)
        self.fc = nn.Linear(self.flatten_dim, neurons)
        self.out = nn.Linear(neurons,10)
        

    def forward(self, x):
        
        if self.activation == 'relu':
                
            x = self.pool(F.relu(self.drop1(self.bn1(self.conv1(x)))))
            x = self.pool(F.relu(self.drop2(self.bn2(self.conv2(x)))))
            x = self.pool(F.relu(self.drop3(self.bn3(self.conv3(x)))))
            x = self.pool(F.relu(self.drop4(self.bn4(self.conv4(x)))))
            x = self.pool(F.relu(self.drop5(self.bn5(self.conv5(x)))))

            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = self.dropout_fc(F.relu(self.fc(x)))
            x = self.out(x)
            return x
            
            
        elif self.activation == 'sigmoid':
            
            x = self.pool(F.sigmoid(self.drop1(self.bn1(self.conv1(x)))))
            x = self.pool(F.sigmoid(self.drop2(self.bn2(self.conv2(x)))))
            x = self.pool(F.sigmoid(self.drop3(self.bn3(self.conv3(x)))))
            x = self.pool(F.sigmoid(self.drop4(self.bn4(self.conv4(x)))))
            x = self.pool(F.sigmoid(self.drop5(self.bn5(self.conv5(x)))))

            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = self.dropout_fc(F.sigmoid(self.fc(x)))
            x = self.out(x)
            return x
        
        elif self.activation == 'tanh':
            
            x = self.pool(F.tanh(self.drop1(self.bn1(self.conv1(x)))))
            x = self.pool(F.tanh(self.drop2(self.bn2(self.conv2(x)))))
            x = self.pool(F.tanh(self.drop3(self.bn3(self.conv3(x)))))
            x = self.pool(F.tanh(self.drop4(self.bn4(self.conv4(x)))))
            x = self.pool(F.tanh(self.drop5(self.bn5(self.conv5(x)))))

            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = self.dropout_fc(F.tanh(self.fc(x)))
            x = self.out(x)
            return x
        
        
            

