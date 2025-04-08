import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
import torch.optim as optim
import torch.nn as nn

from Q1 import Custom_CNN

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

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Defining the dataset path
data_path_train  = "inaturalist_12K/train"
data_path_val = "inaturalist_12K/val"

# Create train and val datasets
train_dataset = inaturalist_dataset(data_path_train, transform=transform)
val_dataset = inaturalist_dataset(data_path_val, transform=transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Custom_CNN(filters1=32,filters2=32,filters3=32,filters4=32,filters5=32).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10


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

        
        
    print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

print('Finished Training')



    
    

