import random
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

import os
import shutil
import random
from pathlib import Path

# Set your paths
root_dir = Path("inaturalist_12K")
train_dir = root_dir / "train"
val_dir = root_dir / "val"

# Create val directory if it doesn't exist
val_dir.mkdir(exist_ok=True)


# For each class in train
for class_folder in train_dir.iterdir():
    if class_folder.is_dir():
        images = list(class_folder.glob("*.jpg"))
        num_val = int(0.2 * len(images))  # 20% of images

        # Randomly choose validation images
        val_images = random.sample(images, num_val)

        # Create corresponding class folder in val
        val_class_dir = val_dir / class_folder.name
        val_class_dir.mkdir(parents=True, exist_ok=True)

        # Move selected images to validation folder
        for img_path in val_images:
            shutil.move(str(img_path), str(val_class_dir / img_path.name))

print("Validation set created successfully.")


