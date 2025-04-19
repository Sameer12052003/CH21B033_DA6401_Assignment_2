import os
import shutil
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Define paths
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "inaturalist_12K"))
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "val")

# Create val directory if it doesn't exist
os.makedirs(val_dir, exist_ok=True)

# For each class folder in train
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
        num_val = int(0.2 * len(images))  # 20% split
        val_images = np.random.choice(images, size=num_val, replace=False)

        # Create corresponding class folder in val
        val_class_path = os.path.join(val_dir, class_name)
        os.makedirs(val_class_path, exist_ok=True)

        # Move selected images to val directory
        for img in val_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(val_class_path, img)
            shutil.move(src_path, dst_path)

print("Validation split completed")
