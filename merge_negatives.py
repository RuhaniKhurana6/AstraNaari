import os
import shutil
import random

# SOURCE: Your downloaded negative backgrounds
neg_dir = r"C:\Users\Asus\OneDrive\Desktop\Hackathon\negative_samples_backgrounds"

# DESTINATION: Dataset #2
dataset2_dir = r"C:\Users\Asus\OneDrive\Desktop\Hackathon\dataset#2"

# Ensure dataset2 dirs exist
for split in ["train", "valid"]:
    os.makedirs(os.path.join(dataset2_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset2_dir, split, "labels"), exist_ok=True)

negatives = []
for root, _, files in os.walk(neg_dir):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(root, file)
            label_path = img_path.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")
            
            if os.path.exists(label_path):
                negatives.append((img_path, label_path))

print(f"Found {len(negatives)} negative samples.")

# Shuffle to prevent bias
random.shuffle(negatives)

# 80% Train, 20% Valid split (YOLO standard)
split_idx = int(0.8 * len(negatives))
train_negs = negatives[:split_idx]
val_negs = negatives[split_idx:]

def copy_to_dataset(data, split):
    for i, (img, label) in enumerate(data):
        # Create unique names to avoid overwriting your existing dataset#2 images
        base_name = os.path.basename(img)
        unique_prefix = f"neg_bg_{split}_{i}_"
        new_img_name = unique_prefix + base_name
        new_lbl_name = new_img_name.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")
        
        shutil.copy(img, os.path.join(dataset2_dir, split, "images", new_img_name))
        shutil.copy(label, os.path.join(dataset2_dir, split, "labels", new_lbl_name))

# Copy them over
copy_to_dataset(train_negs, "train")
copy_to_dataset(val_negs, "valid")

print(f"✅ Successfully merged:")
print(f"   -> {len(train_negs)} negative images added to {dataset2_dir}/train")
print(f"   -> {len(val_negs)} negative images added to {dataset2_dir}/valid")
