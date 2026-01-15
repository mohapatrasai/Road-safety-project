import os
import random
import shutil

# Paths to your dataset
images_dir = r"C:\Users\deepa\OneDrive\Desktop\Road_safety\images"
labels_dir = r"C:\Users\deepa\OneDrive\Desktop\Road_safety\labels"
train_images_dir = os.path.join(images_dir, "train")
val_images_dir = os.path.join(images_dir, "val")
train_labels_dir = os.path.join(labels_dir, "train")
val_labels_dir = os.path.join(labels_dir, "val")

# Create output folders
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Get list of images
images = [f for f in os.listdir(images_dir)
          if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(images_dir, f))]

print(f"Found {len(images)} images in the 'images' folder.")

if not images:
    print("⚠️ No images found. Please check the path and extensions.")
    exit()

# Shuffle and split
random.shuffle(images)
split_idx = int(0.8 * len(images))
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

# Move training files
for img in train_imgs:
    img_path = os.path.join(images_dir, img)
    label_path = os.path.join(labels_dir, os.path.splitext(img)[0] + ".txt")
    if os.path.exists(label_path):
        shutil.move(img_path, os.path.join(train_images_dir, img))
        shutil.move(label_path, os.path.join(train_labels_dir, os.path.basename(label_path)))
    else:
        print(f"⚠️ Label not found for: {img}")

# Move validation files
for img in val_imgs:
    img_path = os.path.join(images_dir, img)
    label_path = os.path.join(labels_dir, os.path.splitext(img)[0] + ".txt")
    if os.path.exists(label_path):
        shutil.move(img_path, os.path.join(val_images_dir, img))
        shutil.move(label_path, os.path.join(val_labels_dir, os.path.basename(label_path)))
    else:
        print(f"⚠️ Label not found for: {img}")

print("✅ Dataset split and moved successfully.")