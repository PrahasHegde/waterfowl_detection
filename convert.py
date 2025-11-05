import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

# --- Paths ---
base_dir = "data_thermal"
csv_path = os.path.join(base_dir, "Bounding Box Label.csv")

# --- Read CSV ---
df = pd.read_csv(csv_path)

# --- YOLO Conversion Function ---
def convert_bbox_to_yolo(size, x, y, w, h):
    img_w, img_h = size
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

# --- Temporary labels folder ---
temp_labels_dir = os.path.join(base_dir, "all_labels")
os.makedirs(temp_labels_dir, exist_ok=True)

# --- Step 1: Generate YOLO labels for images listed in CSV ---
label_count = 0
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting bounding boxes"):
    img_name = row['imageFilename']

    # Try to locate the image
    img_path = None
    for subdir in ["Positive", "Negative"]:
        possible_path = os.path.join(base_dir, subdir, img_name)
        if os.path.exists(possible_path):
            img_path = possible_path
            break

    if img_path is None:
        print(f"⚠️ Image not found: {img_name}")
        continue

    # Read image size
    with Image.open(img_path) as im:
        img_w, img_h = im.size

    # Convert bbox to YOLO format
    x, y, w, h = row['x(column)'], row['y(row)'], row['width'], row['height']
    x_center, y_center, w_norm, h_norm = convert_bbox_to_yolo((img_w, img_h), x, y, w, h)

    # Write label file
    label_path = os.path.join(temp_labels_dir, os.path.splitext(img_name)[0] + ".txt")
    with open(label_path, "a") as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    label_count += 1

print(f"\n✅ Created {label_count} bounding boxes in YOLO format.")

# --- Step 2: Collect all images (positive + negative) ---
all_images = []
for subdir in ["Positive", "Negative"]:
    folder = os.path.join(base_dir, subdir)
    for img_file in os.listdir(folder):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
            all_images.append(os.path.join(subdir, img_file))

print(f"📸 Total images found: {len(all_images)}")

# --- Step 3: Split into train/val/test ---
train_imgs, temp_imgs = train_test_split(all_images, test_size=0.3, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)

print(f"📂 Split: Train={len(train_imgs)} | Val={len(val_imgs)} | Test={len(test_imgs)}")

# --- Step 4: Create YOLO folder structure ---
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(base_dir, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, f"labels/{split}"), exist_ok=True)

# --- Step 5: Copy images & labels ---
def copy_files(img_list, split):
    for rel_path in tqdm(img_list, desc=f"Copying {split} set"):
        src_img = os.path.join(base_dir, rel_path)
        dst_img = os.path.join(base_dir, f"images/{split}", os.path.basename(src_img))
        shutil.copy(src_img, dst_img)

        label_name = os.path.splitext(os.path.basename(src_img))[0] + ".txt"
        src_label = os.path.join(temp_labels_dir, label_name)
        dst_label = os.path.join(base_dir, f"labels/{split}", label_name)

        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            open(dst_label, "w").close()  # empty label for negatives

for split, imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
    copy_files(imgs, split)

# --- Step 6: Save class names ---
with open(os.path.join(base_dir, "classes.txt"), "w") as f:
    f.write("object\n")

print("\n✅ YOLO dataset ready for training!")
print("📁 Folder structure:")
print(f"{base_dir}/")
print(" ├── images/train, val, test/")
print(" └── labels/train, val, test/")
