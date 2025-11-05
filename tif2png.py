import os
import shutil
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ----------------------------
# SETTINGS
# ----------------------------
base_dir = "data_thermal"       # Your dataset folder
csv_path = os.path.join(base_dir, "Bounding Box Label.csv")
folders = ["positive", "negative"]
image_ext = ".png"               # output image extension

# ----------------------------
# 1️⃣ Convert .tif -> 3-channel RGB .png
# ----------------------------
print("🔄 Converting .tif images to 3-channel PNG...")
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".tif"):
            img_path = os.path.join(folder_path, fname)
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")
                out_name = fname.replace(".tif", image_ext)
                im_rgb.save(os.path.join(folder_path, out_name))
            # Optionally remove old .tif
            os.remove(img_path)
print("✅ Image conversion done.")

# ----------------------------
# 2️⃣ Read CSV & convert bboxes to YOLO
# ----------------------------
df = pd.read_csv(csv_path)

def convert_bbox_to_yolo(size, x, y, w, h):
    img_w, img_h = size
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

temp_labels_dir = os.path.join(base_dir, "all_labels")
os.makedirs(temp_labels_dir, exist_ok=True)

print("🔄 Generating YOLO labels for positive images...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_name = row['imageFilename'].replace(".tif", image_ext)
    img_path = None
    for subdir in folders:
        possible = os.path.join(base_dir, subdir, img_name)
        if os.path.exists(possible):
            img_path = possible
            break
    if img_path is None:
        print(f"⚠️ Image not found: {img_name}")
        continue
    with Image.open(img_path) as im:
        w_img, h_img = im.size
    x_c, y_c, w_n, h_n = convert_bbox_to_yolo((w_img, h_img),
                                              row['x(column)'], row['y(row)'],
                                              row['width'], row['height'])
    label_file = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(temp_labels_dir, label_file)
    with open(label_path, "a") as f:
        f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

print("✅ YOLO labels created.")

# ----------------------------
# 3️⃣ Collect all images (positive + negative)
# ----------------------------
all_images = []
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(image_ext):
            all_images.append(os.path.join(folder, fname))

# ----------------------------
# 4️⃣ Split into train / val / test
# ----------------------------
train_imgs, temp_imgs = train_test_split(all_images, test_size=0.3, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)

# ----------------------------
# 5️⃣ Create YOLO folder structure
# ----------------------------
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(base_dir, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, f"labels/{split}"), exist_ok=True)

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

# ----------------------------
# 6️⃣ Save classes.txt
# ----------------------------
with open(os.path.join(base_dir, "classes.txt"), "w") as f:
    f.write("object\n")

# ----------------------------
# 7️⃣ Create data_thermal.yaml
# ----------------------------
yaml_content = f"""
train: {base_dir}/images/train
val: {base_dir}/images/val
test: {base_dir}/images/test

nc: 1
names: ['object']
"""
with open(os.path.join(base_dir, "data_thermal.yaml"), "w") as f:
    f.write(yaml_content.strip())

print("✅ YAML file created at:", os.path.join(base_dir, "data_thermal.yaml"))
print("🎉 Dataset is ready for YOLOv5 training!")
