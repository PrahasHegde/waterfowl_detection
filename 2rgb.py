import os
from PIL import Image

folders = ["train", "val", "test"]
base_dir = "data_thermal/images"

for split in folders:
    path = os.path.join(base_dir, split)
    for fname in os.listdir(path):
        if fname.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
            img_path = os.path.join(path, fname)
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")  # force 3 channels
                im_rgb.save(img_path)
print("✅ All images converted to 3-channel RGB")
