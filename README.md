# YOLOv5 Waterfowl Detection (UAV Thermal & RGB)

This project uses YOLOv5 to detect waterfowl (ducks, geese, etc.) in aerial imagery. The model is designed to work with both standard **RGB (visual)** and **thermal** images captured by UAVs (drones).

![YOLOv5 Demo](httpss://user-images.githubusercontent.com/31566456/230910103-6113b671-b1e6-42d4-b7c1-7427c32087e5.gif)
*(This is a placeholder GIF. You can replace it with a screenshot of your own detection results!)*

---

### 🦢 Detected Classes

This model is trained to identify the following classes:
* `waterfowl`
* *(Update this list based on your dataset's `.yaml` file!)*

---

## 🔧 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PrahasHegde/waterfowl_detection.git](https://github.com/PrahasHegde/waterfowl_detection.git)
    cd waterfowl_detection
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install requirements:**
    This project uses the official YOLOv5 repository. Install its requirements:
    ```bash
    pip install -r requirements.txt
    ```
    *(If you don't have a `requirements.txt` file, you can get it from the original [YOLOv5 repo](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) or just run `pip install yolov5`.)*

---

## 💾 Dataset

**IMPORTANT: The dataset is NOT included in this repository.**

Due to its large size (including `.zip` files over 100MB), the entire dataset is ignored by the `.gitignore` file.

The project expects a complex data structure, which includes the following (ignored) folders:
* `data/`
* `datasets/`
* `data_thermal/`
* `data_waterfowl/`
* `00_UAV-derived Thermal Waterfowl Dataset/`
* `01_RGB Images/`
* `02_Test Orhomosaic/` (This folder contains large `.zip` files)



To train this model, you must provide your own dataset and configure the corresponding `.yaml` file to point to your local `train/` and `val/` image folders.

---

## 🚀 How to Use

All commands are run from the root of the project folder.

### 1. Training
To train the model, you will need a custom dataset `.yaml` file (e.g., `waterfowl.yaml`) that defines your dataset paths and classes.

```bash
# Train a YOLOv5s model for 100 epochs
python train.py --img 640 --batch 16 --epochs 100 --data waterfowl.yaml --weights yolov5s.pt
