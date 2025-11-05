from ultralytics import YOLO

# 1. Load a pre-trained model
# We use 'yolov8n.pt' for a small, fast model.
# You can also use yolov8s, yolov8m, etc., for better accuracy at the cost of speed.
model = YOLO('yolov8n.pt') 

# 2. Train the model
if __name__ == '__main__':
    results = model.train(
        data='yolov5\data\waterfowl.yaml',  # Path to your dataset config file
        epochs=50,            # Number of training cycles (start with 50-100)
        imgsz=640,            # Image size for training (640 is a good default)
        batch=8,              # Number of images to process at once (adjust based on your GPU memory)
        name='yolov8n_waterfowl_thermal' # Name for the training run folder
    )
