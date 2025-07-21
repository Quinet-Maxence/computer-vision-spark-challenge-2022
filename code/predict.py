import os
import pandas as pd
from ultralytics import YOLO

# Load YOLO model
model = YOLO("/mnt/aiongpfs/users/nsharma/yolo_project/runs/detect/train11/weights/best.pt")

# Run the prediction
model.predict(
    source="/mnt/aiongpfs/users/nsharma/yolo_project/yolo_dataset/test/images",  # Path to the test images folder
    save=True,               # Save results in a folder
    save_txt=True,           #  Saves detections in text files
    imgsz=1024,               # Images size
    conf=0.5,               # Minimum confidence to display a detection
    max_det = 1,
    show_labels = True,
    show_boxes = True
)
