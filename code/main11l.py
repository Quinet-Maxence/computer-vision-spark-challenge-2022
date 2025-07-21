from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11l.pt")

# Train the model on the COCO8 example dataset for 100 epochs
model.train(data=r"/mnt/aiongpfs/users/nsharma/yolo_project/yolo_dataset/data.yaml", epochs=50, imgsz=1024 , device = 0 , plots = True)