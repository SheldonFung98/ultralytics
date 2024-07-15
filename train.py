from ultralytics import YOLO

#  base on YOLOv8 n
model = YOLO('yolov8n-pose')  # load a pretrained model (recommended for training)
# model = YOLO('runs/pose/train5/weights/best.pt')  # load a pretrained model (recommended for training)

# Train the 256 model
results = model.train(
    data='hand-pose.yaml',
    degrees=30,
    shear=30,
    batch = 2000,
    epochs = 500,
    imgsz=[128, 192],
    device=0,
    plots=True,
    optimizer='SGD',
    lr0=0.01,
)