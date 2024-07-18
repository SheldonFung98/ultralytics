from ultralytics import YOLO

#  base on YOLOv8 n
# model = YOLO('yolov8n-pose')  # load a pretrained model (recommended for training)
model = YOLO('runs/pose/train3/weights/best.pt')  # load a pretrained model (recommended for training)

# Train the 256 model
results = model.train(
    data='hand-pose.yaml',
    degrees=180,
    shear=30,
    batch = 1000,
    epochs = 500,
    # imgsz=[128, 192],
    imgsz=[160, 256],
    device=0,
    plots=True,
    optimizer='SGD',
    lr0=0.001,
)