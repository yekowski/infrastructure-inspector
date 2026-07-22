import os
from ultralytics import YOLO

# 1. Point to your local dataset configuration
dataset_yaml = os.path.abspath("dataset/data.yaml")

if not os.path.exists(dataset_yaml):
    print(f"ERROR: Could not find '{dataset_yaml}'. Make sure your downloaded 'dataset' folder is in the project root!")
    exit(1)

print(f"Loading dataset from: {dataset_yaml}")
print("Starting YOLOv8 training...\n")

# 2. Load the base YOLOv8 Nano Segmentation model
model = YOLO('yolov8n-seg.pt')

# 3. Start local training loop
results = model.train(
    data=dataset_yaml,
    epochs=50,
    imgsz=640,
    device='cpu',  # Uses your CPU (will use GPU automatically if CUDA is configured)
    patience=10
)

print("\n" + "="*50)
print("SUCCESS: Training complete!")
print("Next step: Copy 'runs/segment/train/weights/best.pt' into 'models/crack_seg_best.pt'")
print("="*50)