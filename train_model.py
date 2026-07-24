from ultralytics import YOLO

if __name__ == '__main__':
    # Load base YOLOv8 Small segmentation model
    model = YOLO("yolov8s-seg.pt")

    # Train on Apple Silicon (M3 Pro)
    model.train(
        data="datasets/cracks-spalling-v1/data.yaml",
        epochs=50,
        imgsz=640,
        device="mps",
        project="runs/segment",
        name="cracks-spalling-v1"
    )