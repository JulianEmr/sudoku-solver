# scripts/train_yolo.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="sudoku_grid_detector",
    project=".",
    exist_ok=True,
    patience=10,
)
print("Training complete. Weights saved to runs/detect/sudoku_grid_detector/weights/best.pt")