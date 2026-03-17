from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="sudoku_grid_detector",
    patience=5,        # stops early if no improvement
    exist_ok=True
)