from ultralytics import YOLO
import os

model = YOLO("runs/detect/sudoku_grid_detector/weights/best.pt")

# test on a new screenshot

test_screenshot = os.listdir("data/val/images")[0]  # just pick the first one for testing
results = model(f"data/val/images/{test_screenshot}")
results[0].show()  # displays image with bounding box

# get the box coordinates
box = results[0].boxes[0].xyxy[0].tolist()  # [x1, y1, x2, y2]
print(f"Grid found at: {box}")
