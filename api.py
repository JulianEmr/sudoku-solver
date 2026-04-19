import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from src.detect_grid import load_model, detect_grid
from src.recognise_digits import load_templates, read_grid
from src.solve_sudoku import solve_sudoku

WEIGHTS   = "runs/detect/sudoku_grid_detector/weights/best.pt"
DATA_YAML = "data/data.yaml"

app = FastAPI(title="Sudoku Solver API")

model     = load_model(WEIGHTS)
templates = load_templates()


@app.post("/train")
def train():
    yolo = YOLO("yolov8n.pt")
    yolo.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        name="sudoku_grid_detector",
        project=".",
        exist_ok=True,
        patience=10,
    )
    return {"status": "done", "weights": WEIGHTS}


@app.post("/solve")
async def solve(image: UploadFile = File(...)):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    try:
        grid_img, (x1, y1, x2, y2), conf = detect_grid(img, model)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Grid detection failed: {e}")

    board, confs = read_grid(grid_img, templates)

    try:
        solution = solve_sudoku([row[:] for row in board])
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "confidence": round(float(conf), 4),
        "grid": board,
        "solution": solution,
    }
