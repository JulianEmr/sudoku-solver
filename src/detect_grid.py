# src/detect_grid.py
import cv2
import numpy as np
from ultralytics import YOLO

def load_model(weights="runs/detect/sudoku_grid_detector/weights/best.pt"):
    return YOLO(weights)

def _refine_grid_bbox(crop):
    """
    Find the tightest bounding rectangle of the actual sudoku grid
    within a (possibly oversized) YOLO crop.

    Strategy: find the largest contour that looks like a rectangle and
    covers a reasonable fraction of the crop. Falls back to the full
    crop if nothing convincing is found.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Threshold to isolate the dark grid lines against the background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop

    h, w = crop.shape[:2]
    crop_area = h * w

    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Must cover at least 40% of the crop (avoids noise)
        if area < 0.4 * crop_area:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        # Prefer 4-sided shapes (the grid border), fall back to bounding rect
        if len(approx) == 4 and area > best_area:
            best = cv2.boundingRect(approx)
            best_area = area

    if best is None:
        # Fall back: bounding rect of the single largest contour
        largest = max(contours, key=cv2.contourArea)
        best = cv2.boundingRect(largest)

    rx, ry, rw, rh = best
    return crop[ry:ry+rh, rx:rx+rw]

def detect_grid(img, model):
    results = model(img, verbose=False)

    if not results[0].boxes:
        raise RuntimeError("No grid detected in image")

    x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0].tolist())
    conf = float(results[0].boxes[0].conf[0])

    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    grid = img[y1:y2, x1:x2]
    grid = _refine_grid_bbox(grid)
    return grid, (x1, y1, x2, y2), conf