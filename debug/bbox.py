# debug/bbox.py
import cv2
import os
import random
from src.detect_grid import load_model, detect_grid

IMG_DIR = "data/real"
model   = load_model()
files   = [f for f in os.listdir(IMG_DIR) if not f.startswith('.')]

for img_file in random.sample(files, 1):
    img = cv2.imread(f"{IMG_DIR}/{img_file}")
    grid, (x1, y1, x2, y2), conf = detect_grid(img, model)

    # draw bbox on original image
    debug = img.copy()
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0,255,0), 2)

    # draw cell divisions on grid
    h, w = grid.shape[:2]
    ch, cw = h // 9, w // 9
    grid_debug = grid.copy()
    for i in range(10):
        cv2.line(grid_debug, (i*cw, 0),   (i*cw, h),   (0,0,255), 1)
        cv2.line(grid_debug, (0,   i*ch), (w,    i*ch), (0,0,255), 1)

    print(f"{img_file} | bbox: x1={x1} y1={y1} x2={x2} y2={y2} | grid: {w}x{h} | cell: {cw}x{ch}")

    scale = 500 / img.shape[0]
    debug = cv2.resize(debug, (int(img.shape[1]*scale), 500))
    grid_debug = cv2.resize(grid_debug, (500, 500))
    cv2.imshow("Full image — green=bbox", debug)
    cv2.imshow("Grid crop — red=cell divisions", grid_debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()