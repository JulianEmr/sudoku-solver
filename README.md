# Sudoku Solver

Vision-only autonomous agent that solves sudoku puzzles on [sudoku.com](https://sudoku.com). No DOM inspection — all grid and digit perception is purely image-based.

## Pipeline

```
Screenshot → YOLO grid detection → OpenCV digit recognition → Backtracking solver → Playwright automation
```

1. **Grid detection** — YOLOv8n fine-tuned to locate the sudoku board in a browser screenshot
2. **Digit recognition** — OpenCV template matching reads each of the 81 cells (Tesseract OCR available as an alternative)
3. **Solver** — backtracking algorithm computes the solution
4. **Automation** — Playwright clicks each cell and types the digit

---

## Requirements

- Python 3.13
- [`uv`](https://github.com/astral-sh/uv) package manager
- [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` (Windows) — only needed for the Tesseract method
- Docker — only needed to run the API

---

## Installation

```bash
# Install all dependencies including browser automation
uv sync --group browser --group local
```

---

## Running the project

### Option A — Fully local (no Docker)

```bash
python run_direct.py
```

Opens a browser, takes a screenshot, runs the full pipeline, and fills the grid.

---

### Option B — API + automation (Docker)

**1. Build and start the API server**

```bash
docker build -t sudoku-solver .
docker run -p 8000:8000 sudoku-solver
```

**2. Run the automation client**

```bash
python auto.py
```

Opens a browser, screenshots the page, sends it to the API, and fills the grid with the solution.

**Solve a specific image without a browser**

```bash
python auto.py --random          # picks a random image from data/real/
```

**Or call the API directly**

```bash
curl -X POST http://localhost:8000/solve -F "image=@path/to/screenshot.png"
```

---

## Training

### Via the API (recommended)

With the Docker container running:

```bash
curl -X POST http://localhost:8000/train
```

Blocks until training is complete. Weights are saved to `runs/detect/sudoku_grid_detector/weights/best.pt`.

### Locally

```bash
python scripts/train_yolo.py
```

Training data must be present in `data/train/` and `data/valid/` (443 + 43 images from Roboflow). Config: `data/data.yaml`.

---

## Benchmarking

Runs both OpenCV and Tesseract methods across 10 random images from `data/real/` and prints measured metrics (time, digits found, average confidence):

```bash
python scripts/benchmark.py
```

---

## Visual comparison

Interactive side-by-side view of OpenCV vs Tesseract results on random images:

```bash
python scripts/compare.py
# [n] next image   [q] quit
```

---

## Project structure

```
api.py                  # FastAPI server (POST /solve, POST /train)
auto.py                 # Automation client — calls the API, drives the browser
run_direct.py           # Full pipeline locally without the API server
src/
  detect_grid.py        # YOLO grid detection
  recognise_digits.py   # OpenCV template matching digit recognition
  solve_sudoku.py       # Backtracking solver
  browser.py            # Playwright browser automation
  visualize.py          # Display utilities
scripts/
  train_yolo.py         # YOLO fine-tuning script
  benchmark.py          # OpenCV vs Tesseract benchmark
  compare.py            # Interactive visual comparison
  tesseract_digits.py   # Tesseract digit recognition method
data/
  real/                 # Real sudoku.com screenshots
  templates/            # Digit templates for OpenCV matching (1.jpg–9.jpg)
  train/ valid/         # YOLO training dataset
  data.yaml             # YOLO dataset config
runs/detect/sudoku_grid_detector/weights/best.pt  # Trained model weights
```

---

## Dataset

Training data sourced from [Sudoku Vision](https://universe.roboflow.com/pete-mksb1/sudoku-vision) on Roboflow (443 train / 43 val images).
