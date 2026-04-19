# main.py
import cv2, time
from src.detect_grid import load_model, detect_grid
from src.recognise_digits import load_templates, read_grid
from src.solve_sudoku import solve_sudoku
from src.visualize import print_board, show_vis
from src.browser import complete_sudoku, open_sudoku, screenshot_page, close_browser

# ── Config ─────────────────────────────────────────────
WEIGHTS    = "runs/detect/sudoku_grid_detector/weights/best.pt"
SCREENSHOT = "data/real/live.png"
# ───────────────────────────────────────────────────────

def main():
    print("Loading YOLO model...")
    model = load_model(WEIGHTS)
    templates = load_templates()

    print("Opening sudoku.com...")
    pw, browser, page = open_sudoku()
    try:
        img_path = screenshot_page(page, SCREENSHOT)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Screenshot failed: {img_path}")

        print("Detecting grid...")
        grid, (x1, y1, x2, y2), conf = detect_grid(img, model)
        print(f"Grid at [{x1},{y1},{x2},{y2}] (conf: {conf:.2f})  ->  crop size: {grid.shape[1]}x{grid.shape[0]}")

        print("Reading digits...")
        start = time.time()
        board, confs = read_grid(grid, templates)
        elapsed = (time.time() - start) * 1000

        filled = sum(d != 0 for row in board for d in row)
        print(f"\n{filled}/81 digits in {elapsed:.0f}ms:\n")
        print_board(board)

        print("Solving...")
        solved = solve_sudoku(board)
        print_board(solved)

        complete_sudoku(page, solved)
        show_vis(grid, board, confs)

        return board, (x1, y1, x2, y2)
    finally:
        close_browser(pw, browser)

if __name__ == "__main__":
    main()
