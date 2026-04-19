"""
Debug script: visualise every cell before and after preprocessing.

Usage:
    python debug/cells.py                          # uses last live screenshot
    python debug/cells.py "data/real/foo.jpg"   # custom image

Keyboard:
    any key  — next cell
    q        — quit
"""
import sys
import cv2
import numpy as np

from src.detect_grid import load_model, detect_grid
from src.recognise_digits import load_templates, preprocess_for_matching, read_digit_opencv

WEIGHTS = "runs/detect/sudoku_grid_detector/weights/best.pt"
IMAGE   = sys.argv[1] if len(sys.argv) > 1 else "data/real/live.png"


def main():
    img = cv2.imread(IMAGE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {IMAGE}")

    model     = load_model(WEIGHTS)
    templates = load_templates()

    grid, _, conf = detect_grid(img, model)
    print(f"Grid detected (conf={conf:.2f})  {grid.shape[1]}x{grid.shape[0]}")

    h, w  = grid.shape[:2]
    ch, cw = h // 9, w // 9

    for row in range(9):
        for col in range(9):
            cell = grid[row*ch:(row+1)*ch, col*cw:(col+1)*cw]
            processed = preprocess_for_matching(cell)
            digit, score = read_digit_opencv(cell, templates)

            # print all per-digit scores so mismatches are diagnosable
            processed_img = preprocess_for_matching(cell)
            all_scores = {}
            for d, variants in templates.items():
                best = max(
                    float(cv2.matchTemplate(processed_img, tmpl, cv2.TM_CCOEFF_NORMED).max())
                    for tmpl in variants
                )
                all_scores[d] = round(best, 3)
            ranked = sorted(all_scores.items(), key=lambda x: -x[1])
            scores_str = "  ".join(f"{d}:{s:.3f}" for d, s in ranked)
            print(f"({row},{col}) -> {digit} ({score:.2f})  |  {scores_str}")

            # --- build side-by-side display ---
            display_size = 256

            # left: original cell (colour)
            left = cv2.resize(cell, (display_size, display_size), interpolation=cv2.INTER_NEAREST)

            # right: preprocessed binary (convert to BGR for stacking)
            right_gray = cv2.resize(processed, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
            right = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)

            # divider
            divider = np.full((display_size, 4, 3), (180, 180, 180), dtype=np.uint8)
            panel = np.hstack([left, divider, right])

            # label bar at the bottom
            label = f"cell ({row},{col})  ->  digit={digit}  conf={score:.2f}"
            bar = np.full((36, panel.shape[1], 3), (30, 30, 30), dtype=np.uint8)
            cv2.putText(bar, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            panel = np.vstack([panel, bar])

            cv2.imshow("cells debug  [any key = next | q = quit]", panel)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
