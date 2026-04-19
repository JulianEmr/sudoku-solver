# scripts/compare.py
import cv2
import numpy as np
import random
import os
import time
from src.detect_grid import load_model, detect_grid
from src.recognise_digits import load_templates, read_grid, CONFIDENCE_THRESHOLD
from scripts.tesseract_digits import read_grid_tesseract

IMG_DIR   = "data/real"
CELL_SIZE = 60
BOARD_W   = 9 * CELL_SIZE
BOARD_H   = 9 * CELL_SIZE
HEADER_H  = 80   # space above the grid for per-method stats
FOOTER_H  = 50   # space below for shared comparison stats
PANEL_H   = HEADER_H + BOARD_H + FOOTER_H

# colours (BGR)
C_EMPTY      = (200, 200, 200)   # grey  — cell left empty
C_LOW_CONF   = (0,   140, 255)   # orange — digit found, conf < threshold + 0.2
C_HIGH_CONF  = (30,  30,  30)    # near-black — confident digit
C_DISAGREE   = (0,   0,   210)   # red background — methods disagree on this cell
C_BG         = (245, 245, 245)
C_GRID_THIN  = (200, 200, 200)
C_GRID_THICK = (80,  80,  80)


def _draw_board(board, confs, title, elapsed_ms, disagreements):
    """
    Render one method's board with per-cell confidence colouring
    and disagreement highlights.
    """
    img = np.ones((PANEL_H, BOARD_W, 3), dtype=np.uint8) * 245

    # ── header ──────────────────────────────────────────────────────────
    filled    = sum(d != 0 for row in board for d in row)
    conf_vals = [confs[r][c] for r in range(9) for c in range(9)
                 if board[r][c] != 0]
    avg_conf  = (sum(conf_vals) / len(conf_vals)) if conf_vals else 0.0

    cv2.putText(img, title,
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(img, f"time: {elapsed_ms:.0f} ms",
                (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(img, f"digits: {filled}/81   avg conf: {avg_conf:.2f}",
                (6, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

    # ── cells ────────────────────────────────────────────────────────────
    for row in range(9):
        for col in range(9):
            x  = col * CELL_SIZE
            y  = HEADER_H + row * CELL_SIZE
            digit = board[row][col]
            conf  = confs[row][col]

            # red tint if this cell disagrees with the other method
            if (row, col) in disagreements:
                cv2.rectangle(img, (x, y), (x + CELL_SIZE, y + CELL_SIZE),
                              (200, 220, 255), -1)   # light-red fill

            if digit != 0:
                color = C_HIGH_CONF if conf >= CONFIDENCE_THRESHOLD + 0.2 else C_LOW_CONF
                label = str(digit)
                fs    = CELL_SIZE / 70
                thick = max(1, CELL_SIZE // 30)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
                cv2.putText(img, label,
                            (x + (CELL_SIZE - tw) // 2, y + (CELL_SIZE + th) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, color, thick, cv2.LINE_AA)

                # small confidence label in corner
                conf_label = f"{conf:.2f}"
                cv2.putText(img, conf_label,
                            (x + 2, y + CELL_SIZE - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1, cv2.LINE_AA)
            else:
                # empty cell marker
                cx = x + CELL_SIZE // 2
                cy = y + CELL_SIZE // 2
                cv2.drawMarker(img, (cx, cy), C_EMPTY, cv2.MARKER_CROSS, 10, 1)

    # ── grid lines ───────────────────────────────────────────────────────
    for i in range(10):
        lw = 2 if i % 3 == 0 else 1
        lc = C_GRID_THICK if i % 3 == 0 else C_GRID_THIN
        x0 = i * CELL_SIZE
        y0 = HEADER_H + i * CELL_SIZE
        cv2.line(img, (x0, HEADER_H), (x0, HEADER_H + BOARD_H), lc, lw)
        cv2.line(img, (0, y0),        (BOARD_W, y0),             lc, lw)

    # ── footer: legend ───────────────────────────────────────────────────
    fy = HEADER_H + BOARD_H + 18
    cv2.putText(img, "digit colour:  dark=high conf   orange=low conf   x=empty   red bg=disagrees",
                (4, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1, cv2.LINE_AA)

    return img


def _make_actual_panel(img):
    """Resize the original screenshot to match PANEL_H."""
    scale  = PANEL_H / img.shape[0]
    resized = cv2.resize(img, (int(img.shape[1] * scale), PANEL_H))
    cv2.putText(resized, "Original", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return resized


def _disagreement_cells(board_a, board_b):
    """Return set of (row, col) where the two boards differ (ignoring double-empty)."""
    diff = set()
    for r in range(9):
        for c in range(9):
            a, b = board_a[r][c], board_b[r][c]
            if a != b:
                diff.add((r, c))
    return diff


def main():
    model     = load_model()
    templates = load_templates()
    files     = [f for f in os.listdir(IMG_DIR) if not f.startswith('.')]

    while True:
        img_file = random.choice(files)
        img      = cv2.imread(f"{IMG_DIR}/{img_file}")

        try:
            grid, _, grid_conf = detect_grid(img, model)
        except RuntimeError:
            continue

        t0 = time.perf_counter()
        board_cv,   confs_cv   = read_grid(grid, templates)
        t_cv = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        board_tess, confs_tess = read_grid_tesseract(grid)
        t_tess = (time.perf_counter() - t0) * 1000

        disagreements = _disagreement_cells(board_cv, board_tess)

        panel_cv   = _draw_board(board_cv,   confs_cv,   "OpenCV",    t_cv,   disagreements)
        panel_tess = _draw_board(board_tess, confs_tess, "Tesseract", t_tess, disagreements)
        actual     = _make_actual_panel(img)

        # pad actual width to match board panels
        bw = panel_cv.shape[1]
        if actual.shape[1] < bw:
            pad    = np.ones((PANEL_H, bw - actual.shape[1], 3), dtype=np.uint8) * 245
            actual = cv2.hconcat([actual, pad])
        else:
            actual = actual[:, :bw]

        display = cv2.hconcat([actual, panel_cv, panel_tess])

        # ── summary overlay on the actual panel (top-left) ───────────────
        agree  = 81 - len(disagreements)
        n_both = sum(board_cv[r][c] != 0 and board_tess[r][c] != 0
                     for r in range(9) for c in range(9))
        summary = (f"grid conf: {grid_conf:.2f}   "
                   f"agreement: {agree}/81   "
                   f"both filled: {n_both}/81   "
                   f"cv {t_cv:.0f}ms vs tess {t_tess:.0f}ms  ({t_tess/t_cv:.1f}x slower)")
        cv2.putText(display, summary,
                    (6, PANEL_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (30, 30, 30), 1, cv2.LINE_AA)

        title = f"{img_file}  |  [n]=next  [q]=quit"
        cv2.imshow(title, display)

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == ord('q'):
            break

        cv2.imwrite(f"comparison.png", display)


if __name__ == "__main__":
    main()
