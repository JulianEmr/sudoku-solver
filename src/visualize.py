# src/visualize.py
import cv2


def print_board(board):
    for i, row in enumerate(board):
        if i % 3 == 0 and i != 0:
            print("------+-------+------")
        row_str = ""
        for j, val in enumerate(row):
            if j % 3 == 0 and j != 0:
                row_str += "| "
            row_str += (str(val) if val != 0 else ".") + " "
        print(row_str)


def build_vis(grid, board, confs):
    """
    Draw the recognised digits over the grid image.
    Green  = digit found (high conf)
    Orange = digit found (low conf, < 0.6)
    Red    = empty cell (nothing recognised)
    """
    vis = grid.copy()
    h, w = vis.shape[:2]
    ch, cw = h // 9, w // 9

    for row in range(9):
        for col in range(9):
            digit = board[row][col]
            conf  = confs[row][col]
            cx = col * cw + cw // 2
            cy = row * ch + ch // 2

            if digit == 0:
                color = (0, 0, 220)       # red — nothing found
                label = "?"
            elif conf < 0.6:
                color = (0, 140, 255)     # orange — low confidence
                label = str(digit)
            else:
                color = (0, 200, 0)       # green — confident
                label = str(digit)

            font_scale = cw / 60
            thickness  = max(1, cw // 30)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.putText(vis, label,
                        (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA)

    # Draw cell grid lines so misalignment is obvious
    for i in range(10):
        lw = 2 if i % 3 == 0 else 1
        cv2.line(vis, (i * cw, 0),      (i * cw, h),      (180, 180, 180), lw)
        cv2.line(vis, (0,      i * ch), (w,      i * ch), (180, 180, 180), lw)

    return vis


def show_vis(grid, board, confs, output_path="result.jpg"):
    """Build, save, and display the visualisation. Blocks until a key is pressed."""
    vis = build_vis(grid, board, confs)
    cv2.imwrite(output_path, vis)
    print(f"Saved {output_path}")
    cv2.imshow("Sudoku — recognised digits (any key to close)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
