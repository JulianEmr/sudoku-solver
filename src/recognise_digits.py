# src/recognise_digits.py
import cv2
import numpy as np
import os

TEMPLATES_DIR = "data/templates"
CONFIDENCE_THRESHOLD = 0.6

TARGET_SIZE = 128
# Minimum fraction of the cell area a contour must cover to be considered a digit.
# Filters out grid-line specks while keeping thin digits like "1".
_MIN_DIGIT_AREA_RATIO = 0.01


def _binarize(gray):
    """Otsu threshold + ensure black digit on white background."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    return binary


def _crop_to_digit(binary):
    """
    Find the digit's bounding box and crop tightly to it.
    Returns the cropped image, or None if no digit is found.

    Normalises both position and scale — a digit that's slightly
    smaller/larger or off-centre in its cell will still match the
    template correctly after this crop + resize to TARGET_SIZE.
    """
    min_area = binary.size * _MIN_DIGIT_AREA_RATIO
    inv = cv2.bitwise_not(binary)   # digit pixels are now white
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not contours:
        return None

    # bounding box that encloses all digit contours
    pts = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(pts)

    # small padding so the digit doesn't touch the edge after resize
    pad = max(2, TARGET_SIZE // 16)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(binary.shape[1], x + w + pad)
    y2 = min(binary.shape[0], y + h + pad)
    return binary[y1:y2, x1:x2]


def preprocess_for_matching(cell_or_template):
    """
    Normalize any cell/template to a clean binary image
    for consistent template matching.
    """
    if len(cell_or_template.shape) == 3:
        gray = cv2.cvtColor(cell_or_template, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_or_template.copy()

    # trim borders — 15% removes grid lines and the thick 3x3 separator bleed
    h, w = gray.shape
    mh, mw = int(h * 0.15), int(w * 0.15)
    gray = gray[mh:h-mh, mw:w-mw]

    # work at 2× target size before cropping for better contour precision
    gray = cv2.resize(gray, (TARGET_SIZE * 2, TARGET_SIZE * 2), interpolation=cv2.INTER_CUBIC)
    binary = _binarize(gray)
    cropped = _crop_to_digit(binary)

    if cropped is None:
        # empty cell or unreadable — return all-white (will score ~0 against any template)
        return np.ones((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8) * 255

    resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    _, result = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return result


def load_templates():
    """
    Load and preprocess all templates once at startup.
    Supports multiple templates per digit: name them 8.jpg, 8b.jpg, 8c.jpg, etc.
    Returns dict mapping digit -> list of processed images.
    """
    templates = {}
    for d in range(1, 10):
        variants = []
        # primary
        path = f"{TEMPLATES_DIR}/{d}.jpg"
        if os.path.exists(path):
            variants.append(preprocess_for_matching(cv2.imread(path)))
        # extra variants: 8b.jpg, 8c.jpg, ...
        for suffix in "bcdefghij":
            path = f"{TEMPLATES_DIR}/{d}{suffix}.jpg"
            if os.path.exists(path):
                variants.append(preprocess_for_matching(cv2.imread(path)))
        if variants:
            templates[d] = variants
        else:
            print(f"⚠️  Missing template for digit {d}")
    return templates


def read_digit_opencv(cell, templates):
    processed = preprocess_for_matching(cell)

    # all-white → empty cell (no contours found during preprocessing)
    if np.mean(processed) > 250:
        return 0, 0.0

    best_digit, best_score = 0, -1.0
    for digit, variants in templates.items():
        for template in variants:
            result = cv2.matchTemplate(processed, template, cv2.TM_CCOEFF_NORMED)
            score = float(result.max())
            if score > best_score:
                best_score = score
                best_digit = digit

    if best_score < CONFIDENCE_THRESHOLD:
        return 0, round(best_score, 2)

    return best_digit, round(best_score, 2)


def read_grid(grid_img, templates):
    """Read all 81 cells with OpenCV template matching.
    Returns (board 9x9, confidences 9x9)."""
    h, w = grid_img.shape[:2]
    ch, cw = h // 9, w // 9
    board, confs = [], []

    for row in range(9):
        board_row, conf_row = [], []
        for col in range(9):
            cell = grid_img[row*ch:(row+1)*ch, col*cw:(col+1)*cw]
            digit, conf = read_digit_opencv(cell, templates)
            board_row.append(digit)
            conf_row.append(conf)
        board.append(board_row)
        confs.append(conf_row)

    return board, confs
