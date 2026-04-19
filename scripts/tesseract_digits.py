# scripts/tesseract_digits.py
# Tesseract OCR digit recognition — used by benchmark.py and compare.py only.
import os
import cv2
import pytesseract
from PIL import Image

from src.recognise_digits import CONFIDENCE_THRESHOLD

if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _preprocess_cell(cell):
    h, w = cell.shape[:2]
    mh, mw = int(h * 0.08), int(w * 0.08)
    cell = cell[mh:h-mh, mw:w-mw]
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    thresh = cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return thresh


def read_digit_tesseract(cell):
    """Returns (digit, confidence)."""
    processed = _preprocess_cell(cell)
    pil_img   = Image.fromarray(processed)
    config    = '--psm 6 --oem 3 -c tessedit_char_whitelist=123456789'
    data      = pytesseract.image_to_data(pil_img, config=config,
                                          output_type=pytesseract.Output.DICT)
    for i, text in enumerate(data['text']):
        text = text.strip()
        if text.isdigit() and 1 <= int(text) <= 9:
            conf = round(float(data['conf'][i]) / 100, 2)
            if conf < CONFIDENCE_THRESHOLD:
                return 0, conf
            return int(text), conf
    return 0, 0.0


def read_grid_tesseract(grid_img):
    """Read all 81 cells with Tesseract. Returns (board 9x9, confidences 9x9)."""
    h, w  = grid_img.shape[:2]
    ch, cw = h // 9, w // 9
    board, confs = [], []

    for row in range(9):
        board_row, conf_row = [], []
        for col in range(9):
            cell = grid_img[row*ch:(row+1)*ch, col*cw:(col+1)*cw]
            digit, conf = read_digit_tesseract(cell)
            board_row.append(digit)
            conf_row.append(conf)
        board.append(board_row)
        confs.append(conf_row)

    return board, confs
