"""Debug: show all 9 processed templates in a grid."""
import cv2
import numpy as np
from src.recognise_digits import load_templates

templates = load_templates()

size = 200
cols = 3
rows = 3
pad = 10
cell_w = size + pad * 2
cell_h = size + pad * 2 + 30  # 30px label bar

canvas = np.full((rows * cell_h, cols * cell_w, 3), 50, dtype=np.uint8)

for i, digit in enumerate(range(1, 10)):
    r, c = divmod(i, cols)
    variants = templates[digit]
    # show all variants side by side within the cell slot
    slot_w = cell_w - pad * 2
    var_w = slot_w // len(variants)
    for vi, tmpl in enumerate(variants):
        img = cv2.resize(tmpl, (var_w, size), interpolation=cv2.INTER_NEAREST)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x = c * cell_w + pad + vi * var_w
        y = r * cell_h + pad
        canvas[y:y+size, x:x+var_w] = img_bgr
    label = f"digit {digit}" + (f" x{len(variants)}" if len(variants) > 1 else "")
    cv2.putText(canvas, label, (c * cell_w + pad, r * cell_h + pad + size + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 1, cv2.LINE_AA)

cv2.imshow("Processed templates", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
