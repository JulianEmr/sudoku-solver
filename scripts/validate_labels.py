import cv2, os

def check_labels(n=5):
    img_dir = "data/train/images"
    lbl_dir = "data/train/labels"
    
    files = os.listdir(img_dir)[:n]
    for img_file in files:
        img = cv2.imread(f"{img_dir}/{img_file}")
        h, w = img.shape[:2]
        
        label_file = img_file.rsplit(".", 1)[0] + ".txt"
        label_path = f"{lbl_dir}/{label_file}"
        
        if not os.path.exists(label_path):
            print(f"⚠️  Missing label for {img_file}")
            continue
            
        with open(label_path) as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        cv2.imshow(f"Check: {img_file}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

check_labels()