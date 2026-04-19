# scripts/benchmark.py
import cv2, os, time, json
import random
from src.detect_grid import load_model, detect_grid
from src.recognise_digits import load_templates, read_grid
from scripts.tesseract_digits import read_grid_tesseract

IMG_DIR      = "data/real"
WEIGHTS      = "runs/detect/sudoku_grid_detector/weights/best.pt"
RESULTS_DIR  = "benchmark_results"
N_IMAGES     = 10

def run_benchmark():
    model     = load_model(WEIGHTS)
    templates = load_templates()
    img_list = os.listdir(IMG_DIR)
    random.shuffle(img_list)
    images    = [f for f in img_list if not f.startswith('.')][:N_IMAGES]
    summary   = {"opencv": [], "tesseract": []}


    for img_file in images:
        img = cv2.imread(f"{IMG_DIR}/{img_file}")
        try:
            grid, _, conf = detect_grid(img, model)
        except RuntimeError:
            print(f"⚠️  Skipping {img_file}")
            continue

        for method in ["opencv", "tesseract"]:
            start = time.time()
            if method == "opencv":
                board, confs = read_grid(grid, templates)
            else:
                board, confs = read_grid_tesseract(grid)
            elapsed = (time.time() - start) * 1000

            filled = sum(d != 0 for row in board for d in row)
            avg_conf = sum(c for row in confs for c in row if c > 0) / max(filled, 1)

            summary[method].append({
                "file": img_file,
                "time_ms": round(elapsed, 1),
                "digits_found": filled,
                "avg_conf": round(avg_conf, 2)
            })
            print(f"[{method:10}] {img_file} → {filled}/81 digits | {elapsed:.0f}ms | conf {avg_conf:.2f}")

    aggregates = {}
    print("\n=== BENCHMARK SUMMARY ===")
    for method, runs in summary.items():
        avg_time   = sum(r["time_ms"]      for r in runs) / len(runs)
        avg_digits = sum(r["digits_found"] for r in runs) / len(runs)
        avg_conf   = sum(r["avg_conf"]     for r in runs) / len(runs)
        print(f"{method:10} | {avg_time:.0f}ms avg | {avg_digits:.1f}/81 digits | conf {avg_conf:.2f}")
        aggregates[method] = {"avg_time_ms": round(avg_time, 1), "avg_digits": round(avg_digits, 1), "avg_conf": round(avg_conf, 3)}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "benchmark.json")
    with open(out_path, "w") as f:
        json.dump({"n_images": N_IMAGES, "details": summary, "summary": aggregates}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return summary

if __name__ == "__main__":
    run_benchmark()