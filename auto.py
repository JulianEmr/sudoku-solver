"""
Full autonomous pipeline — requires the API server to be running.
Start the server first: uv run uvicorn api:app --port 8000

Usage:
  python auto.py               # open browser, screenshot, solve, fill
  python auto.py --random      # pick a random image from data/real, solve, print result
"""
import os
import random
import argparse
import requests
from src.visualize import print_board

API_URL    = os.environ.get("API_URL", "http://localhost:8000")
SCREENSHOT = "data/real/live.png"
REAL_DIR   = "data/real"


def call_solve(img_path: str) -> dict:
    with open(img_path, "rb") as f:
        resp = requests.post(f"{API_URL}/solve", files={"image": (os.path.basename(img_path), f, "image/png")})
    if not resp.ok:
        raise RuntimeError(f"API error {resp.status_code}: {resp.json().get('detail')}")
    return resp.json()


def print_result(data: dict, img_path: str) -> None:
    filled = sum(d != 0 for row in data["grid"] for d in row)
    print(f"File: {img_path}")
    print(f"Grid detected (conf: {data['confidence']:.2f})  —  {filled}/81 digits read\n")
    print_board(data["grid"])
    print("\nSolution:")
    print_board(data["solution"])


def run_random():
    files = [f for f in os.listdir(REAL_DIR) if not f.startswith(".")]
    if not files:
        raise FileNotFoundError(f"No files found in {REAL_DIR}/")
    img_path = os.path.join(REAL_DIR, random.choice(files))
    print(f"Selected: {img_path}")
    data = call_solve(img_path)
    print_result(data, img_path)


def run_browser():
    from src.browser import open_sudoku, screenshot_page, close_browser, complete_sudoku

    print("Opening sudoku.com...")
    pw, browser, page = open_sudoku()
    try:
        img_path = screenshot_page(page, SCREENSHOT)
        print("Sending screenshot to API...")
        data = call_solve(img_path)
        print_result(data, img_path)
        print("\nFilling grid...")
        complete_sudoku(page, data["solution"])
        print("Done.")
    finally:
        close_browser(pw, browser)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", help=f"Pick a random image from {REAL_DIR}/ and solve it")
    args = parser.parse_args()

    if args.random:
        run_random()
    else:
        run_browser()
