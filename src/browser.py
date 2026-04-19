"""Browser automation for sudoku.com."""
import os
from time import sleep
from playwright.sync_api import sync_playwright, Page, Browser

SUDOKU_URL = "https://sudoku.com/extreme/"

def open_sudoku() -> tuple:
    """
    Launch browser, navigate to sudoku.com, dismiss cookie banner.
    Returns (playwright, browser, page) — caller must close them.
    Set HEADLESS=1 to run without a display (e.g. inside Docker).
    """
    headless = os.environ.get("HEADLESS", "0") == "1"
    pw = sync_playwright().start()
    browser: Browser = pw.chromium.launch(headless=headless)
    page: Page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(SUDOKU_URL)
    sleep(5)

    try:
        page.click("#onetrust-accept-btn-handler", timeout=3000)
        sleep(5)
    except Exception:
        pass

    return pw, browser, page


def screenshot_page(page: Page, output_path: str = "data/real/live.png") -> str:
    """Take a screenshot of the current page. Returns the output path."""
    page.screenshot(path=output_path, full_page=False)
    return output_path


def close_browser(pw, browser: Browser) -> None:
    browser.close()
    pw.stop()

def complete_sudoku(page: Page, board) -> None:
    """
    Fill the sudoku grid on the page with the given board.
    Assumes the page is already open and showing the sudoku grid.
    """
    for row in range(9):
        for col in range(9):
            digit = board[row][col]
            page.press("body", "Digit"+str(digit))
            page.press("body", "ArrowRight")
        page.press("body", "ArrowDown")
        for _ in range(9):
            page.press("body", "ArrowLeft")