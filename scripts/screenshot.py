import os
from playwright.sync_api import sync_playwright, expect

os.makedirs("data/screenshots", exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True) 
    
    for i in range(1, 71):
        context = browser.new_context()
        page = context.new_page()
        
        try:
            print(f"Iteration {i}: Loading page...")
            page.goto("https://sudoku.com/")
            
            cookies = page.get_by_label("We Care About Your Privacy").get_by_text("Accept", exact=True)
            page.wait_for_timeout(5000)
            
            if cookies.is_visible():
                cookies.click()
                expect(cookies).to_be_hidden()
            
            page.evaluate("if(document.querySelector('.game-tip')) document.querySelector('.game-tip').remove();")
            
            page.wait_for_timeout(2000)
            
            screenshot_path = f"data/screenshots/sudoku_{i:03d}.png"
            page.screenshot(path=screenshot_path)
            print(f"Saved: {screenshot_path}")
            
        except Exception as e:
            print(f"Error on iteration {i}: {e}")
            
        finally:
            context.close()
    
    browser.close()