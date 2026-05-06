import asyncio
from playwright.async_api import async_playwright
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "report" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEB_URL = "http://159.65.139.148:8501"
SAMPLE_IMAGE_PATH = Path(__file__).parent.parent / "reports" / "sample_predictions.png"

async def capture_screenshots():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        
        await page.goto(WEB_URL, wait_until="networkidle")
        
        # Capture Diagnosis page (empty state)
        await page.click("text=Diagnosis")
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(1)
        screenshot_path = OUTPUT_DIR / "gambar_4_5_diagnosis_page.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"Captured: {screenshot_path.name}")
        
        # Capture Prediction result (after upload)
        if SAMPLE_IMAGE_PATH.exists():
            file_input = page.locator("input[type='file']")
            await file_input.set_input_files(str(SAMPLE_IMAGE_PATH))
            await asyncio.sleep(3)
            screenshot_path = OUTPUT_DIR / "gambar_4_6_prediction_result.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"Captured: {screenshot_path.name}")
        
        # Capture Feedback form (after prediction)
        await asyncio.sleep(1)
        screenshot_path = OUTPUT_DIR / "gambar_4_7_feedback_form.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"Captured: {screenshot_path.name}")
        
        # Navigate to other pages
        pages = {
            "active_learning": "Active Learning",
            "monitoring": "Monitoring",
            "model_registry": "Model Registry"
        }
        
        for page_key, page_name in pages.items():
            try:
                await page.click(f"text={page_name}")
                await page.wait_for_load_state("networkidle")
                await asyncio.sleep(1)
                screenshot_path = OUTPUT_DIR / f"gambar_4_{page_key}_page.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)
                print(f"Captured: {screenshot_path.name}")
            except Exception as e:
                print(f"Failed to capture {page_name}: {e}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(capture_screenshots())
