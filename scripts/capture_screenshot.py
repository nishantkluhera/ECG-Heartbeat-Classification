#!/usr/bin/env python
"""Capture a screenshot of the running Streamlit demo for the README.

Requires the app running on http://localhost:8501 and Playwright + Chromium:
    pip install playwright && python -m playwright install chromium
    python scripts/capture_screenshot.py [full|main]
"""
import os
import sys

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "demo_screenshot.png")
URL = "http://localhost:8501"
MODE = sys.argv[1] if len(sys.argv) > 1 else "main"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1360, "height": 1400}, device_scale_factor=2)
        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_selector("text=ECG Insight", timeout=60000)
        page.wait_for_timeout(2500)
        page.locator('button:has-text("Analyze ECG")').first.click(timeout=60000)
        page.wait_for_selector("text=Most likely category", timeout=60000)
        page.wait_for_timeout(2000)

        if MODE == "main":
            page.locator('section[data-testid="stMain"]').screenshot(path=OUT)
        else:
            page.screenshot(path=OUT, full_page=True)
        browser.close()
    print("Saved", OUT)


if __name__ == "__main__":
    main()
