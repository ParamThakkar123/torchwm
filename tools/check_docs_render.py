#!/usr/bin/env python3
"""Headless check: open built docs pages, capture console logs and screenshots.

Saves output to tools/render_check/ with screenshots and console logs.
"""

from pathlib import Path
import sys
import time

try:
    from playwright.sync_api import sync_playwright
except Exception as e:
    print("Playwright not installed:", e)
    print(
        "Install with: python -m pip install playwright && python -m playwright install chromium"
    )
    sys.exit(2)


OUT_DIR = Path("tools/render_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def file_url(path: Path) -> str:
    p = path.resolve()
    # Convert to file URI with forward slashes
    return "file:///" + p.as_posix()


def check_page(page_path: Path, name: str):
    url = file_url(page_path)
    print(f"Opening {url}")
    logs = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1200, "height": 900})
        page = context.new_page()

        def on_console(msg):
            text = f"CONSOLE [{msg.type}] {msg.text}"
            logs.append(text)
            print(text)

        page.on("console", on_console)

        # Catch page errors
        def on_page_error(err):
            text = f"PAGE_ERROR {err}"
            logs.append(text)
            print(text)

        page.on("pageerror", on_page_error)

        page.goto(url, wait_until="networkidle")
        # give client-side scripts time to run (Mermaid/MathJax)
        time.sleep(2)

        # wait briefly for mermaid svg if present
        try:
            page.wait_for_selector(".mermaid svg", timeout=3000)
            print("Found .mermaid svg element")
        except Exception:
            print("No .mermaid svg found within timeout")

        # capture screenshot and save page html
        screenshot = OUT_DIR / f"{name}.png"
        html_file = OUT_DIR / f"{name}.html"
        page.screenshot(path=str(screenshot), full_page=True)
        html_file.write_text(page.content(), encoding="utf-8")
        print(f"Saved screenshot: {screenshot}")
        print(f"Saved page html: {html_file}")

        # save console log
        log_file = OUT_DIR / f"{name}.console.log"
        log_file.write_text("\n".join(logs), encoding="utf-8")
        print(f"Saved console log: {log_file}")

        context.close()
        browser.close()


def main():
    base = Path("docs/build/html")
    pages = [
        (base / "dit.html", "dit"),
        (base / "dreamer.html", "dreamer"),
        (base / "jepa.html", "jepa"),
        (base / "iris.html", "iris"),
    ]
    for path, name in pages:
        if path.exists():
            try:
                check_page(path, name)
            except Exception as e:
                print(f"Error checking {path}: {e}")
        else:
            print(f"Page not found: {path}")


if __name__ == "__main__":
    main()
