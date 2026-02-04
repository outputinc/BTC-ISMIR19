#!/usr/bin/env python
"""
Download MoisesDB dataset from Music AI research page.
"""

import os
import sys
import asyncio
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Please install playwright: pip install playwright && playwright install chromium")
    sys.exit(1)


async def download_moisesdb(output_dir: str):
    """Click the MoisesDB download button and handle terms/modal."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        download_url = None

        def capture_response(response):
            nonlocal download_url
            url = response.url
            if '.zip' in url and 'moises' in url.lower():
                print(f"*** DOWNLOAD URL FOUND: {url}")
                download_url = url

        page.on('response', capture_response)

        print("Loading Music AI research page...")
        await page.goto("https://music.ai/research/", wait_until="networkidle")
        await page.wait_for_timeout(3000)

        # Scroll to datasets section
        await page.evaluate("window.scrollTo(0, 2500)")
        await page.wait_for_timeout(2000)

        # Find and click MoisesDB download button
        moisesdb_text = await page.query_selector('text=MoisesDB is a comprehensive')
        if moisesdb_text:
            print("Found MoisesDB section")
            section = await moisesdb_text.evaluate_handle('el => el.closest("div").parentElement')
            download_btn = await section.query_selector('button:has-text("Download")')

            if download_btn:
                print("Clicking Download button...")
                await download_btn.click()
                await page.wait_for_timeout(3000)

                # Take screenshot to see what happened
                await page.screenshot(path="/tmp/after_click.png", full_page=True)
                print("Screenshot saved to /tmp/after_click.png")

                # Check for modal/dialog
                page_content = await page.content()

                # Look for common modal/dialog patterns
                modals = await page.query_selector_all('[role="dialog"], .modal, [class*="Modal"], [class*="Dialog"]')
                print(f"Found {len(modals)} potential modals/dialogs")

                for i, modal in enumerate(modals):
                    try:
                        is_visible = await modal.is_visible()
                        if is_visible:
                            text = await modal.inner_text()
                            print(f"\n=== Modal {i} content ===")
                            print(text[:1000])

                            # Look for accept/agree/download buttons in modal
                            accept_btns = await modal.query_selector_all('button:has-text("Accept"), button:has-text("Agree"), button:has-text("Download"), a:has-text("Download")')
                            for btn in accept_btns:
                                btn_text = await btn.inner_text()
                                href = await btn.get_attribute('href')
                                print(f"Action button: {btn_text} -> {href}")

                                if href and ('storage' in href or '.zip' in href):
                                    download_url = href
                                    print(f"Found download URL: {download_url}")

                            # Try clicking accept if available
                            if accept_btns:
                                print("Clicking accept/download button in modal...")
                                await accept_btns[0].click()
                                await page.wait_for_timeout(5000)
                    except Exception as e:
                        print(f"Modal error: {e}")

                # Check for any new download links that appeared
                all_links = await page.query_selector_all('a[href*="storage"], a[href*=".zip"], a[href*="download"]')
                for link in all_links:
                    href = await link.get_attribute('href')
                    if href and ('moisesdb' in href.lower() or '.zip' in href):
                        print(f"Found download link: {href}")
                        download_url = href

        if download_url:
            print(f"\n*** Download URL: {download_url}")
            await browser.close()
            return download_url

        # Final attempt - look at all page links
        print("\nSearching entire page for download links...")
        all_hrefs = await page.evaluate('''() => {
            return Array.from(document.querySelectorAll('a[href]'))
                .map(a => a.href)
                .filter(h => h.includes('storage') || h.includes('.zip') || h.includes('moisesdb'));
        }''')
        if all_hrefs:
            print(f"Found hrefs: {all_hrefs}")
            download_url = all_hrefs[0]

        await browser.close()
        return download_url


async def main():
    output_dir = os.path.expanduser("~/datasets/moisesdb")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Attempting to find MoisesDB download URL...")
    print("=" * 60)

    download_url = await download_moisesdb(output_dir)

    if download_url:
        print(f"""
============================================================
DOWNLOAD URL FOUND
============================================================

URL: {download_url}

To download and extract:
  wget -O ~/datasets/moisesdb.zip "{download_url}"
  unzip ~/datasets/moisesdb.zip -d ~/datasets/moisesdb/
  export MOISESDB_PATH=~/datasets/moisesdb
  python prepare_moisesdb.py --data_path ~/datasets/moisesdb
""")
        # Try to download with wget
        import subprocess
        zip_path = os.path.expanduser("~/datasets/moisesdb.zip")
        print(f"Attempting download with wget...")
        try:
            subprocess.run(["wget", "-O", zip_path, download_url], check=True)
            print(f"Downloaded to {zip_path}")
        except Exception as e:
            print(f"wget failed: {e}")
            print("Please download manually using the URL above")
    else:
        print("""
============================================================
MANUAL DOWNLOAD REQUIRED
============================================================

Could not automatically find the download URL.
The website may require accepting terms via a modal dialog.

Please download manually:
1. Open browser: https://music.ai/research/
2. Scroll to "Datasets" section
3. Find "MoisesDB" and click "Download"
4. Accept the license terms (CC BY-NC-SA 4.0)
5. Save the moisesdb.zip file

After downloading:
  mkdir -p ~/datasets/moisesdb
  unzip ~/Downloads/moisesdb.zip -d ~/datasets/moisesdb/
  export MOISESDB_PATH=~/datasets/moisesdb
  python prepare_moisesdb.py --data_path ~/datasets/moisesdb

Expected checksums:
  MD5: 13cf74eda129c38b914a51ea79fb1778
  SHA256: 4cde33ce416ac7c868cffcb60eb31f5c741ab7ae5601cbb9d99ed498b72c48c1
""")


if __name__ == "__main__":
    asyncio.run(main())
