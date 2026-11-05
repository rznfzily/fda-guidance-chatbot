"""
FDA Guidance Scraper v2

Reads the website's own table. Metadata is the table's text as shown: Summary (title), Issue Date,
FDA Organization, Topic, Guidance Status and Open for Comment. Nothing is inferred or parsed.
- Rows whose Summary contains "withdrawn" (or withdraw/withdrawal) are dropped.
- Document column is a PDF  -> source_type "pdf", pdf_url taken from that link.
- Document column is blank  -> source_type "html", pdf_url null; the Summary link is kept in
                               landing_page_url and 2_ingest_pdfs.py reads the page text.
- Anything else (e.g. XLSX) is listed as a failure.

No document content is fetched here, only metadata and links.

Output: data/fda_guidance_metadata_v2.json (the original metadata file is not touched).
Usage:  python scripts/1_scrape_metadata_v2.py [--visible]
"""

import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "data" / "fda_guidance_metadata_v2.json"
FAILURES_CSV = REPO_ROOT / "data" / "scrape_failures_v2.csv"

WITHDRAWN_RE = re.compile(r"withdr", re.IGNORECASE)

FDA_SEARCH_URL = "https://www.fda.gov/regulatory-information/search-fda-guidance-documents"


def clear_all_filters(driver):
    """Clear all filters to show all documents"""
    try:
        print("   🧹 Clearing filters...")
        clear_buttons = driver.find_elements(By.XPATH, "//*[contains(text(), 'Clear Filters')]")

        for button in clear_buttons:
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", button)
                time.sleep(0.5)
                button.click()
                print("   ✅ Clicked Clear Filters button")
                time.sleep(3)
                return True
            except:
                continue
        return False
    except Exception as e:
        print(f"   ⚠️  Error clearing filters: {e}")
        return False


def set_show_all_entries(driver):
    """Try to show maximum entries per page"""
    try:
        print("   📊 Setting entries per page...")

        selectors = [
            "select[name*='length']",
            "select[name*='_length']",
            ".dataTables_length select"
        ]

        for selector in selectors:
            try:
                select_elem = driver.find_element(By.CSS_SELECTOR, selector)
                select = Select(select_elem)

                options_text = [opt.text for opt in select.options]

                if "All" in options_text:
                    select.select_by_visible_text("All")
                    print(f"   ✅ Selected 'All' entries")
                else:
                    select.select_by_index(len(select.options) - 1)
                    print(f"   ✅ Selected '{select.first_selected_option.text}' entries")

                time.sleep(3)
                return True
            except:
                continue

        return False
    except Exception as e:
        print(f"   ⚠️  Error setting entries: {e}")
        return False

TABLE_ROWS_JS = """
return Array.from(document.querySelectorAll('table.lcds-datatable--sfgd tbody tr')).map(tr =>
  Array.from(tr.children).map(td => ({
    text: td.innerText.trim(),
    links: Array.from(td.querySelectorAll('a')).map(a => ({text: a.innerText.trim(), href: a.href}))
  }))
);
"""


def setup_driver(headless):
    """Chrome driver with Chrome/chromedriver console logging switched off."""
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--log-level=3")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    return webdriver.Chrome(options=options, service=Service(log_output=subprocess.DEVNULL))


def read_table(driver):
    """Return every row of the (show-all) guidance table as a list of cell dicts."""
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table")))
    time.sleep(3)
    clear_all_filters(driver)
    set_show_all_entries(driver)

    info = ""
    for _ in range(30):
        info = driver.find_element(By.CSS_SELECTOR, ".dataTables_info").text
        m = re.search(r"of ([\d,]+) entries", info)
        if m and "Showing 1 to" in info and int(m.group(1).replace(",", "")) > 100:
            break
        time.sleep(1)
    print(f"   Table says: {info}")

    rows = driver.execute_script(TABLE_ROWS_JS)
    expected = int(re.search(r"of ([\d,]+) entries", info).group(1).replace(",", ""))
    if len(rows) != expected:
        print(f"   WARNING: read {len(rows)} rows but table reports {expected}")
    return rows


def is_pdf_label(label):
    """'PDF (613 KB)' is a PDF. A label with no file type at all, e.g. '()', is a link the site failed to label, so treat it as PDF too."""
    first_line = label.strip().splitlines()[0] if label.strip() else ""
    return first_line.upper().startswith("PDF") or (bool(first_line) and not re.search(r"[A-Za-z]", first_line))


def cell_text(cells, i):
    return cells[i]["text"] if len(cells) > i else ""


def build_record(cells, title, landing_url, pdf_url):
    """Metadata is the table's own cell text, not parsed or inferred."""
    organizations = [o.strip() for o in cell_text(cells, 3).split("\n\n") if o.strip()]
    return {
        "title": title,
        "source_type": "pdf" if pdf_url else "html",
        "pdf_url": pdf_url,
        "landing_page_url": landing_url,
        "issue_date": cell_text(cells, 2),
        "fda_organization": " | ".join(organizations),
        "topic": cell_text(cells, 4),
        "guidance_status": cell_text(cells, 5),
        "open_for_comment": cell_text(cells, 6),
        "scraped_at": datetime.now().isoformat(),
    }


def main():
    headless = "--visible" not in sys.argv
    print("=" * 70)
    print("FDA Guidance Scraper v2")
    print("=" * 70)

    driver = setup_driver(headless)
    try:
        print("\nLoading FDA guidance search page...")
        driver.get(FDA_SEARCH_URL)
        time.sleep(5)
        rows = read_table(driver)
    finally:
        driver.quit()

    failures = []
    records = []
    withdrawn_skipped = 0
    pdf_count = 0
    page_count = 0

    for cells in rows:
        title = cell_text(cells, 0)
        landing_url = cells[0]["links"][0]["href"] if cells and cells[0]["links"] else None

        if WITHDRAWN_RE.search(title):
            withdrawn_skipped += 1
            continue

        doc_text = cell_text(cells, 1)
        if is_pdf_label(doc_text):
            pdf_urls = [l["href"] for l in cells[1]["links"] if is_pdf_label(l["text"])]
            if not pdf_urls:
                failures.append({"title": title, "url": landing_url, "reason": "Document says PDF but has no link"})
            for idx, pdf_url in enumerate(pdf_urls):
                doc_title = f"{title} - PDF {idx + 1}" if len(pdf_urls) > 1 else title
                records.append(build_record(cells, doc_title, landing_url, pdf_url))
                pdf_count += 1
        elif not doc_text:
            if landing_url:
                records.append(build_record(cells, title, landing_url, None))
                page_count += 1
            else:
                failures.append({"title": title, "url": None, "reason": "Document blank and Summary has no link"})
        else:
            failures.append({"title": title, "url": landing_url, "reason": f"Unsupported Document type: {doc_text.splitlines()[0]}"})

    for i, rec in enumerate(records, 1):
        rec["id"] = i

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "scraped_at": datetime.now().isoformat(),
            "total_documents": len(records),
            "failures": failures,
            "documents": records,
        }, f, indent=2, ensure_ascii=False)

    with open(FAILURES_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "url", "reason"])
        writer.writeheader()
        writer.writerows(failures)

    print("\n" + "=" * 70)
    print(f"Rows in table: {len(rows)} | withdrawn skipped: {withdrawn_skipped}")
    print(f"Files extracted: {len(records)}  (PDF links: {pdf_count}, page-text links: {page_count})")
    print(f"Failed: {len(failures)}")
    for fail in failures:
        print(f"   - {fail['title']} | {fail['reason']} | {fail['url']}")
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Failures: {FAILURES_CSV}")


if __name__ == "__main__":
    main()
