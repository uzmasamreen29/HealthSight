import os
import json
from datetime import datetime, timezone
from urllib.parse import urljoin
import requests
import pandas as pd
from bs4 import BeautifulSoup

RAW_DIR = os.path.join("data", "raw")
NHS_MANUAL_DIR = os.path.join(RAW_DIR, "nhs")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(NHS_MANUAL_DIR, exist_ok=True)

SF_URL = "https://data.sf.gov/api/v3/views/ppwr-akuv/export.csv?accessType=DOWNLOAD"
NHS_PAGE_URL = "https://digital.nhs.uk/data-and-information/publications/statistical/hospital-admitted-patient-care-activity/2025-26"

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


def inspect_nhs_file(local_path: str) -> dict:
    fname = os.path.basename(local_path)
    if fname.lower().endswith(".csv"):
        df = pd.read_csv(local_path, nrows=10)
        print(f"  CSV [{fname}] Columns ({len(df.columns)}): {list(df.columns)[:12]}")
        return {"file": fname, "type": "csv", "columns": list(df.columns)}
    else:
        xl = pd.ExcelFile(local_path)
        print(f"  Excel [{fname}] Sheets ({len(xl.sheet_names)}): {xl.sheet_names[:12]}")
        return {"file": fname, "type": "excel", "sheets": xl.sheet_names}


def verify_nhs_england():
    print("=" * 70)
    print("VERIFYING NHS ENGLAND HOSPITAL ADMITTED PATIENT CARE (2025-26)")
    print("=" * 70)

    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)

    try:
        # Warm up session on homepage to acquire Akamai/WAF cookies
        session.get("https://digital.nhs.uk/", timeout=30)
        resp = session.get(NHS_PAGE_URL, timeout=60)
        print(f"HTTP Status for NHS Page: {resp.status_code}")
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        file_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if any(href.lower().split("?")[0].endswith(ext) for ext in [".csv", ".xlsx", ".xls"]):
                full_url = urljoin(NHS_PAGE_URL, href)
                title = a.get_text(strip=True) or os.path.basename(href)
                file_links.append({"title": title, "url": full_url})

        print(f"Found {len(file_links)} downloadable CSV/Excel files on page:")
        for idx, item in enumerate(file_links, 1):
            print(f"  [{idx}] {item['title']} -> {item['url']}")

        inspected_files = []
        for item in file_links[:4]:
            fname = os.path.basename(item["url"].split("?")[0])
            local_path = os.path.join(NHS_MANUAL_DIR, fname)
            print(f"\nDownloading: {fname}...")
            f_resp = session.get(item["url"], timeout=60)
            f_resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(f_resp.content)
            inspected_files.append(inspect_nhs_file(local_path))

        return {
            "status": "success",
            "discovered_files": file_links,
            "inspected_samples": inspected_files,
        }

    except Exception as e:
        print(f"Automated fetch blocked/failed: {e}")
        # Check if user placed official NHS files manually in data/raw/nhs/
        local_files = [
            os.path.join(NHS_MANUAL_DIR, f)
            for f in os.listdir(NHS_MANUAL_DIR)
            if f.lower().endswith((".csv", ".xlsx", ".xls"))
        ]
        if local_files:
            print(f"\nFound {len(local_files)} locally downloaded NHS file(s) in {NHS_MANUAL_DIR}:")
            inspected = [inspect_nhs_file(p) for p in local_files]
            return {"status": "local_files_verified", "inspected_samples": inspected}

        print(
            "\nACTION NEEDED IF 403 PERSISTS:\n"
            "1. Open this official link in your browser:\n"
            f"   {NHS_PAGE_URL}\n"
            "2. Download the summary/diagnosis/demographics CSV or XLSX files into:\n"
            f"   {os.path.abspath(NHS_MANUAL_DIR)}\n"
            "3. Re-run `python src/verify_sources.py` to inspect their exact columns/sheets."
        )
        return {"status": "blocked_403", "error": str(e)}


if __name__ == "__main__":
    nhs_report = verify_nhs_england()
    meta_path = os.path.join(RAW_DIR, "nhs_verification_report.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "nhs_england": nhs_report},
            f,
            indent=2,
        )
    print(f"\nSaved NHS verification report to {meta_path}")