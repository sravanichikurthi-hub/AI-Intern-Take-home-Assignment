"""
scraper.py — Scrapes SHL Individual Test Solutions catalog.
Run once to produce catalog.json; commit the JSON so the API server
never blocks on SHL's website at startup.

Usage:
    python scraper.py           # writes catalog.json
    python scraper.py --out my_catalog.json
"""

import argparse
import json
import logging
import re
import time
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
PAGE_SIZE = 12  # SHL shows 12 items per page

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def _get(url: str, params: Optional[dict] = None, retries: int = 3) -> BeautifulSoup:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.content, "html.parser")
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def _parse_catalog_page(soup: BeautifulSoup) -> list[dict]:
    """Extract assessment rows from one catalog page (type=1 = Individual Test Solutions)."""
    rows = []
    table = soup.find("table")
    if not table:
        return rows

    for tr in table.find_all("tr")[1:]:          # skip header row
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue

        # Column layout: Name | Remote | Adaptive | Test Type(s)
        name_cell = cells[0]
        link_tag = name_cell.find("a", href=True)
        if not link_tag:
            continue

        name = link_tag.get_text(strip=True)
        href = link_tag["href"]
        url = urljoin(BASE_URL, href) if href.startswith("/") else href

        remote = "Yes" in cells[1].get_text()
        adaptive = "Yes" in cells[2].get_text()

        # Test type badges (may be multiple, e.g. "A K")
        type_badges = [
            span.get_text(strip=True)
            for span in cells[3].find_all(["span", "div"])
            if len(span.get_text(strip=True)) == 1
        ]
        if not type_badges:
            # fallback: any single uppercase letter in the cell
            type_badges = re.findall(r"\b[A-ZCDEBS]\b", cells[3].get_text())
        type_badges = list(dict.fromkeys(type_badges))   # deduplicate, preserve order

        rows.append({
            "name": name,
            "url": url,
            "remote_testing": remote,
            "adaptive_irt": adaptive,
            "test_type": " ".join(type_badges) if type_badges else "K",
        })

    return rows


def _fetch_detail(url: str) -> dict:
    """Scrape the individual product page for description, duration, and job levels."""
    try:
        soup = _get(url)
    except RuntimeError:
        return {}

    description = ""
    desc_el = soup.find("div", class_=re.compile(r"product-catalog__description|catalog__body|description"))
    if desc_el:
        description = desc_el.get_text(" ", strip=True)
    else:
        # fall back: first large <p> in main content
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 80:
                description = text
                break

    # Duration
    duration = None
    dur_match = re.search(r"(\d+)\s*(min|minute)", soup.get_text(), re.I)
    if dur_match:
        duration = int(dur_match.group(1))

    # Job levels — often appear as badge chips
    job_levels = [
        el.get_text(strip=True)
        for el in soup.find_all(attrs={"class": re.compile(r"job.?level|level.?chip|level.?badge", re.I)})
        if el.get_text(strip=True)
    ]

    return {
        "description": description,
        "duration_minutes": duration,
        "job_levels": job_levels,
    }


def scrape_catalog(fetch_details: bool = True, delay: float = 0.5) -> list[dict]:
    """
    Scrape all Individual Test Solutions from SHL catalog.

    Args:
        fetch_details: Whether to follow each product link for description/duration.
        delay: Polite delay between requests in seconds.

    Returns:
        List of assessment dicts ready to be saved as catalog.json.
    """
    assessments: list[dict] = []
    seen_urls: set[str] = set()
    start = 0

    log.info("Starting SHL catalog scrape (type=1 = Individual Test Solutions)…")

    while True:
        params = {"type": "1", "start": start}
        log.info("  Fetching page start=%d …", start)
        soup = _get(CATALOG_URL, params=params)
        page_items = _parse_catalog_page(soup)

        if not page_items:
            log.info("  No more items. Done after %d assessments.", len(assessments))
            break

        for item in page_items:
            if item["url"] in seen_urls:
                continue
            seen_urls.add(item["url"])

            if fetch_details:
                log.info("    Fetching detail: %s", item["name"])
                detail = _fetch_detail(item["url"])
                item.update(detail)
                time.sleep(delay)

            assessments.append(item)

        start += PAGE_SIZE
        time.sleep(delay)

    return assessments


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape SHL Individual Test Solutions catalog.")
    parser.add_argument("--out", default="catalog.json", help="Output JSON file path")
    parser.add_argument("--no-details", action="store_true", help="Skip per-product detail pages")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")
    args = parser.parse_args()

    assessments = scrape_catalog(fetch_details=not args.no_details, delay=args.delay)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(assessments, fh, indent=2, ensure_ascii=False)

    log.info("Wrote %d assessments to %s", len(assessments), args.out)


if __name__ == "__main__":
    main()
