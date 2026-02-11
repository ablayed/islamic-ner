"""Download Sanadset CSV from Mendeley Data.

Primary source:
https://data.mendeley.com/datasets/5xth87zwb5/4

Behavior:
- Tries a direct/programmatic download into `data/raw/sanadset/sanadset.csv`.
- If direct download is blocked, prints manual download instructions and exits
  gracefully while expecting the CSV at `data/raw/sanadset/sanadset.csv`.

Run:
    python scripts/download_sanadset.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

MENDELEY_DATASET_URL = "https://data.mendeley.com/datasets/5xth87zwb5/4"
REQUEST_TIMEOUT = 60
CHUNK_SIZE = 1024 * 64
EXPECTED_FILENAME = "sanadset.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sanadset"
EXPECTED_PATH = OUTPUT_DIR / EXPECTED_FILENAME


def parse_dataset_url(url: str) -> tuple[str, str]:
    """Extract dataset id and version from a Mendeley dataset URL."""

    match = re.search(r"/datasets/([a-zA-Z0-9]+)/([0-9]+)", url)
    if not match:
        raise ValueError(f"Could not parse dataset id/version from URL: {url}")
    return match.group(1), match.group(2)


def parse_content_disposition_filename(header_value: str | None) -> str | None:
    """Extract a filename from a Content-Disposition header if present."""

    if not header_value:
        return None

    filename_match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', header_value)
    if not filename_match:
        return None

    candidate = filename_match.group(1).strip()
    if not candidate:
        return None
    return Path(candidate).name


def normalize_text(value: str) -> str:
    """Whitespace-normalize a string."""

    return re.sub(r"\s+", " ", value).strip()


def gather_links_from_html(html: str, base_url: str) -> list[str]:
    """Extract likely CSV/download links from HTML."""

    url_candidates: set[str] = set()

    attr_patterns = [
        r'href=["\']([^"\']+)["\']',
        r'data-url=["\']([^"\']+)["\']',
        r'data-file-url=["\']([^"\']+)["\']',
    ]
    for pattern in attr_patterns:
        for raw_link in re.findall(pattern, html, flags=re.IGNORECASE):
            absolute = urljoin(base_url, raw_link)
            lowered = absolute.lower()
            if (
                ".csv" in lowered
                or "file_downloaded" in lowered
                or "/files/" in lowered
                or "download" in lowered
            ):
                url_candidates.add(absolute)

    # Try to parse embedded JSON blobs that may contain file metadata.
    json_blocks = re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in json_blocks:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        url_candidates.update(extract_csv_links_from_json(parsed, base_url))

    return sorted(url_candidates)


def walk_json_nodes(node: Any) -> list[dict[str, Any]]:
    """Yield all dictionary nodes from arbitrary JSON."""

    nodes: list[dict[str, Any]] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            nodes.append(current)
            for value in current.values():
                stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return nodes


def extract_csv_links_from_json(payload: Any, base_url: str) -> set[str]:
    """Extract likely CSV/download links from JSON metadata payloads."""

    links: set[str] = set()
    url_keys = {
        "url",
        "href",
        "link",
        "download",
        "download_url",
        "downloadurl",
        "downloadUrl",
        "content_url",
        "contentUrl",
    }
    name_keys = {"name", "filename", "file_name", "label", "title"}

    for node in walk_json_nodes(payload):
        names = [
            normalize_text(str(node[key]))
            for key in node.keys()
            if key in name_keys and isinstance(node[key], str)
        ]
        urls = [
            urljoin(base_url, normalize_text(str(node[key])))
            for key in node.keys()
            if key in url_keys and isinstance(node[key], str)
        ]

        has_csv_name = any(name.lower().endswith(".csv") for name in names)
        for link in urls:
            lowered = link.lower()
            if has_csv_name or ".csv" in lowered or "file_downloaded" in lowered:
                links.add(link)

    return links


def build_metadata_urls(dataset_id: str, version: str) -> list[str]:
    """Build potential Mendeley metadata endpoints."""

    return [
        f"https://data.mendeley.com/public-api/datasets/{dataset_id}/versions/{version}",
        f"https://data.mendeley.com/public-api/datasets/{dataset_id}",
        f"https://data.mendeley.com/datasets/{dataset_id}/{version}",
    ]


def collect_candidate_download_urls(session: requests.Session) -> list[str]:
    """Collect potential file URLs from API endpoints and landing page HTML."""

    dataset_id, version = parse_dataset_url(MENDELEY_DATASET_URL)
    candidates: set[str] = set()

    for metadata_url in build_metadata_urls(dataset_id, version):
        try:
            response = session.get(metadata_url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            print(f"[WARN] Failed to query metadata URL {metadata_url}: {exc}")
            continue

        if response.ok:
            content_type = response.headers.get("content-type", "").lower()
            if "json" in content_type:
                try:
                    payload = response.json()
                except json.JSONDecodeError:
                    payload = None
                if payload is not None:
                    found = extract_csv_links_from_json(payload, metadata_url)
                    if found:
                        print(f"[INFO] Found {len(found)} candidate link(s) in {metadata_url}")
                        candidates.update(found)
            else:
                found = gather_links_from_html(response.text, metadata_url)
                if found:
                    print(f"[INFO] Found {len(found)} candidate link(s) in {metadata_url}")
                    candidates.update(found)
        else:
            print(f"[WARN] Metadata URL responded with status {response.status_code}: {metadata_url}")

    try:
        landing = session.get(MENDELEY_DATASET_URL, timeout=REQUEST_TIMEOUT)
        landing.raise_for_status()
        html_candidates = gather_links_from_html(landing.text, MENDELEY_DATASET_URL)
        if html_candidates:
            print(f"[INFO] Found {len(html_candidates)} candidate link(s) in dataset page HTML")
            candidates.update(html_candidates)
    except requests.RequestException as exc:
        print(f"[WARN] Could not parse dataset page HTML: {exc}")

    return sorted(candidates)


def is_likely_csv_response(response: requests.Response, source_url: str) -> bool:
    """Best-effort check whether an HTTP response likely contains CSV content."""

    content_type = response.headers.get("content-type", "").lower()
    filename = parse_content_disposition_filename(response.headers.get("content-disposition"))
    source_lower = source_url.lower()

    if "text/csv" in content_type or "application/csv" in content_type:
        return True
    if filename and filename.lower().endswith(".csv"):
        return True
    if ".csv" in source_lower:
        return True
    if "file_downloaded" in source_lower or "/files/" in source_lower:
        return True
    return False


def download_to_expected_path(
    session: requests.Session,
    url: str,
    destination: Path,
) -> bool:
    """Download URL into destination with lightweight progress reporting."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=True) as response:
            response.raise_for_status()

            if not is_likely_csv_response(response, url):
                print(f"[WARN] Skipping non-CSV-like response: {url}")
                return False

            total = int(response.headers.get("content-length", "0") or 0)
            downloaded = 0
            next_report = 10

            with temp_path.open("wb") as out_file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        percent = int((downloaded / max(total, downloaded)) * 100)
                        while percent >= next_report and next_report <= 100:
                            print(f"    {next_report}% ({downloaded:,}/{max(total, downloaded):,} bytes)")
                            next_report += 10

            if downloaded == 0:
                print(f"[WARN] Download produced an empty file: {url}")
                return False

        temp_path.replace(destination)
        return True
    except requests.RequestException as exc:
        print(f"[WARN] Download failed for {url}: {exc}")
    except OSError as exc:
        print(f"[WARN] Could not write {destination}: {exc}")
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return False


def print_manual_download_instructions() -> None:
    """Print manual fallback instructions when direct download is unavailable."""

    print("\n[MANUAL DOWNLOAD REQUIRED]")
    print("Direct download from Mendeley Data appears blocked in this environment.")
    print(f"1. Open: {MENDELEY_DATASET_URL}")
    print("2. Download the main Sanadset CSV file (often named `sanadset.csv` or similar).")
    print(f"3. Place it at: {EXPECTED_PATH}")
    print("4. If the filename differs, rename it to `sanadset.csv`.")
    print("After that, continue with the notebook at:")
    print(f"  {PROJECT_ROOT / 'notebooks' / '03_sanadset_exploration.ipynb'}")


def main() -> int:
    """Run Sanadset download workflow and return process exit code."""

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Target output: {EXPECTED_PATH}")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "islamic-ner-sanadset-downloader/1.0",
            "Accept": "*/*",
        }
    )

    try:
        candidate_urls = collect_candidate_download_urls(session)
        if candidate_urls:
            print(f"Collected {len(candidate_urls)} candidate download URL(s).")
        else:
            print("[WARN] No direct candidate URLs discovered.")

        for index, url in enumerate(candidate_urls, start=1):
            print(f"[{index}/{len(candidate_urls)}] Trying: {url}")
            if download_to_expected_path(session, url, EXPECTED_PATH):
                size = EXPECTED_PATH.stat().st_size
                print(f"[OK] Saved {EXPECTED_PATH.relative_to(PROJECT_ROOT)} ({size:,} bytes)")
                return 0

        print_manual_download_instructions()
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
