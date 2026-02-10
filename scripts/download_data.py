"""Download raw hadith datasets from GitHub.

This script downloads:
1. The `hadith-json` 9-book JSON files to `data/raw/hadith_json/`.
2. All CSV files from `Open-Hadith-Data` to `data/raw/open_hadith/`.

Run:
    python scripts/download_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

HADITH_JSON_OWNER = "AhmedBaset"
HADITH_JSON_REPO = "hadith-json"
HADITH_JSON_SUBDIR = "db/by_book/the_9_books"
HADITH_JSON_FILES = [
    "bukhari.json",
    "muslim.json",
    "abudawud.json",
    "ibnmajah.json",
    "nasai.json",
    "tirmidhi.json",
    "malik.json",
    "ahmad.json",
    "darimi.json",
]
HADITH_JSON_REMOTE_CANDIDATES = {
    # Upstream repository uses `ahmed.json`, but local output remains `ahmad.json`.
    "ahmad.json": ["ahmad.json", "ahmed.json"],
}

OPEN_HADITH_OWNER = "mhashim6"
OPEN_HADITH_REPO = "Open-Hadith-Data"

GITHUB_API_BASE = "https://api.github.com/repos"
RAW_GITHUB_BASE = "https://raw.githubusercontent.com"

REQUEST_TIMEOUT = 60
CHUNK_SIZE = 1024 * 64

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HADITH_JSON_OUT_DIR = PROJECT_ROOT / "data" / "raw" / "hadith_json"
OPEN_HADITH_OUT_DIR = PROJECT_ROOT / "data" / "raw" / "open_hadith"


def fetch_repo_default_branch(session: requests.Session, owner: str, repo: str) -> str:
    """Fetch and return the default branch name for a GitHub repository."""

    url = f"{GITHUB_API_BASE}/{owner}/{repo}"
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    payload = response.json()
    branch = payload.get("default_branch")
    if not isinstance(branch, str) or not branch:
        raise RuntimeError(f"Could not resolve default branch for {owner}/{repo}.")
    return branch


def print_byte_progress(downloaded: int, total: int) -> None:
    """Print lightweight transfer progress for a single file."""

    if total > 0:
        percent = (downloaded / total) * 100
        print(f"    {percent:6.2f}% ({downloaded:,}/{total:,} bytes)")
        return

    print(f"    downloaded {downloaded:,} bytes")


def download_file(session: requests.Session, url: str, destination: Path) -> bool:
    """Download a URL into a destination file with simple progress updates."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", "0") or 0)
            downloaded = 0
            next_report_percent = 10
            next_report_bytes = 1024 * 1024

            with temp_path.open("wb") as out_file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue

                    out_file.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        effective_total = max(total, downloaded)
                        percent = int((downloaded / effective_total) * 100)
                        while percent >= next_report_percent and next_report_percent <= 100:
                            print_byte_progress(downloaded, effective_total)
                            next_report_percent += 10
                    elif downloaded >= next_report_bytes:
                        print_byte_progress(downloaded, total)
                        next_report_bytes += 1024 * 1024

            if total > 0 and next_report_percent <= 100:
                print_byte_progress(downloaded, max(total, downloaded))

        temp_path.replace(destination)
        return True
    except requests.RequestException as exc:
        print(f"  [ERROR] Network error for {url}: {exc}")
    except OSError as exc:
        print(f"  [ERROR] File write error for {destination}: {exc}")
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return False


def download_hadith_json(session: requests.Session, branch: str) -> tuple[int, int]:
    """Download the fixed 9 hadith-json files."""

    HADITH_JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(HADITH_JSON_FILES)
    success = 0

    print("\n=== Downloading hadith-json (the 9 books) ===")
    for index, filename in enumerate(HADITH_JSON_FILES, start=1):
        destination = HADITH_JSON_OUT_DIR / filename
        remote_candidates = HADITH_JSON_REMOTE_CANDIDATES.get(filename, [filename])

        print(f"[{index}/{total}] {filename}")
        downloaded = False
        for remote_name in remote_candidates:
            url = (
                f"{RAW_GITHUB_BASE}/{HADITH_JSON_OWNER}/{HADITH_JSON_REPO}/"
                f"{branch}/{HADITH_JSON_SUBDIR}/{remote_name}"
            )
            if download_file(session, url, destination):
                file_size = destination.stat().st_size
                print(f"  [OK] Saved {destination.relative_to(PROJECT_ROOT)} ({file_size:,} bytes)")
                if remote_name != filename:
                    print(f"  [INFO] Used upstream fallback filename: {remote_name}")
                success += 1
                downloaded = True
                break

        if not downloaded:
            candidate_list = ", ".join(remote_candidates)
            print(f"  [ERROR] Could not download {filename}. Tried: {candidate_list}")

    return success, total


def fetch_open_hadith_csv_paths(
    session: requests.Session,
    branch: str,
) -> list[str]:
    """List all CSV file paths in the Open-Hadith-Data repository tree."""

    url = (
        f"{GITHUB_API_BASE}/{OPEN_HADITH_OWNER}/{OPEN_HADITH_REPO}/"
        f"git/trees/{branch}?recursive=1"
    )
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    payload = response.json()
    tree = payload.get("tree")
    if not isinstance(tree, list):
        raise RuntimeError("Unexpected Open-Hadith-Data tree payload from GitHub API.")

    csv_paths = []
    for node in tree:
        if not isinstance(node, dict):
            continue

        path = node.get("path")
        node_type = node.get("type")
        if node_type == "blob" and isinstance(path, str) and path.lower().endswith(".csv"):
            csv_paths.append(path)

    return sorted(csv_paths)


def detect_tashkeel_variants(csv_paths: list[str]) -> tuple[bool, bool]:
    """Heuristically detect whether with/without-tashkeel files are present."""

    lower_paths = [path.lower() for path in csv_paths]
    without_markers = ("without", "no_tashkeel", "no-tashkeel", "without_tashkeel")
    with_markers = ("tashkeel", "diacritic", "mushakkala")

    has_without = any(any(marker in path for marker in without_markers) for path in lower_paths) or any(
        not any(marker in path for marker in with_markers) for path in lower_paths
    )
    has_with = any(any(marker in path for marker in with_markers) for path in lower_paths)

    return has_with, has_without


def download_open_hadith_csvs(session: requests.Session, branch: str) -> tuple[int, int]:
    """Download CSV files from Open-Hadith-Data into data/raw/open_hadith/."""

    OPEN_HADITH_OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        csv_paths = fetch_open_hadith_csv_paths(session, branch)
    except requests.RequestException as exc:
        print(f"[ERROR] Failed to fetch Open-Hadith-Data file list: {exc}")
        return 0, 0
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse Open-Hadith-Data tree response: {exc}")
        return 0, 0
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 0, 0

    if not csv_paths:
        print("[ERROR] No CSV files were found in Open-Hadith-Data.")
        return 0, 0

    print("\n=== Downloading Open-Hadith-Data CSV files ===")
    print(f"Found {len(csv_paths)} CSV files in the repository.")

    has_with, has_without = detect_tashkeel_variants(csv_paths)
    if has_with and has_without:
        print("Detected both with-tashkeel and without-tashkeel CSV naming patterns.")
    else:
        print(
            "[WARN] Could not confidently classify both tashkeel variants by file name. "
            "All CSV files will still be downloaded."
        )

    success = 0
    total = len(csv_paths)

    for index, relative_path in enumerate(csv_paths, start=1):
        url = (
            f"{RAW_GITHUB_BASE}/{OPEN_HADITH_OWNER}/{OPEN_HADITH_REPO}/"
            f"{branch}/{relative_path}"
        )
        destination = OPEN_HADITH_OUT_DIR / Path(relative_path)

        print(f"[{index}/{total}] {relative_path}")
        if download_file(session, url, destination):
            file_size = destination.stat().st_size
            print(f"  [OK] Saved {destination.relative_to(PROJECT_ROOT)} ({file_size:,} bytes)")
            success += 1

    return success, total


def main() -> int:
    """Run full dataset download workflow and return process exit code."""

    print(f"Project root: {PROJECT_ROOT}")
    print("Starting dataset downloads...\n")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "islamic-ner-data-downloader/1.0",
            "Accept": "application/vnd.github+json",
        }
    )

    try:
        hadith_branch = fetch_repo_default_branch(session, HADITH_JSON_OWNER, HADITH_JSON_REPO)
        open_hadith_branch = fetch_repo_default_branch(
            session,
            OPEN_HADITH_OWNER,
            OPEN_HADITH_REPO,
        )

        print(f"hadith-json default branch: {hadith_branch}")
        print(f"Open-Hadith-Data default branch: {open_hadith_branch}")

        hadith_success, hadith_total = download_hadith_json(session, hadith_branch)
        open_success, open_total = download_open_hadith_csvs(session, open_hadith_branch)
    except requests.RequestException as exc:
        print(f"[ERROR] Could not communicate with GitHub: {exc}")
        return 1
    finally:
        session.close()

    failures = (hadith_total - hadith_success) + (open_total - open_success)

    print("\n=== Download Summary ===")
    print(f"hadith-json: {hadith_success}/{hadith_total} files downloaded")
    print(f"open_hadith: {open_success}/{open_total} files downloaded")

    if failures > 0:
        print(f"[WARN] Completed with {failures} failed download(s).")
        return 1

    print("All datasets downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
