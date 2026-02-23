"""Terminal helper for manual gold annotation over JSONL records.

Input record format (one JSON per line):
{
  "id": str,
  "tokens": list[str],
  "silver_labels": list[str],
  "gold_labels": list[str]
}
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "gold" / "gold_annotation_input.jsonl"
DEFAULT_PROGRESS_PATH = PROJECT_ROOT / "data" / "gold" / "gold_annotation_progress.json"
DEFAULT_NOTES_PATH = PROJECT_ROOT / "data" / "gold" / "error_analysis_notes.md"

ENTITY_TYPES = ["SCHOLAR", "BOOK", "CONCEPT", "PLACE", "HADITH_REF"]
VALID_TAGS = ["O"] + [
    f"{prefix}-{entity}" for entity in ENTITY_TYPES for prefix in ("B", "I")
]

ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
ANSI_BOLD = "\033[1m"
COLOR_BY_ENTITY = {
    "SCHOLAR": "\033[96m",  # bright cyan
    "BOOK": "\033[95m",  # bright magenta
    "CONCEPT": "\033[93m",  # bright yellow
    "PLACE": "\033[92m",  # bright green
    "HADITH_REF": "\033[94m",  # bright blue
    "O": "\033[90m",  # gray
}


def configure_stdio() -> None:
    # Windows terminals can default to cp1252, which breaks Arabic token output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive terminal annotation helper for gold NER labels."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="JSONL file to annotate.",
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=DEFAULT_PROGRESS_PATH,
        help="Progress JSON file (reviewed sentence indices).",
    )
    parser.add_argument(
        "--notes-path",
        type=Path,
        default=DEFAULT_NOTES_PATH,
        help="Markdown file for error-pattern notes.",
    )
    parser.add_argument(
        "--session-target",
        type=int,
        default=25,
        help="Stop automatically after reviewing this many sentences in this session.",
    )
    parser.add_argument(
        "--start-over",
        action="store_true",
        help="Ignore existing progress and start fresh from the first sentence.",
    )
    return parser.parse_args()


def now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Annotation input not found: {path}")

    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no}: expected JSON object.")
            validate_record(row, line_no)
            rows.append(row)

    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def validate_record(record: Dict, line_no: int) -> None:
    required = ["id", "tokens", "silver_labels", "gold_labels"]
    for key in required:
        if key not in record:
            raise ValueError(f"Line {line_no}: missing key '{key}'.")

    tokens = record["tokens"]
    silver = record["silver_labels"]
    gold = record["gold_labels"]
    if (
        not isinstance(tokens, list)
        or not isinstance(silver, list)
        or not isinstance(gold, list)
    ):
        raise ValueError(
            f"Line {line_no}: tokens/silver_labels/gold_labels must be lists."
        )
    if len(tokens) != len(silver) or len(tokens) != len(gold):
        raise ValueError(
            f"Line {line_no}: length mismatch tokens={len(tokens)} silver={len(silver)} gold={len(gold)}."
        )
    for tag in gold:
        if str(tag) not in VALID_TAGS:
            raise ValueError(f"Line {line_no}: invalid gold tag '{tag}'.")


def save_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def load_progress(path: Path, total: int, start_over: bool) -> Dict:
    if start_over or not path.exists():
        return {
            "reviewed_indices": [],
            "updated_at": now_utc_iso(),
            "sessions": [],
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid progress JSON format: {path}")
    reviewed = payload.get("reviewed_indices", [])
    if not isinstance(reviewed, list):
        raise ValueError(f"Invalid reviewed_indices in: {path}")

    cleaned = sorted(
        {int(i) for i in reviewed if isinstance(i, int) and 0 <= i < total}
    )
    sessions = payload.get("sessions", [])
    if not isinstance(sessions, list):
        sessions = []
    return {
        "reviewed_indices": cleaned,
        "updated_at": payload.get("updated_at", now_utc_iso()),
        "sessions": sessions,
    }


def save_progress(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = now_utc_iso()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def color_for_tag(tag: str) -> str:
    if tag == "O":
        return COLOR_BY_ENTITY["O"]
    entity = tag.split("-", 1)[1]
    return COLOR_BY_ENTITY.get(entity, "")


def render_record(
    record: Dict,
    sentence_index: int,
    total: int,
    reviewed_count: int,
    session_done: int,
    session_target: int,
) -> None:
    rec_id = str(record["id"])
    tokens = [str(t) for t in record["tokens"]]
    silver = [str(t) for t in record["silver_labels"]]
    gold = [str(t) for t in record["gold_labels"]]

    print("\n" + "=" * 88)
    print(f"Sentence {sentence_index + 1}/{total} | id={rec_id}")
    print(
        f"Reviewed: {reviewed_count}/{total} | Session: {session_done}/{session_target}"
    )
    print("-" * 88)
    print("idx | token | silver -> gold")
    print("-" * 88)

    for idx, (token, silver_tag, gold_tag) in enumerate(zip(tokens, silver, gold)):
        changed = silver_tag != gold_tag
        tag_color = color_for_tag(gold_tag)
        marker = "*" if changed else " "
        gold_str = f"{tag_color}{ANSI_BOLD}{gold_tag}{ANSI_RESET}"
        silver_str = f"{ANSI_DIM}{silver_tag}{ANSI_RESET}"
        print(f"{idx:03d}{marker} | {token} | {silver_str} -> {gold_str}")

    print("-" * 88)
    print("Commands:")
    print("  Enter: accept current gold_labels for this sentence")
    print("  edits: token_index=TAG (or range start-end=TAG), space-separated")
    print("         example: 1=B-SCHOLAR 2=I-SCHOLAR 10-12=O")
    print("  note <text>: append note to error_analysis_notes.md")
    print("  r: reset this sentence gold_labels back to silver_labels")
    print("  s: skip for now")
    print("  q: save and quit")


def validate_bio(tags: Sequence[str]) -> List[str]:
    errors: List[str] = []
    prev_type = ""
    prev_prefix = "O"
    for idx, tag in enumerate(tags):
        if tag not in VALID_TAGS:
            errors.append(f"Invalid tag '{tag}' at token index {idx}.")
            prev_type = ""
            prev_prefix = "O"
            continue
        if tag == "O":
            prev_type = ""
            prev_prefix = "O"
            continue
        prefix, entity = tag.split("-", 1)
        if prefix == "I":
            if prev_prefix not in {"B", "I"} or prev_type != entity:
                prev_tag = "START" if idx == 0 else tags[idx - 1]
                errors.append(
                    f"Invalid I-tag transition at token {idx}: {tag} after {prev_tag}."
                )
        prev_type = entity
        prev_prefix = prefix
    return errors


def parse_edit_command(text: str, token_count: int) -> Dict[int, str]:
    edits: Dict[int, str] = {}
    parts = [p for p in text.strip().split() if p]
    if not parts:
        raise ValueError("No edits provided.")

    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid edit token '{part}'. Expected token_index=TAG.")
        left, tag = part.split("=", 1)
        tag = tag.strip()
        if tag not in VALID_TAGS:
            raise ValueError(f"Invalid tag '{tag}'.")

        left = left.strip()
        if "-" in left:
            start_s, end_s = left.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start > end:
                raise ValueError(f"Invalid range '{left}': start > end.")
            for idx in range(start, end + 1):
                if idx < 0 or idx >= token_count:
                    raise ValueError(
                        f"Index {idx} out of range [0, {token_count - 1}]."
                    )
                edits[idx] = tag
        else:
            idx = int(left)
            if idx < 0 or idx >= token_count:
                raise ValueError(f"Index {idx} out of range [0, {token_count - 1}].")
            edits[idx] = tag
    return edits


def append_note(notes_path: Path, sentence_idx: int, record_id: str, text: str) -> None:
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    if not notes_path.exists():
        notes_path.write_text("# Error analysis notes\n\n", encoding="utf-8")
    with notes_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"- [{now_utc_iso()}] sentence={sentence_idx + 1} id={record_id}: {text}\n"
        )


def mark_reviewed_and_save(
    rows: List[Dict],
    input_path: Path,
    progress: Dict,
    progress_path: Path,
    reviewed_indices: Set[int],
) -> None:
    progress["reviewed_indices"] = sorted(reviewed_indices)
    save_jsonl(input_path, rows)
    save_progress(progress_path, progress)


def main() -> int:
    configure_stdio()
    args = parse_args()
    if args.session_target <= 0:
        raise ValueError("--session-target must be > 0")

    rows = load_jsonl(args.input_path)
    progress = load_progress(args.progress_path, len(rows), args.start_over)
    reviewed_indices: Set[int] = set(progress.get("reviewed_indices", []))

    total = len(rows)
    pending_indices = [idx for idx in range(total) if idx not in reviewed_indices]
    if not pending_indices:
        print("All sentences are already reviewed.")
        print(f"Reviewed: {len(reviewed_indices)}/{total}")
        return 0

    print(f"Loaded {total} sentences from {args.input_path}")
    print(f"Already reviewed: {len(reviewed_indices)}")
    print(f"Session target: {args.session_target} (recommended 20-30 per sitting)")

    session_done = 0
    for idx in pending_indices:
        if session_done >= args.session_target:
            break

        record = rows[idx]
        while True:
            render_record(
                record=record,
                sentence_index=idx,
                total=total,
                reviewed_count=len(reviewed_indices),
                session_done=session_done,
                session_target=args.session_target,
            )

            try:
                raw = input("> ").strip()
            except EOFError:
                raw = "q"

            if raw == "":
                reviewed_indices.add(idx)
                session_done += 1
                mark_reviewed_and_save(
                    rows,
                    args.input_path,
                    progress,
                    args.progress_path,
                    reviewed_indices,
                )
                print(f"Saved. Reviewed {len(reviewed_indices)}/{total}.")
                break

            lowered = raw.lower()
            if lowered in {"q", "quit", "exit"}:
                progress.setdefault("sessions", []).append(
                    {
                        "at": now_utc_iso(),
                        "reviewed_in_session": session_done,
                        "reviewed_total": len(reviewed_indices),
                    }
                )
                save_progress(args.progress_path, progress)
                print("Progress saved. Exiting.")
                return 0

            if lowered in {"s", "skip"}:
                print("Skipped for now.")
                break

            if lowered in {"r", "reset"}:
                record["gold_labels"] = list(record["silver_labels"])
                save_jsonl(args.input_path, rows)
                print("Reset gold labels to silver labels for this sentence.")
                continue

            if lowered.startswith("note ") or lowered.startswith("n "):
                note_text = raw.split(" ", 1)[1].strip() if " " in raw else ""
                if not note_text:
                    print("Note text is empty.")
                    continue
                append_note(args.notes_path, idx, str(record["id"]), note_text)
                print(f"Note saved to {args.notes_path}.")
                continue

            try:
                edits = parse_edit_command(raw, len(record["tokens"]))
                candidate = list(record["gold_labels"])
                for token_idx, tag in edits.items():
                    candidate[token_idx] = tag
                bio_errors = validate_bio(candidate)
                if bio_errors:
                    print("BIO validation failed:")
                    for err in bio_errors:
                        print(f"- {err}")
                    print("No changes saved. Enter corrected edits.")
                    continue

                record["gold_labels"] = candidate
                reviewed_indices.add(idx)
                session_done += 1
                mark_reviewed_and_save(
                    rows,
                    args.input_path,
                    progress,
                    args.progress_path,
                    reviewed_indices,
                )
                print(f"Saved edits. Reviewed {len(reviewed_indices)}/{total}.")
                break
            except Exception as exc:
                print(f"Invalid input: {exc}")
                continue

    progress.setdefault("sessions", []).append(
        {
            "at": now_utc_iso(),
            "reviewed_in_session": session_done,
            "reviewed_total": len(reviewed_indices),
        }
    )
    save_progress(args.progress_path, progress)
    print("\nSession complete.")
    print(f"Reviewed this session: {session_done}")
    print(f"Total reviewed: {len(reviewed_indices)}/{total}")
    if len(reviewed_indices) < total:
        print("Run again to continue: python scripts/annotate_gold.py")
    else:
        print("All sentences reviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
