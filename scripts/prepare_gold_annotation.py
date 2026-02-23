"""Prepare a reproducible gold-annotation pack from held-out silver data.

Creates:
1) data/gold/gold_annotation_input.jsonl
2) data/gold/annotation_guide.md

Optional:
3) data/gold/error_analysis_notes.md (starter notes template)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "silver" / "test_held_out.json"
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "data" / "gold" / "gold_annotation_input.jsonl"
DEFAULT_GUIDE_PATH = PROJECT_ROOT / "data" / "gold" / "annotation_guide.md"
DEFAULT_NOTES_PATH = PROJECT_ROOT / "data" / "gold" / "error_analysis_notes.md"


ANNOTATION_GUIDE_TEXT = """# Gold Annotation Guide (Phase 1 NER Schema)

This guide is a quick reference for correcting `gold_labels` in `gold_annotation_input.jsonl`.

## Entity Schema

- `SCHOLAR`: Person names in sanad/matn narration context (narrators, scholars, named persons).
- `BOOK`: Names/titles of hadith books or authored works.
- `CONCEPT`: Islamic concepts/terms (fiqh/aqidah/ibadah terms, doctrinal or legal terms).
- `PLACE`: Geographic locations.
- `HADITH_REF`: Structural references (chapter/hadith numbering patterns, section references).

Use BIO tags:
- `B-<TYPE>` for the first token in an entity span.
- `I-<TYPE>` for continuation tokens in the same span.
- `O` for non-entity tokens.

## Span Rules

- Include full name spans for `SCHOLAR` when name particles are part of the name:
  - Typical inside-name particles: `بن`, `ابن`, `أبو`, `أبي`, `عبد` (context-dependent).
- Exclude honorifics from entity span:
  - Examples: `رضي الله عنه`, `رضي الله عنها`, `رحمه الله`, `صلى الله عليه وسلم`.
- Do not include punctuation in entity spans when punctuation is a separate token.

## BOOK vs SCHOLAR Disambiguation

- If context is clearly a book title (e.g., with cues like `صحيح`, `سنن`, `مسند`, `موطأ`), label as `BOOK`.
- If context is clearly a person mention/narrator chain, label as `SCHOLAR`.
- In ambiguous cases, prefer the local syntactic role in the sentence.

## HADITH_REF vs BOOK

- `HADITH_REF`: structural citation cues such as:
  - `كتاب ...`
  - `باب ...`
  - `حديث رقم ...`
  - `رقم ...`
- `BOOK`: work title itself (for example, the title of a collection).

## Common Corrections To Expect

1. Honorific over-inclusion in SCHOLAR spans
   - Silver: `B-SCHOLAR/I-SCHOLAR` extends into honorific tokens
   - Gold: stop SCHOLAR before honorific, set honorific tokens to `O`

2. Missing name continuation tokens
   - Silver: `B-SCHOLAR` on first token only
   - Gold: continue with `I-SCHOLAR` for full name span

3. SCHOLAR mislabeled where BOOK intended
   - Silver: `B-SCHOLAR` in book-title context
   - Gold: relabel to `B-BOOK` (+ `I-BOOK` if multi-token)

4. Structural citations mislabeled as BOOK/SCHOLAR
   - Silver: book/person label on citation marker text
   - Gold: relabel citation marker phrase to `HADITH_REF`

5. Fragmented spans around punctuation
   - Silver: breaks name/title awkwardly
   - Gold: keep contiguous semantic span with valid BIO sequence

## Annotation Discipline

- If uncertain, keep the least-committal valid span and add a note in `error_analysis_notes.md`.
- Always keep BIO validity:
  - `I-X` cannot start an entity without a preceding `B-X`/`I-X` of same type.
"""


ERROR_NOTES_TEMPLATE = """# Gold Annotation Error Analysis Notes

Use this file during annotation. Add one short bullet per recurring pattern.

## Common Pattern Log

- [ ] Honorific leakage into SCHOLAR spans (e.g., included `رضي الله عنه`)
- [ ] Missing continuation tokens in multi-token scholar names
- [ ] SCHOLAR vs BOOK confusion in title context
- [ ] Citation phrases that should be HADITH_REF
- [ ] Boundary issues around punctuation

## Session Notes

### Session 1
- Reviewed:
- Observed errors:
- Representative IDs:

### Session 2
- Reviewed:
- Observed errors:
- Representative IDs:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare gold annotation inputs from held-out silver set."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to held-out silver JSON file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of sentences to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Output JSONL path for annotation input.",
    )
    parser.add_argument(
        "--guide-path",
        type=Path,
        default=DEFAULT_GUIDE_PATH,
        help="Output path for annotation guide markdown.",
    )
    parser.add_argument(
        "--notes-path",
        type=Path,
        default=DEFAULT_NOTES_PATH,
        help="Output path for error analysis notes markdown.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def load_held_out(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Held-out file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level list in {path}")

    validated: List[Dict] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Row {idx} is not a JSON object.")
        tokens = row.get("tokens")
        labels = row.get("ner_tags")
        if not isinstance(tokens, list) or not isinstance(labels, list):
            raise ValueError(f"Row {idx} missing list 'tokens' or 'ner_tags'.")
        if len(tokens) != len(labels):
            raise ValueError(
                f"Row {idx} token/label length mismatch: {len(tokens)} vs {len(labels)}."
            )
        validated.append(row)
    return validated


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_write_text(path: Path, text: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"File already exists (use --force to overwrite): {path}")
    path.write_text(text, encoding="utf-8")


def build_gold_seed_records(
    rows: List[Dict], sample_size: int, seed: int
) -> List[Dict]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be > 0")
    if sample_size > len(rows):
        raise ValueError(
            f"--sample-size ({sample_size}) exceeds available held-out rows ({len(rows)})."
        )

    rng = random.Random(seed)
    sampled_indices = rng.sample(range(len(rows)), sample_size)

    output_rows: List[Dict] = []
    for pos, source_idx in enumerate(sampled_indices):
        row = rows[source_idx]
        tokens = list(row["tokens"])
        silver_labels = list(row["ner_tags"])
        record_id = str(row.get("id", f"held_out_{source_idx}"))

        output_rows.append(
            {
                "id": record_id,
                "tokens": tokens,
                "silver_labels": silver_labels,
                "gold_labels": list(silver_labels),
            }
        )
    return output_rows


def main() -> int:
    args = parse_args()

    held_out_rows = load_held_out(args.input_path)
    seed_records = build_gold_seed_records(held_out_rows, args.sample_size, args.seed)

    if args.output_jsonl.exists() and not args.force:
        raise FileExistsError(
            f"File already exists (use --force to overwrite): {args.output_jsonl}"
        )
    write_jsonl(args.output_jsonl, seed_records)

    safe_write_text(args.guide_path, ANNOTATION_GUIDE_TEXT, force=args.force)
    if not args.notes_path.exists() or args.force:
        safe_write_text(args.notes_path, ERROR_NOTES_TEMPLATE, force=True)

    print("Prepared gold annotation input.")
    print(f"- Input: {args.input_path}")
    print(f"- Sample size: {len(seed_records)}")
    print(f"- Seed: {args.seed}")
    print(f"- JSONL: {args.output_jsonl}")
    print(f"- Guide: {args.guide_path}")
    print(f"- Notes: {args.notes_path}")
    print("Next: run `python scripts/annotate_gold.py`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
