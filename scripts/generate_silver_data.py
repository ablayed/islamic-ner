"""Generate silver Arabic NER data from Sanadset and hadith-json unified CSV.

Outputs:
- data/silver/silver_train.json
- data/silver/silver_stats.json
- data/silver/quality_sample.json
- data/silver/train.json
- data/silver/dev.json
- data/silver/test_held_out.json

Run:
    python scripts/generate_silver_data.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: object):  # type: ignore[misc]
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ner import SilverAnnotator
from src.preprocessing.normalize import ArabicNormalizer


ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "cp1256", "windows-1256"]
TAG_SPLIT_RE = re.compile(r"(<[^>]+>)")
TAG_FULL_RE = re.compile(r"^<[^>]+>$")

BOOK_ID_TO_NAME = {
    1: "bukhari",
    2: "muslim",
    3: "abudawud",
    4: "ibnmajah",
    5: "nasai",
    6: "tirmidhi",
    7: "malik",
    8: "ahmad",
    9: "darimi",
}

EXPECTED_ENTITY_TYPES = ["SCHOLAR", "BOOK", "CONCEPT", "PLACE", "HADITH_REF"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate silver NER dataset files.")
    parser.add_argument(
        "--sanadset-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "sanadset" / "sanadset.csv",
        help="Path to Sanadset CSV.",
    )
    parser.add_argument(
        "--hadith-unified-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "hadith_unified.csv",
        help="Path to Phase 1 hadith-json unified CSV.",
    )
    parser.add_argument(
        "--gazetteer-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "gazetteers",
        help="Directory containing gazetteer files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "silver",
        help="Output directory for generated silver files.",
    )
    parser.add_argument(
        "--sanadset-nrows",
        type=int,
        default=50000,
        help="Row cap for Sanadset CSV load before sampling.",
    )
    parser.add_argument(
        "--sanadset-sample-size",
        type=int,
        default=3000,
        help="Random sample size from loaded Sanadset rows to annotate.",
    )
    parser.add_argument(
        "--hadith-unified-nrows",
        type=int,
        default=None,
        help="Optional row cap for hadith unified CSV.",
    )
    parser.add_argument(
        "--hadith-sample-size",
        type=int,
        default=1000,
        help="Random sample size from hadith unified rows to annotate.",
    )
    parser.add_argument(
        "--quality-sample-size",
        type=int,
        default=50,
        help="Number of random records to export for manual quality review.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle/sampling.",
    )
    return parser.parse_args()


def load_csv_with_fallback(path: Path, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    last_error: Optional[Exception] = None
    for enc in ENCODING_CANDIDATES:
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows), enc
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(f"Failed reading {path} with encodings {ENCODING_CANDIDATES}: {last_error}")


def find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    lower_map = {column.lower(): column for column in df.columns}
    for name in candidates:
        column = lower_map.get(name.lower())
        if column is not None:
            return column
    return ""


def detect_sanadset_text_column(df: pd.DataFrame, sample_size: int = 3000) -> str:
    scored: List[Tuple[float, float, str]] = []
    for col in df.columns:
        series = df[col]
        if not (pd.api.types.is_string_dtype(series) or series.dtype == object):
            continue

        sample = series.dropna().astype(str).head(sample_size)
        if sample.empty:
            continue

        nar_hit = sample.str.contains(r"<\s*NAR\s*>", regex=True, case=False).mean()
        tag_hit = sample.str.contains(r"</?(?:NAR|SANAD|MATN)>", regex=True, case=False).mean()
        scored.append((float(nar_hit), float(tag_hit), col))

    if not scored:
        fallback = find_first_column(df, ["hadith", "text", "tagged_text", "hadith_text", "content"])
        if fallback:
            return fallback
        raise RuntimeError("Could not detect Sanadset text column.")

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if scored[0][0] > 0:
        return scored[0][2]

    fallback = find_first_column(df, ["hadith", "text", "tagged_text", "hadith_text", "content"])
    if fallback:
        return fallback
    return scored[0][2]


def detect_hadith_unified_text_column(df: pd.DataFrame) -> str:
    text_col = find_first_column(df, ["arabic_text", "text", "hadith_text", "arabic_normalized"])
    if not text_col:
        raise RuntimeError("Could not detect text column in hadith unified CSV.")
    return text_col


def sample_rows(df: pd.DataFrame, text_col: str, sample_size: int, seed: int) -> pd.DataFrame:
    if text_col not in df.columns:
        return df.copy()

    pool = df[df[text_col].notna()]
    if sample_size <= 0 or sample_size >= len(pool):
        return pool.copy()

    return pool.sample(n=sample_size, random_state=seed)


def normalize_tagged_text(tagged_text: str, normalizer: ArabicNormalizer) -> str:
    text = str(tagged_text)
    parts = TAG_SPLIT_RE.split(text)
    out_parts: List[str] = []

    for part in parts:
        if not part:
            continue
        if TAG_FULL_RE.match(part):
            out_parts.append(part)
            continue

        normalized = normalizer.normalize(part)
        if normalized:
            out_parts.append(f" {normalized} ")

    joined = "".join(out_parts)
    return re.sub(r"\s+", " ", joined).strip()


def resolve_book_name(raw_value: object) -> str:
    if pd.isna(raw_value):
        return "unknown"

    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if not value:
            return "unknown"
        if value.isdigit():
            return BOOK_ID_TO_NAME.get(int(value), value)
        return value

    if isinstance(raw_value, (int, float)):
        as_int = int(raw_value)
        return BOOK_ID_TO_NAME.get(as_int, str(as_int))

    return str(raw_value).strip().lower() or "unknown"


def split_annotations(annotations: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    tokens = [token for token, _ in annotations]
    labels = [label for _, label in annotations]
    return tokens, labels


def count_entities(labels: List[str]) -> int:
    return sum(1 for label in labels if label.startswith("B-"))


def collect_entity_counts(records: List[Dict]) -> Dict[str, int]:
    counts = {entity_type: 0 for entity_type in EXPECTED_ENTITY_TYPES}
    for record in records:
        for label in record["ner_tags"]:
            if not label.startswith("B-"):
                continue
            entity_type = label[2:]
            counts[entity_type] = counts.get(entity_type, 0) + 1
    return counts


def collect_label_distribution(records: List[Dict]) -> Dict[str, float]:
    label_counter: Counter = Counter()
    total = 0
    for record in records:
        labels = record["ner_tags"]
        label_counter.update(labels)
        total += len(labels)

    if total == 0:
        return {}

    distribution = {label: round((count / total) * 100, 3) for label, count in sorted(label_counter.items())}
    return distribution


def passes_training_filters(tokens: List[str], labels: List[str]) -> bool:
    if len(tokens) < 5 or len(tokens) > 128:
        return False
    if count_entities(labels) == 0:
        return False
    return True


def to_quality_item(record: Dict) -> Dict:
    entities = []
    current_type = ""
    current_tokens: List[str] = []
    start_index = 0

    for idx, (token, label) in enumerate(zip(record["tokens"], record["ner_tags"])):
        if label == "O":
            if current_type:
                entities.append(
                    {
                        "type": current_type,
                        "start_token": start_index,
                        "end_token": idx,
                        "text": " ".join(current_tokens),
                    }
                )
            current_type = ""
            current_tokens = []
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or (prefix == "I" and entity_type != current_type):
            if current_type:
                entities.append(
                    {
                        "type": current_type,
                        "start_token": start_index,
                        "end_token": idx,
                        "text": " ".join(current_tokens),
                    }
                )
            current_type = entity_type
            current_tokens = [token]
            start_index = idx
        else:
            current_tokens.append(token)

    if current_type:
        entities.append(
            {
                "type": current_type,
                "start_token": start_index,
                "end_token": len(record["tokens"]),
                "text": " ".join(current_tokens),
            }
        )

    return {
        "id": record["id"],
        "source": record["source"],
        "book": record["book"],
        "text": " ".join(record["tokens"]),
        "tokens": record["tokens"],
        "ner_tags": record["ner_tags"],
        "entities": entities,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    pipeline_start = perf_counter()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Sanadset CSV: {args.sanadset_path}")
    print(f"Hadith unified CSV: {args.hadith_unified_path}")
    print(f"Gazetteers: {args.gazetteer_dir}")
    print(f"Output dir: {args.output_dir}")

    annotator = SilverAnnotator(gazetteer_dir=str(args.gazetteer_dir))
    normalizer = ArabicNormalizer()

    load_start = perf_counter()
    sanadset_df, sanadset_encoding = load_csv_with_fallback(args.sanadset_path, nrows=args.sanadset_nrows)
    hadith_df, hadith_encoding = load_csv_with_fallback(args.hadith_unified_path, nrows=args.hadith_unified_nrows)
    load_elapsed = perf_counter() - load_start

    sanadset_text_col = detect_sanadset_text_column(sanadset_df)
    sanadset_id_col = find_first_column(sanadset_df, ["id", "hadith_id", "index", "row_id"])
    sanadset_book_col = find_first_column(sanadset_df, ["book", "book_name", "collection"])

    hadith_text_col = detect_hadith_unified_text_column(hadith_df)
    hadith_id_col = find_first_column(hadith_df, ["hadith_id", "id", "number"])
    hadith_book_col = find_first_column(hadith_df, ["book", "book_name", "collection"])

    print("\nDetected columns:")
    print(f"- Sanadset text: {sanadset_text_col} (encoding={sanadset_encoding})")
    print(f"- Sanadset id: {sanadset_id_col or '[index fallback]'}")
    print(f"- Sanadset book: {sanadset_book_col or '[none]'}")
    print(f"- Hadith unified text: {hadith_text_col} (encoding={hadith_encoding})")
    print(f"- Hadith unified id: {hadith_id_col or '[index fallback]'}")
    print(f"- Hadith unified book: {hadith_book_col or '[none]'}")

    sanadset_sample_df = sample_rows(
        sanadset_df,
        sanadset_text_col,
        sample_size=args.sanadset_sample_size,
        seed=args.seed,
    )
    hadith_sample_df = sample_rows(
        hadith_df,
        hadith_text_col,
        sample_size=args.hadith_sample_size,
        seed=args.seed,
    )

    print("\nSampling plan:")
    print(
        f"- Sanadset loaded rows: {len(sanadset_df):,} "
        f"(nrows={args.sanadset_nrows if args.sanadset_nrows is not None else 'all'})"
    )
    print(f"- Sanadset sampled rows: {len(sanadset_sample_df):,} (target={args.sanadset_sample_size:,})")
    print(
        f"- Hadith unified loaded rows: {len(hadith_df):,} "
        f"(nrows={args.hadith_unified_nrows if args.hadith_unified_nrows is not None else 'all'})"
    )
    print(f"- Hadith unified sampled rows: {len(hadith_sample_df):,} (target={args.hadith_sample_size:,})")

    records: List[Dict] = []
    rejected_zero_entities = 0
    rejected_token_len = 0

    timing = {
        "load_csv_s": load_elapsed,
        "sanadset_normalize_s": 0.0,
        "hadith_normalize_s": 0.0,
        "sanadset_annotate_s": 0.0,
        "hadith_annotate_s": 0.0,
    }

    print("\nAnnotating Sanadset sample...")
    sanadset_start = perf_counter()
    sanadset_rows = sanadset_sample_df.to_dict(orient="records")
    sanadset_indexes = sanadset_sample_df.index.tolist()
    for pos, row in enumerate(
        tqdm(sanadset_rows, total=len(sanadset_rows), desc="Sanadset", unit="sent")
    ):
        row_idx = sanadset_indexes[pos]
        raw_text = row.get(sanadset_text_col)
        if pd.isna(raw_text):
            continue

        normalize_start = perf_counter()
        normalized_tagged = normalize_tagged_text(str(raw_text), normalizer)
        timing["sanadset_normalize_s"] += perf_counter() - normalize_start

        annotate_start = perf_counter()
        annotations = annotator.annotate_from_sanadset(normalized_tagged)
        timing["sanadset_annotate_s"] += perf_counter() - annotate_start
        tokens, labels = split_annotations(annotations)

        if count_entities(labels) == 0:
            rejected_zero_entities += 1
            continue
        if len(tokens) < 5 or len(tokens) > 128:
            rejected_token_len += 1
            continue

        raw_id = row.get(sanadset_id_col) if sanadset_id_col else row_idx
        book_name = resolve_book_name(row.get(sanadset_book_col)) if sanadset_book_col else "unknown"

        records.append(
            {
                "id": f"sanadset_{raw_id}",
                "tokens": tokens,
                "ner_tags": labels,
                "source": "sanadset",
                "book": book_name,
            }
        )
    timing["sanadset_total_s"] = perf_counter() - sanadset_start

    print("Annotating hadith-json sample...")
    hadith_start = perf_counter()
    hadith_rows = hadith_sample_df.to_dict(orient="records")
    hadith_indexes = hadith_sample_df.index.tolist()
    for pos, row in enumerate(
        tqdm(hadith_rows, total=len(hadith_rows), desc="Hadith-json", unit="sent")
    ):
        row_idx = hadith_indexes[pos]
        raw_text = row.get(hadith_text_col)
        if pd.isna(raw_text):
            continue

        normalize_start = perf_counter()
        normalized_text = normalizer.normalize(str(raw_text))
        timing["hadith_normalize_s"] += perf_counter() - normalize_start

        annotate_start = perf_counter()
        annotations = annotator.annotate_from_raw(normalized_text, is_normalized=True)
        timing["hadith_annotate_s"] += perf_counter() - annotate_start
        tokens, labels = split_annotations(annotations)

        if count_entities(labels) == 0:
            rejected_zero_entities += 1
            continue
        if len(tokens) < 5 or len(tokens) > 128:
            rejected_token_len += 1
            continue

        raw_id = row.get(hadith_id_col) if hadith_id_col else row_idx
        book_name = resolve_book_name(row.get(hadith_book_col)) if hadith_book_col else "unknown"

        records.append(
            {
                "id": f"hadith_json_{raw_id}",
                "tokens": tokens,
                "ner_tags": labels,
                "source": "hadith_json",
                "book": book_name,
            }
        )
    timing["hadith_total_s"] = perf_counter() - hadith_start

    if not records:
        raise RuntimeError("No silver records remained after filtering.")

    rng = random.Random(args.seed)
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)

    quality_k = min(args.quality_sample_size, len(shuffled_records))
    quality_sample_records = rng.sample(shuffled_records, k=quality_k) if quality_k > 0 else []
    quality_payload = [to_quality_item(record) for record in quality_sample_records]

    total = len(shuffled_records)
    train_end = int(total * 0.8)
    dev_end = train_end + int(total * 0.1)

    train_records = shuffled_records[:train_end]
    dev_records = shuffled_records[train_end:dev_end]
    held_out_records = shuffled_records[dev_end:]

    from_source = Counter(record["source"] for record in shuffled_records)
    entity_counts = collect_entity_counts(shuffled_records)
    label_distribution = collect_label_distribution(shuffled_records)
    total_entities = sum(entity_counts.values())
    pipeline_elapsed = perf_counter() - pipeline_start

    processed_sentence_count = len(sanadset_sample_df) + len(hadith_sample_df)
    timing["pipeline_total_s"] = pipeline_elapsed
    timing["input_sentences"] = float(processed_sentence_count)
    timing["input_sentences_per_sec"] = (
        float(processed_sentence_count / pipeline_elapsed) if pipeline_elapsed > 0 else 0.0
    )
    timing = {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in timing.items()
    }

    gazetteer_profile = annotator.gazetteer.get_profile_stats()
    gazetteer_share = (
        (gazetteer_profile["total_s"] / timing["pipeline_total_s"]) * 100
        if timing["pipeline_total_s"] > 0
        else 0.0
    )
    gazetteer_profile["pipeline_share_pct"] = round(gazetteer_share, 3)

    stats_payload = {
        "total_sentences": total,
        "from_sanadset": int(from_source.get("sanadset", 0)),
        "from_hadith_json": int(from_source.get("hadith_json", 0)),
        "from_other": int(total - from_source.get("sanadset", 0) - from_source.get("hadith_json", 0)),
        "entity_counts": entity_counts,
        "avg_entities_per_sentence": round(total_entities / total, 6) if total else 0.0,
        "label_distribution": label_distribution,
        "split_sizes": {
            "train": len(train_records),
            "dev": len(dev_records),
            "test_held_out": len(held_out_records),
        },
        "filters": {
            "rejected_zero_entities": rejected_zero_entities,
            "rejected_token_length": rejected_token_len,
            "min_tokens": 5,
            "max_tokens": 128,
        },
        "sampling": {
            "sanadset_loaded_rows": int(len(sanadset_df)),
            "sanadset_sampled_rows": int(len(sanadset_sample_df)),
            "hadith_loaded_rows": int(len(hadith_df)),
            "hadith_sampled_rows": int(len(hadith_sample_df)),
        },
        "timing_seconds": timing,
        "gazetteer_profile": gazetteer_profile,
    }

    silver_train_path = args.output_dir / "silver_train.json"
    stats_path = args.output_dir / "silver_stats.json"
    quality_path = args.output_dir / "quality_sample.json"
    train_path = args.output_dir / "train.json"
    dev_path = args.output_dir / "dev.json"
    held_out_path = args.output_dir / "test_held_out.json"

    write_json(silver_train_path, shuffled_records)
    write_json(stats_path, stats_payload)
    write_json(quality_path, quality_payload)
    write_json(train_path, train_records)
    write_json(dev_path, dev_records)
    write_json(held_out_path, held_out_records)

    print("\n=== Silver Generation Summary ===")
    print(f"Total kept sentences: {total:,}")
    print(f"From Sanadset: {from_source.get('sanadset', 0):,}")
    print(f"From hadith_json: {from_source.get('hadith_json', 0):,}")
    print(f"From other: {stats_payload['from_other']:,}")
    print(f"Rejected (0 entities): {rejected_zero_entities:,}")
    print(f"Rejected (token length outside 5-128): {rejected_token_len:,}")
    print("Entity counts:")
    for entity_type in EXPECTED_ENTITY_TYPES:
        print(f"  {entity_type}: {entity_counts.get(entity_type, 0):,}")
    print(f"Average entities/sentence: {stats_payload['avg_entities_per_sentence']}")
    print("Split sizes:")
    print(f"  train: {len(train_records):,}")
    print(f"  dev: {len(dev_records):,}")
    print(f"  test_held_out: {len(held_out_records):,}")
    print("Runtime:")
    print(f"  total pipeline seconds: {timing['pipeline_total_s']}")
    print(f"  input sentences: {int(timing['input_sentences']):,}")
    print(f"  input sentences/sec: {timing['input_sentences_per_sec']}")
    print(f"  sanadset annotate seconds: {timing['sanadset_annotate_s']}")
    print(f"  hadith annotate seconds: {timing['hadith_annotate_s']}")
    print("Gazetteer profiling:")
    print(f"  calls: {gazetteer_profile['calls']:,}")
    print(f"  total seconds: {gazetteer_profile['total_s']}")
    print(f"  avg ms/call: {gazetteer_profile['avg_ms_per_call']}")
    print(f"  regex scan seconds: {gazetteer_profile['regex_scan_s']}")
    print(f"  pipeline share (%): {gazetteer_profile['pipeline_share_pct']}")
    if gazetteer_profile["pipeline_share_pct"] >= 50:
        print("  bottleneck: gazetteer matching is dominant")
    else:
        print("  bottleneck: gazetteer matching is not dominant")
    print("\nSaved files:")
    print(f"  {silver_train_path}")
    print(f"  {stats_path}")
    print(f"  {quality_path}")
    print(f"  {train_path}")
    print(f"  {dev_path}")
    print(f"  {held_out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
