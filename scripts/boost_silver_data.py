"""Boost silver NER data with updated gazetteers and book templates.

Pipeline:
1. Load existing train/dev/test silver splits.
2. Re-annotate hadith_json records in train/dev with updated gazetteers.
3. Enrich sanadset records in train/dev with additional PLACE/HADITH_REF spans.
4. Annotate sentences from data/silver/book_boost_raw.txt and append to train.
5. Keep test_held_out unchanged.
6. Save updated splits and data/silver/silver_stats_boosted.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ner import SilverAnnotator  # noqa: E402

ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "cp1256", "windows-1256"]
EXPECTED_ENTITY_TYPES = ["SCHOLAR", "BOOK", "CONCEPT", "PLACE", "HADITH_REF"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Boost silver NER data with updated gazetteers."
    )
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "silver",
        help="Directory containing train/dev/test_held_out json files.",
    )
    parser.add_argument(
        "--gazetteer-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "gazetteers",
        help="Directory containing gazetteer files.",
    )
    parser.add_argument(
        "--book-boost-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "silver" / "book_boost_raw.txt",
        help="Text file with one raw sentence per line for BOOK boosting.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=None,
        help="Optional override path for train split.",
    )
    parser.add_argument(
        "--dev-path",
        type=Path,
        default=None,
        help="Optional override path for dev split.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=None,
        help="Optional override path for held-out test split.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional override path for boosted stats output json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reserved for deterministic behavior.",
    )
    parser.add_argument(
        "--fail-on-target-miss",
        action="store_true",
        help="Exit non-zero when target minimums are not met.",
    )
    return parser.parse_args()


def read_json(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_lines_with_fallback(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Boost text file not found: {path}")

    last_error: Exception | None = None
    for encoding in ENCODING_CANDIDATES:
        try:
            text = path.read_text(encoding=encoding)
            return [line.strip() for line in text.splitlines() if line.strip()]
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed reading {path} with encodings {ENCODING_CANDIDATES}: {last_error}"
    )


def split_annotations(
    annotations: Sequence[Tuple[str, str]],
) -> Tuple[List[str], List[str]]:
    tokens = [token for token, _ in annotations]
    labels = [label for _, label in annotations]
    return tokens, labels


def extract_entity_spans(
    labels: Sequence[str], allowed_types: Iterable[str] | None = None
) -> List[Tuple[int, int, str]]:
    allowed = set(allowed_types) if allowed_types is not None else None
    spans: List[Tuple[int, int, str]] = []
    idx = 0

    while idx < len(labels):
        label = labels[idx]
        if label == "O" or "-" not in label:
            idx += 1
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix not in {"B", "I"}:
            idx += 1
            continue

        start = idx
        idx += 1
        while idx < len(labels) and labels[idx] == f"I-{entity_type}":
            idx += 1
        end = idx

        if allowed is not None and entity_type not in allowed:
            continue
        spans.append((start, end, entity_type))

    return spans


def collect_entity_counts(records: Sequence[Dict]) -> Dict[str, int]:
    counts = {entity_type: 0 for entity_type in EXPECTED_ENTITY_TYPES}
    for record in records:
        for label in record.get("ner_tags", []):
            if not isinstance(label, str) or not label.startswith("B-"):
                continue
            entity_type = label[2:]
            counts[entity_type] = counts.get(entity_type, 0) + 1
    return counts


def count_entities_in_record(record: Dict) -> Dict[str, int]:
    counts = {entity_type: 0 for entity_type in EXPECTED_ENTITY_TYPES}
    for label in record.get("ner_tags", []):
        if not isinstance(label, str) or not label.startswith("B-"):
            continue
        entity_type = label[2:]
        counts[entity_type] = counts.get(entity_type, 0) + 1
    return counts


def collect_source_counts(records: Sequence[Dict]) -> Dict[str, int]:
    counter: Counter = Counter()
    for record in records:
        source = str(record.get("source", "unknown"))
        counter[source] += 1
    return dict(sorted(counter.items()))


def collect_label_distribution(records: Sequence[Dict]) -> Dict[str, float]:
    counter: Counter = Counter()
    total = 0
    for record in records:
        labels = record.get("ner_tags", [])
        if not isinstance(labels, list):
            continue
        counter.update(labels)
        total += len(labels)

    if total == 0:
        return {}
    return {
        label: round((count / total) * 100, 3)
        for label, count in sorted(counter.items())
    }


def build_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    keys = set(before) | set(after)
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in sorted(keys)}


def merge_allowed_entities(
    base_labels: Sequence[str],
    candidate_labels: Sequence[str],
    allowed_types: Iterable[str],
) -> Tuple[List[str], Dict[str, int]]:
    if len(base_labels) != len(candidate_labels):
        raise ValueError("Label length mismatch while merging selected entities.")

    merged = list(base_labels)
    added_counter: Counter = Counter()
    spans = extract_entity_spans(candidate_labels, allowed_types=allowed_types)
    spans.sort(key=lambda span: (span[0], -(span[1] - span[0])))

    for start, end, entity_type in spans:
        if start < 0 or end <= start or end > len(merged):
            continue
        if any(label != "O" for label in merged[start:end]):
            continue
        merged[start] = f"B-{entity_type}"
        for idx in range(start + 1, end):
            merged[idx] = f"I-{entity_type}"
        added_counter[entity_type] += 1

    return merged, dict(added_counter)


def reannotate_hadith_record(
    record: Dict, annotator: SilverAnnotator
) -> Tuple[Dict, bool]:
    text = " ".join(record.get("tokens", []))
    annotations = annotator.annotate_from_raw(text, is_normalized=True)
    tokens, labels = split_annotations(annotations)

    updated = dict(record)
    updated["tokens"] = tokens
    updated["ner_tags"] = labels

    length_changed = len(tokens) != len(record.get("tokens", []))
    return updated, length_changed


def enrich_sanadset_record(
    record: Dict, annotator: SilverAnnotator
) -> Tuple[Dict, Dict[str, int], bool]:
    text = " ".join(record.get("tokens", []))
    annotations = annotator.annotate_from_raw(text, is_normalized=True)
    candidate_tokens, candidate_labels = split_annotations(annotations)

    base_tokens = list(record.get("tokens", []))
    base_labels = list(record.get("ner_tags", []))
    if len(candidate_tokens) != len(base_tokens) or len(candidate_labels) != len(
        base_labels
    ):
        return dict(record), {}, True

    merged_labels, added_counter = merge_allowed_entities(
        base_labels,
        candidate_labels,
        allowed_types={"PLACE", "HADITH_REF"},
    )

    updated = dict(record)
    updated["ner_tags"] = merged_labels
    return updated, added_counter, False


def process_split(
    records: Sequence[Dict], annotator: SilverAnnotator
) -> Tuple[List[Dict], Dict[str, int]]:
    updated_records: List[Dict] = []
    stats: Counter = Counter()

    for record in records:
        source = str(record.get("source", ""))
        if source == "hadith_json":
            updated, length_changed = reannotate_hadith_record(record, annotator)
            updated_records.append(updated)
            stats["hadith_json_reannotated"] += 1
            if length_changed:
                stats["hadith_json_token_length_changed"] += 1
            continue

        if source == "sanadset":
            updated, added_counter, had_mismatch = enrich_sanadset_record(
                record, annotator
            )
            updated_records.append(updated)
            stats["sanadset_scanned"] += 1
            if had_mismatch:
                stats["sanadset_token_length_mismatch"] += 1
            stats["sanadset_added_place"] += int(added_counter.get("PLACE", 0))
            stats["sanadset_added_hadith_ref"] += int(
                added_counter.get("HADITH_REF", 0)
            )
            continue

        updated_records.append(dict(record))
        stats["other_records_unchanged"] += 1

    return updated_records, dict(stats)


def build_boost_records(
    lines: Sequence[str], annotator: SilverAnnotator
) -> Tuple[List[Dict], Dict[str, int]]:
    records: List[Dict] = []
    stats: Counter = Counter()

    for idx, line in enumerate(lines, start=1):
        annotations = annotator.annotate_from_raw(line)
        tokens, labels = split_annotations(annotations)
        if not tokens:
            stats["boost_empty_after_annotation"] += 1
            continue

        stats["boost_records_created"] += 1
        spans = extract_entity_spans(labels)
        for _, _, entity_type in spans:
            stats[f"boost_entities_{entity_type.lower()}"] += 1

        records.append(
            {
                "id": f"book_boost_{idx}",
                "tokens": tokens,
                "ner_tags": labels,
                "source": "book_boost",
                "book": "book_boost_templates",
            }
        )

    return records, dict(stats)


def clone_record(record: Dict, new_id: str, new_source: str) -> Dict:
    cloned = dict(record)
    cloned["id"] = new_id
    cloned["source"] = new_source
    cloned["tokens"] = list(record.get("tokens", []))
    cloned["ner_tags"] = list(record.get("ner_tags", []))
    return cloned


def enforce_targets_with_oversampling(
    train_records: List[Dict],
    candidate_pool: Sequence[Dict],
    entity_counts: Dict[str, int],
    targets: Dict[str, int],
    seed: int,
) -> Dict[str, Dict]:
    rng = random.Random(seed)
    oversample_summary: Dict[str, Dict] = {}

    for entity_type, threshold in targets.items():
        before_value = int(entity_counts.get(entity_type, 0))
        if before_value >= threshold:
            oversample_summary[entity_type] = {
                "before": before_value,
                "after": before_value,
                "target_min": threshold,
                "records_added": 0,
                "met": True,
            }
            continue

        candidates: List[Tuple[Dict, int, Dict[str, int]]] = []
        for record in candidate_pool:
            per_record_counts = count_entities_in_record(record)
            entity_yield = int(per_record_counts.get(entity_type, 0))
            if entity_yield > 0:
                candidates.append((record, entity_yield, per_record_counts))

        if not candidates:
            oversample_summary[entity_type] = {
                "before": before_value,
                "after": before_value,
                "target_min": threshold,
                "records_added": 0,
                "met": False,
                "reason": "no_candidate_records",
            }
            continue

        rng.shuffle(candidates)
        candidates.sort(key=lambda item: item[1], reverse=True)

        added_records = 0
        cursor = 0
        while entity_counts.get(entity_type, 0) < threshold:
            record, _, per_record_counts = candidates[cursor % len(candidates)]
            cursor += 1
            added_records += 1
            new_id = f"{record.get('id', 'record')}__oversample_{entity_type.lower()}_{added_records}"
            new_source = f"{record.get('source', 'unknown')}_oversample"
            train_records.append(
                clone_record(record, new_id=new_id, new_source=new_source)
            )

            for key, value in per_record_counts.items():
                entity_counts[key] = int(entity_counts.get(key, 0) + value)

        after_value = int(entity_counts.get(entity_type, 0))
        oversample_summary[entity_type] = {
            "before": before_value,
            "after": after_value,
            "target_min": threshold,
            "records_added": added_records,
            "met": after_value >= threshold,
        }

    return oversample_summary


def main() -> int:
    args = parse_args()
    started = perf_counter()

    silver_dir = args.silver_dir
    train_path = args.train_path or (silver_dir / "train.json")
    dev_path = args.dev_path or (silver_dir / "dev.json")
    test_path = args.test_path or (silver_dir / "test_held_out.json")
    stats_output = args.stats_output or (silver_dir / "silver_stats_boosted.json")

    print(f"Silver dir: {silver_dir}")
    print(f"Gazetteer dir: {args.gazetteer_dir}")
    print(f"Book boost raw: {args.book_boost_file}")
    print(f"Train path: {train_path}")
    print(f"Dev path: {dev_path}")
    print(f"Test path: {test_path}")

    train_records = read_json(train_path)
    dev_records = read_json(dev_path)
    test_records = read_json(test_path)
    test_records_original = json.dumps(test_records, ensure_ascii=False, sort_keys=True)

    before_all_records = [*train_records, *dev_records, *test_records]
    before_counts = collect_entity_counts(before_all_records)

    annotator = SilverAnnotator(gazetteer_dir=str(args.gazetteer_dir))

    updated_train, train_stats = process_split(train_records, annotator)
    updated_dev, dev_stats = process_split(dev_records, annotator)

    boost_lines = read_lines_with_fallback(args.book_boost_file)
    boost_records, boost_stats = build_boost_records(boost_lines, annotator)
    updated_train.extend(boost_records)

    updated_test = list(test_records)
    updated_all_records = [*updated_train, *updated_dev, *updated_test]
    after_counts = collect_entity_counts(updated_all_records)

    targets = {
        "BOOK": 50,
        "PLACE": 250,
        "HADITH_REF": 80,
    }
    oversample_summary = enforce_targets_with_oversampling(
        train_records=updated_train,
        candidate_pool=[*updated_train, *updated_dev],
        entity_counts=after_counts,
        targets=targets,
        seed=args.seed,
    )

    updated_all_records = [*updated_train, *updated_dev, *updated_test]
    after_counts = collect_entity_counts(updated_all_records)
    delta_counts = build_delta(before_counts, after_counts)

    write_json(train_path, updated_train)
    write_json(dev_path, updated_dev)
    write_json(test_path, updated_test)

    test_records_after = json.dumps(updated_test, ensure_ascii=False, sort_keys=True)
    test_unchanged = test_records_original == test_records_after

    target_status = {
        entity_type: {
            "target_min": threshold,
            "after": int(after_counts.get(entity_type, 0)),
            "met": int(after_counts.get(entity_type, 0)) >= threshold,
        }
        for entity_type, threshold in targets.items()
    }

    elapsed = perf_counter() - started
    stats_payload = {
        "summary": {
            "runtime_seconds": round(elapsed, 6),
            "before_total_sentences": len(before_all_records),
            "after_total_sentences": len(updated_all_records),
            "test_held_out_unchanged": test_unchanged,
        },
        "split_sizes": {
            "before": {
                "train": len(train_records),
                "dev": len(dev_records),
                "test_held_out": len(test_records),
            },
            "after": {
                "train": len(updated_train),
                "dev": len(updated_dev),
                "test_held_out": len(updated_test),
            },
        },
        "sources": {
            "before": collect_source_counts(before_all_records),
            "after": collect_source_counts(updated_all_records),
        },
        "entity_counts_before": before_counts,
        "entity_counts_after": after_counts,
        "entity_count_delta": delta_counts,
        "label_distribution_before": collect_label_distribution(before_all_records),
        "label_distribution_after": collect_label_distribution(updated_all_records),
        "processing": {
            "train": train_stats,
            "dev": dev_stats,
            "boost": {
                "input_lines": len(boost_lines),
                **boost_stats,
            },
            "oversampling": oversample_summary,
        },
        "targets": target_status,
    }
    write_json(stats_output, stats_payload)

    print("\nBefore/After entity counts:")
    for entity_type in EXPECTED_ENTITY_TYPES:
        before_value = before_counts.get(entity_type, 0)
        after_value = after_counts.get(entity_type, 0)
        print(f"- {entity_type}: was {before_value}, now {after_value}")

    print("\nBoost processing:")
    print(
        f"- hadith_json reannotated (train+dev): {train_stats.get('hadith_json_reannotated', 0) + dev_stats.get('hadith_json_reannotated', 0)}"
    )
    print(
        f"- sanadset scanned (train+dev): {train_stats.get('sanadset_scanned', 0) + dev_stats.get('sanadset_scanned', 0)}"
    )
    print(
        f"- sanadset PLACE additions: {train_stats.get('sanadset_added_place', 0) + dev_stats.get('sanadset_added_place', 0)}"
    )
    print(
        f"- sanadset HADITH_REF additions: {train_stats.get('sanadset_added_hadith_ref', 0) + dev_stats.get('sanadset_added_hadith_ref', 0)}"
    )
    print(
        f"- boost sentences ingested: {boost_stats.get('boost_records_created', 0)} / {len(boost_lines)}"
    )
    print(
        f"- oversampled BOOK records: {oversample_summary.get('BOOK', {}).get('records_added', 0)}"
    )
    print(
        f"- oversampled HADITH_REF records: {oversample_summary.get('HADITH_REF', {}).get('records_added', 0)}"
    )
    print(f"- kept held-out test unchanged: {test_unchanged}")
    print(f"- runtime seconds: {round(elapsed, 3)}")

    print("\nTarget checks:")
    all_targets_met = True
    for entity_type, result in target_status.items():
        met = bool(result["met"])
        all_targets_met = all_targets_met and met
        status = "MET" if met else "MISS"
        print(
            f"- {entity_type}: {status} "
            f"(target >= {result['target_min']}, now {result['after']})"
        )

    print("\nSaved:")
    print(f"- {train_path}")
    print(f"- {dev_path}")
    print(f"- {test_path}")
    print(f"- {stats_output}")

    if args.fail_on_target_miss and not all_targets_met:
        print(
            "\nFailing because --fail-on-target-miss is enabled and some targets were missed."
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
