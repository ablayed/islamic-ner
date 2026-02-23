"""Build Neo4j graph from trained NER model over Bukhari hadiths.

Pipeline per hadith:
1) normalize Arabic text
2) run NER inference
3) extract relations
4) insert entities + relations into Neo4j

Run example:
    python scripts/build_graph.py --limit 500 --clear-graph
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.builder import KnowledgeGraphBuilder  # noqa: E402
from src.preprocessing.normalize import ArabicNormalizer  # noqa: E402
from src.relations.extract import RelationExtractor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Islamic knowledge graph from Bukhari hadiths."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "islamic_ner_best",
        help="Path to trained NER model directory.",
    )
    parser.add_argument(
        "--bukhari-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "hadith_json" / "bukhari.json",
        help="Path to bukhari.json file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of hadiths to process from bukhari.json.",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j Bolt URI.",
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Neo4j username.",
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default="password",
        help="Neo4j password.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Max subword sequence length for model inference.",
    )
    parser.add_argument(
        "--word-window",
        type=int,
        default=120,
        help="Approximate max words per inference chunk.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Progress print frequency.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "graph_build_log.json",
        help="Where to save processing log JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--clear-graph",
        action="store_true",
        help="Clear graph before inserting new nodes/edges.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run normalization + NER + relation extraction without Neo4j insertion.",
    )
    parser.add_argument(
        "--no-model-fallback",
        action="store_true",
        help="Disable fallback model path resolution if --model-dir does not exist.",
    )
    return parser.parse_args()


def resolve_model_dir(model_dir: Path, allow_fallback: bool) -> Tuple[Path, List[str]]:
    notes: List[str] = []
    if model_dir.exists():
        return model_dir, notes

    if not allow_fallback:
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. "
            "Create models/islamic_ner_best/ or omit --no-model-fallback."
        )

    fallback_candidates = [
        PROJECT_ROOT / "models" / "islamic_ner_standard" / "final_model",
        PROJECT_ROOT / "models" / "islamic_ner_camelbert_ca" / "final_model",
        PROJECT_ROOT / "models" / "islamic_ner_weighted" / "final_model",
    ]
    for candidate in fallback_candidates:
        if candidate.exists():
            notes.append(
                f"Requested model '{model_dir}' not found; using fallback '{candidate}'."
            )
            return candidate, notes

    raise FileNotFoundError(
        f"Model directory not found: {model_dir}, and no fallback final_model path exists."
    )


def load_bukhari_hadiths(path: Path, limit: int) -> List[Dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    hadiths = raw.get("hadiths", [])
    chapters = {
        chapter.get("id"): str(chapter.get("arabic", ""))
        for chapter in raw.get("chapters", [])
        if chapter.get("id") is not None
    }
    book_title = str(
        raw.get("metadata", {}).get("arabic", {}).get("title", "صحيح البخاري")
    )

    records: List[Dict] = []
    for hadith in hadiths[:limit]:
        hadith_num = hadith.get("id")
        hadith_text = str(hadith.get("arabic", "") or "").strip()
        if hadith_num is None or not hadith_text:
            continue
        chapter_name = chapters.get(hadith.get("chapterId"), "")
        records.append(
            {
                "id": f"bukhari_{hadith_num}",
                "text": hadith_text,
                "book_ref": book_title,
                "chapter": chapter_name,
                "raw_id": hadith_num,
            }
        )
    return records


class WordLevelNER:
    """Word-level BIO inference wrapper for token-classification models."""

    def __init__(
        self,
        model_dir: Path,
        *,
        max_seq_length: int,
        word_window: int,
        device: str,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            str(model_dir), local_files_only=True
        )
        self.model.eval()

        self.max_seq_length = int(max_seq_length)
        self.word_window = int(word_window)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)

        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def predict_bio(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []

        all_labels: List[str] = []
        cursor = 0
        n_tokens = len(tokens)

        while cursor < n_tokens:
            chunk_end = min(cursor + self.word_window, n_tokens)
            chunk_tokens = tokens[cursor:chunk_end]
            chunk_labels, covered = self._predict_chunk(chunk_tokens)
            if covered <= 0:
                # Safety fallback to avoid infinite loops.
                all_labels.append("O")
                cursor += 1
                continue

            all_labels.extend(chunk_labels[:covered])
            cursor += covered

        if len(all_labels) != n_tokens:
            if len(all_labels) < n_tokens:
                all_labels.extend(["O"] * (n_tokens - len(all_labels)))
            else:
                all_labels = all_labels[:n_tokens]
        return self._repair_bio(all_labels)

    def _predict_chunk(self, chunk_tokens: List[str]) -> Tuple[List[str], int]:
        encoded = self.tokenizer(
            chunk_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        if not hasattr(encoded, "word_ids"):
            raise RuntimeError(
                "Tokenizer must be fast and support word_ids for word-level decoding."
            )

        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        logits = self.model(**model_inputs).logits[0]
        pred_ids = logits.argmax(dim=-1).tolist()

        word_ids = encoded.word_ids(batch_index=0)
        if word_ids is None:
            return [], 0

        covered = max((wid for wid in word_ids if wid is not None), default=-1) + 1
        if covered <= 0:
            return [], 0

        labels = ["O"] * covered
        seen_word_ids = set()
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen_word_ids or word_id >= covered:
                continue
            seen_word_ids.add(word_id)
            raw_label = self.id2label.get(int(pred_ids[token_idx]), "O")
            labels[word_id] = str(raw_label)
        return labels, covered

    @staticmethod
    def _repair_bio(labels: List[str]) -> List[str]:
        repaired: List[str] = []
        prev_type = ""
        prev_is_entity = False

        for label in labels:
            if label == "O" or "-" not in label:
                repaired.append("O")
                prev_type = ""
                prev_is_entity = False
                continue

            prefix, entity_type = label.split("-", 1)
            if prefix not in {"B", "I"} or not entity_type:
                repaired.append("O")
                prev_type = ""
                prev_is_entity = False
                continue

            if prefix == "I" and (not prev_is_entity or prev_type != entity_type):
                repaired_label = f"B-{entity_type}"
            else:
                repaired_label = f"{prefix}-{entity_type}"

            repaired.append(repaired_label)
            prev_type = entity_type
            prev_is_entity = True

        return repaired


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    model_dir, model_notes = resolve_model_dir(
        args.model_dir,
        allow_fallback=not args.no_model_fallback,
    )
    records = load_bukhari_hadiths(args.bukhari_path, args.limit)
    if not records:
        raise RuntimeError(f"No hadith records found in {args.bukhari_path}")

    normalizer = ArabicNormalizer()
    relation_extractor = RelationExtractor()
    tagger = WordLevelNER(
        model_dir=model_dir,
        max_seq_length=args.max_seq_length,
        word_window=args.word_window,
        device=args.device,
    )

    graph_builder = None
    if not args.dry_run:
        graph_builder = KnowledgeGraphBuilder(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
        )
        if args.clear_graph:
            print("Clearing graph ...")
            graph_builder.clear_graph()
        graph_builder.create_constraints()

    total = len(records)
    processed = 0
    skipped = 0
    failed = 0
    total_entities_inserted = 0
    total_relations_inserted = 0
    total_relations_extracted = 0
    errors: List[Dict] = []
    sample_log: List[Dict] = []

    print(
        f"Starting graph build: hadiths={total}, model={model_dir}, "
        f"mode={'dry-run' if args.dry_run else 'insert'}"
    )
    for note in model_notes:
        print(f"NOTE: {note}")

    try:
        for idx, record in enumerate(records, start=1):
            hadith_id = record["id"]
            try:
                normalized_text = normalizer.normalize(record["text"])
                tokens = normalized_text.split()
                if not tokens:
                    skipped += 1
                    continue

                labels = tagger.predict_bio(tokens)
                metadata = {
                    "hadith_id": hadith_id,
                    "book_ref": record["book_ref"],
                    "chapter": record["chapter"],
                }
                extracted_relations = relation_extractor.extract(
                    tokens, labels, metadata=metadata
                )

                if args.dry_run:
                    pipeline_result = {"entities_inserted": 0, "relations_inserted": 0}
                else:
                    pipeline_result = graph_builder.process_hadith(
                        tokens=tokens,
                        labels=labels,
                        hadith_id=hadith_id,
                        metadata=metadata,
                    )

                processed += 1
                total_relations_extracted += len(extracted_relations)
                total_entities_inserted += int(pipeline_result["entities_inserted"])
                total_relations_inserted += int(pipeline_result["relations_inserted"])

                if len(sample_log) < 25:
                    sample_log.append(
                        {
                            "hadith_id": hadith_id,
                            "token_count": len(tokens),
                            "relations_extracted": len(extracted_relations),
                            "entities_inserted": int(
                                pipeline_result["entities_inserted"]
                            ),
                            "relations_inserted": int(
                                pipeline_result["relations_inserted"]
                            ),
                        }
                    )
            except Exception as exc:  # pragma: no cover - runtime guard
                failed += 1
                errors.append({"hadith_id": hadith_id, "error": str(exc)})

            if idx % args.progress_every == 0 or idx == total:
                elapsed = time.perf_counter() - start_time
                rate = idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{idx}/{total}] processed={processed} skipped={skipped} failed={failed} "
                    f"relations={total_relations_inserted} rate={rate:.2f} hadith/s"
                )
    finally:
        if graph_builder is not None:
            graph_builder.close()

    runtime_seconds = time.perf_counter() - start_time

    final_stats = {}
    final_stats_error = ""
    if not args.dry_run:
        try:
            stats_builder = KnowledgeGraphBuilder(
                uri=args.neo4j_uri,
                user=args.neo4j_user,
                password=args.neo4j_password,
            )
            try:
                final_stats = stats_builder.get_stats()
            finally:
                stats_builder.close()
        except Exception as exc:  # pragma: no cover - runtime guard
            final_stats_error = str(exc)

    log_payload = {
        "started_at_unix": int(start_time),
        "runtime_seconds": round(runtime_seconds, 3),
        "config": {
            "model_dir_requested": str(args.model_dir),
            "model_dir_used": str(model_dir),
            "bukhari_path": str(args.bukhari_path),
            "limit": int(args.limit),
            "neo4j_uri": args.neo4j_uri,
            "neo4j_user": args.neo4j_user,
            "max_seq_length": int(args.max_seq_length),
            "word_window": int(args.word_window),
            "device": str(tagger.device),
            "clear_graph": bool(args.clear_graph),
            "dry_run": bool(args.dry_run),
        },
        "model_resolution_notes": model_notes,
        "summary": {
            "hadiths_total": total,
            "hadiths_processed": processed,
            "hadiths_skipped": skipped,
            "hadiths_failed": failed,
            "relations_extracted_total": total_relations_extracted,
            "entities_inserted_total": total_entities_inserted,
            "relations_inserted_total": total_relations_inserted,
            "throughput_hadiths_per_sec": (
                round((total / runtime_seconds), 4) if runtime_seconds > 0 else 0.0
            ),
        },
        "graph_stats": final_stats,
        "graph_stats_error": final_stats_error,
        "sample_records": sample_log,
        "errors": errors[:200],
    }

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text(
        json.dumps(log_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nBuild complete.")
    print(f"Processed: {processed}/{total}, failed={failed}, skipped={skipped}")
    print(f"Runtime: {runtime_seconds:.2f}s")
    if final_stats:
        print(f"Graph stats: {json.dumps(final_stats, ensure_ascii=False)}")
    print(f"Log saved to: {args.log_path}")


if __name__ == "__main__":
    main()
