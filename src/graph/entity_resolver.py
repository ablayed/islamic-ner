"""Entity normalization utilities for graph insertion."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

from src.preprocessing.normalize import ArabicNormalizer


class EntityResolver:
    """
    Resolves entity mentions to canonical forms.
    Uses gazetteers for known entities and fuzzy matching for unknown.
    """

    def __init__(self, gazetteer_dir: str = "data/gazetteers/"):
        self.normalizer = ArabicNormalizer()
        # (normalized_variant, entity_type) -> normalized_canonical
        self.canonical_map: Dict[Tuple[str, str], str] = {}
        # normalized_canonical -> {"type": str, "variants": [str, ...]}
        self.entity_metadata: Dict[str, Dict[str, object]] = {}
        self._load_gazetteers(gazetteer_dir)

    def resolve(self, entity_text: str, entity_type: str) -> Dict:
        """
        Resolve one entity mention to a canonical form.
        """
        normalized_text = self.normalizer.normalize(entity_text or "")
        normalized_type = str(entity_type or "").upper()
        key = (normalized_text, normalized_type)

        if key in self.canonical_map:
            canonical = self.canonical_map[key]
            return {
                "canonical_name": canonical,
                "original_text": entity_text,
                "entity_type": normalized_type,
                "confidence": 1.0,
                "match_type": "exact",
            }

        fuzzy = self._fuzzy_match(normalized_text, normalized_type)
        if fuzzy is not None:
            canonical, score = fuzzy
            return {
                "canonical_name": canonical,
                "original_text": entity_text,
                "entity_type": normalized_type,
                "confidence": round(score, 4),
                "match_type": "fuzzy",
            }

        canonical_new = normalized_text
        if canonical_new and canonical_new not in self.entity_metadata:
            self.entity_metadata[canonical_new] = {
                "type": normalized_type,
                "variants": [canonical_new],
            }

        return {
            "canonical_name": canonical_new,
            "original_text": entity_text,
            "entity_type": normalized_type,
            "confidence": 0.5,
            "match_type": "new",
        }

    def _fuzzy_match(
        self,
        text: str,
        entity_type: str,
        threshold: float = 0.8,
    ) -> Tuple[str, float] | None:
        """
        Simple fuzzy matching using SequenceMatcher ratio.
        """
        if not text:
            return None

        candidates: List[Tuple[str, float]] = []
        for canonical_name, metadata in self.entity_metadata.items():
            if str(metadata.get("type")) != entity_type:
                continue

            best_for_canonical = SequenceMatcher(None, text, canonical_name).ratio()
            for variant in metadata.get("variants", []):
                ratio = SequenceMatcher(None, text, str(variant)).ratio()
                if ratio > best_for_canonical:
                    best_for_canonical = ratio
            candidates.append((canonical_name, best_for_canonical))

        if not candidates:
            return None

        best_candidate = max(candidates, key=lambda item: item[1])
        if best_candidate[1] < threshold:
            return None
        return best_candidate

    def _load_gazetteers(self, gazetteer_dir: str) -> None:
        path = self._resolve_gazetteer_dir(gazetteer_dir)
        file_map = {
            "scholars.txt": "SCHOLAR",
            "books.txt": "BOOK",
            "concepts.txt": "CONCEPT",
            "places.txt": "PLACE",
        }

        for file_name, entity_type in file_map.items():
            self._load_gazetteer_file(path / file_name, entity_type)

    def _load_gazetteer_file(self, file_path: Path, entity_type: str) -> None:
        if not file_path.exists():
            return

        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip().lstrip("\ufeff")
            if not line or line.startswith("#"):
                continue

            parts = [
                part.strip().lstrip("\ufeff")
                for part in line.split("|")
                if part.strip()
            ]
            if not parts:
                continue

            canonical_raw = parts[0]
            canonical_norm = self.normalizer.normalize(canonical_raw)
            if not canonical_norm:
                continue

            metadata = self.entity_metadata.setdefault(
                canonical_norm,
                {"type": entity_type, "variants": []},
            )
            metadata["type"] = entity_type

            variant_bucket = metadata["variants"]
            if (
                isinstance(variant_bucket, list)
                and canonical_norm not in variant_bucket
            ):
                variant_bucket.append(canonical_norm)

            for variant in parts:
                variant_norm = self.normalizer.normalize(variant)
                if not variant_norm:
                    continue
                self.canonical_map[(variant_norm, entity_type)] = canonical_norm
                if (
                    isinstance(variant_bucket, list)
                    and variant_norm not in variant_bucket
                ):
                    variant_bucket.append(variant_norm)

    def _resolve_gazetteer_dir(self, gazetteer_dir: str) -> Path:
        path = Path(gazetteer_dir)
        if path.is_absolute():
            return path
        project_root = Path(__file__).resolve().parents[2]
        return project_root / path
