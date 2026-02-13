"""Gazetteer loading and matching utilities for Arabic NER bootstrapping."""

from __future__ import annotations

from collections import defaultdict
import re
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

from src.preprocessing.normalize import ArabicNormalizer

_TASHKEEL_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")
_ALIF_RE = re.compile(r"[\u0622\u0623\u0625]")


class GazetteerMatcher:
    """Load gazetteers and provide longest-match-first entity matching."""

    def __init__(self, gazetteer_dir: str = "data/gazetteers/") -> None:
        self.normalizer = ArabicNormalizer()
        self.gazetteer_dir = self._resolve_gazetteer_dir(gazetteer_dir)

        self.lookup: Dict[str, Tuple[str, str]] = {}
        self._master_pattern: Optional[re.Pattern[str]] = None
        self._profile: Dict[str, float] = {
            "calls": 0.0,
            "normalize_s": 0.0,
            "regex_scan_s": 0.0,
            "resolve_s": 0.0,
            "total_s": 0.0,
        }

        self._load_all_gazetteers()
        self._compile_master_pattern()

    def _resolve_gazetteer_dir(self, gazetteer_dir: str) -> Path:
        path = Path(gazetteer_dir)
        if path.is_absolute():
            return path

        project_root = Path(__file__).resolve().parents[2]
        return project_root / path

    def _load_all_gazetteers(self) -> None:
        self._load_gazetteer_file("scholars.txt", "SCHOLAR", split_on_pipe=True)
        self._load_gazetteer_file("books.txt", "BOOK", split_on_pipe=True)
        # The spec says one concept per line; splitting by pipe keeps backward compatibility.
        self._load_gazetteer_file("concepts.txt", "CONCEPT", split_on_pipe=True)
        self._load_gazetteer_file("places.txt", "PLACE", split_on_pipe=True)

    def _load_gazetteer_file(
        self,
        file_name: str,
        entity_type: str,
        *,
        split_on_pipe: bool,
    ) -> None:
        file_path = self.gazetteer_dir / file_name
        if not file_path.exists():
            return

        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if split_on_pipe:
                variants = [part.strip() for part in line.split("|") if part.strip()]
            else:
                variants = [line]

            if not variants:
                continue

            canonical_normalized = self.normalizer.normalize(variants[0])
            if not canonical_normalized:
                continue

            for variant in variants:
                normalized_variant = self.normalizer.normalize(variant)
                if not normalized_variant:
                    continue
                self.lookup.setdefault(
                    normalized_variant,
                    (canonical_normalized, entity_type),
                )

    def _compile_master_pattern(self) -> None:
        if not self.lookup:
            self._master_pattern = None
            return

        by_type: Dict[str, List[str]] = defaultdict(list)
        for normalized_name, (_, entity_type) in self.lookup.items():
            by_type[entity_type].append(normalized_name)

        variants: List[str] = []
        for entity_type in sorted(by_type):
            unique_values = sorted(set(by_type[entity_type]), key=len, reverse=True)
            variants.extend(unique_values)

        if not variants:
            self._master_pattern = None
            return

        escaped = [re.escape(value) for value in variants]
        regex_body = "|".join(escaped)
        # Use Unicode-aware word boundaries around Arabic terms and aliases.
        self._master_pattern = re.compile(rf"(?<!\w)(?:{regex_body})(?!\w)")

    def _normalize_text_with_alignment(self, text: str) -> Tuple[str, List[int]]:
        chars: List[str] = []
        original_indexes: List[int] = []

        for index, char in enumerate(text):
            if _TASHKEEL_RE.fullmatch(char):
                continue
            if char == "\u0640":  # Tatweel
                continue

            if _ALIF_RE.fullmatch(char):
                normalized_char = "\u0627"
            elif char == "\u0629":
                normalized_char = "\u0647"
            elif char == "\u0649":
                normalized_char = "\u064A"
            elif char.isspace():
                normalized_char = " "
            else:
                normalized_char = char

            chars.append(normalized_char)
            original_indexes.append(index)

        collapsed_chars: List[str] = []
        collapsed_indexes: List[int] = []
        previous_space = False

        for char, original_index in zip(chars, original_indexes):
            if char == " ":
                if not collapsed_chars or previous_space:
                    previous_space = True
                    continue
                collapsed_chars.append(char)
                collapsed_indexes.append(original_index)
                previous_space = True
                continue

            collapsed_chars.append(char)
            collapsed_indexes.append(original_index)
            previous_space = False

        if collapsed_chars and collapsed_chars[-1] == " ":
            collapsed_chars.pop()
            collapsed_indexes.pop()

        return "".join(collapsed_chars), collapsed_indexes

    def _extend_end_for_removed_marks(self, text: str, end_index: int) -> int:
        while end_index < len(text):
            char = text[end_index]
            if _TASHKEEL_RE.fullmatch(char) or char == "\u0640":
                end_index += 1
                continue
            break
        return end_index

    def match(self, text: str) -> List[Dict]:
        """Return all gazetteer matches found in text."""
        total_start = perf_counter()
        normalize_elapsed = 0.0
        scan_elapsed = 0.0
        resolve_elapsed = 0.0
        matches: List[Dict] = []

        try:
            if self._master_pattern is None:
                return []

            normalize_start = perf_counter()
            normalized_text, alignment = self._normalize_text_with_alignment(text)
            normalize_elapsed = perf_counter() - normalize_start
            if not normalized_text:
                return []

            scan_start = perf_counter()
            raw_matches = list(self._master_pattern.finditer(normalized_text))
            scan_elapsed = perf_counter() - scan_start
            if not raw_matches:
                return []

            resolve_start = perf_counter()
            for raw_match in raw_matches:
                start_norm, end_norm = raw_match.span()
                normalized_span = raw_match.group(0)
                lookup = self.lookup.get(normalized_span)
                if not lookup:
                    continue

                canonical_name, entity_type = lookup
                original_start = alignment[start_norm]
                original_end = alignment[end_norm - 1] + 1
                original_end = self._extend_end_for_removed_marks(text, original_end)

                matched_span = text[original_start:original_end]
                matches.append(
                    {
                        "text": matched_span,
                        "start": original_start,
                        "end": original_end,
                        "entity_type": entity_type,
                        "canonical_name": canonical_name,
                    }
                )
            resolve_elapsed = perf_counter() - resolve_start
            return matches
        finally:
            total_elapsed = perf_counter() - total_start
            self._profile["calls"] += 1.0
            self._profile["normalize_s"] += normalize_elapsed
            self._profile["regex_scan_s"] += scan_elapsed
            self._profile["resolve_s"] += resolve_elapsed
            self._profile["total_s"] += total_elapsed

    def get_profile_stats(self) -> Dict[str, float]:
        calls = int(self._profile["calls"])
        if calls == 0:
            return {
                "calls": 0,
                "total_s": 0.0,
                "normalize_s": 0.0,
                "regex_scan_s": 0.0,
                "resolve_s": 0.0,
                "avg_ms_per_call": 0.0,
            }

        total_s = self._profile["total_s"]
        return {
            "calls": calls,
            "total_s": round(total_s, 6),
            "normalize_s": round(self._profile["normalize_s"], 6),
            "regex_scan_s": round(self._profile["regex_scan_s"], 6),
            "resolve_s": round(self._profile["resolve_s"], 6),
            "avg_ms_per_call": round((total_s / calls) * 1000.0, 6),
        }
