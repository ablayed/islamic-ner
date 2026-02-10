"""Gazetteer loading and matching utilities for Arabic NER bootstrapping."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from src.preprocessing.normalize import ArabicNormalizer

_TASHKEEL_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")
_ALIF_RE = re.compile(r"[\u0622\u0623\u0625]")


class GazetteerMatcher:
    """Load gazetteers and provide longest-match-first entity matching."""

    def __init__(self, gazetteer_dir: str = "data/gazetteers/") -> None:
        self.normalizer = ArabicNormalizer()
        self.gazetteer_dir = self._resolve_gazetteer_dir(gazetteer_dir)

        self.lookup: Dict[str, Tuple[str, str]] = {}
        self._patterns: List[Tuple[str, str, str]] = []

        self._load_all_gazetteers()
        self._patterns = sorted(
            (
                (normalized_name, canonical_name, entity_type)
                for normalized_name, (canonical_name, entity_type) in self.lookup.items()
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )

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

    def _find_occurrences(self, text: str, pattern: str) -> List[Tuple[int, int]]:
        matches: List[Tuple[int, int]] = []
        start = 0

        while True:
            found = text.find(pattern, start)
            if found == -1:
                break

            end = found + len(pattern)
            if self._is_word_boundary(text, found, end):
                matches.append((found, end))
            start = found + 1

        return matches

    def _is_word_boundary(self, text: str, start: int, end: int) -> bool:
        if start > 0 and text[start - 1].isalnum():
            return False
        if end < len(text) and text[end].isalnum():
            return False
        return True

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

        normalized_text, alignment = self._normalize_text_with_alignment(text)
        if not normalized_text:
            return []

        candidates = []
        for pattern, canonical_name, entity_type in self._patterns:
            for start_norm, end_norm in self._find_occurrences(normalized_text, pattern):
                candidates.append(
                    {
                        "start_norm": start_norm,
                        "end_norm": end_norm,
                        "entity_type": entity_type,
                        "canonical_name": canonical_name,
                    }
                )

        if not candidates:
            return []

        occupied = [False] * len(normalized_text)
        selected = []

        candidates.sort(
            key=lambda candidate: (
                -(candidate["end_norm"] - candidate["start_norm"]),
                candidate["start_norm"],
            )
        )

        for candidate in candidates:
            start_norm = candidate["start_norm"]
            end_norm = candidate["end_norm"]

            if any(occupied[start_norm:end_norm]):
                continue

            for index in range(start_norm, end_norm):
                occupied[index] = True
            selected.append(candidate)

        selected.sort(key=lambda candidate: candidate["start_norm"])

        matches: List[Dict] = []
        for candidate in selected:
            start_norm = candidate["start_norm"]
            end_norm = candidate["end_norm"]

            original_start = alignment[start_norm]
            original_end = alignment[end_norm - 1] + 1
            original_end = self._extend_end_for_removed_marks(text, original_end)

            matched_span = text[original_start:original_end]
            matches.append(
                {
                    "text": matched_span,
                    "start": original_start,
                    "end": original_end,
                    "entity_type": candidate["entity_type"],
                    "canonical_name": candidate["canonical_name"],
                }
            )

        return matches
