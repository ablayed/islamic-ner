"""Silver annotation utilities for weakly supervised Arabic NER."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from src.preprocessing.gazetteers import GazetteerMatcher
from src.preprocessing.normalize import ArabicNormalizer

_TOKEN_RE = re.compile(r"\S+")
_XML_TAG_RE = re.compile(r"</?\s*([A-Za-z0-9_:-]+)\s*>")
_ARABIC_LETTER_RE = re.compile(r"[\u0621-\u064A]")


class SilverAnnotator:
    """
    Generates silver (automatic) NER annotations by combining:
    1. Sanadset <NAR> tag extraction (for SCHOLAR)
    2. Pattern-based rules (for SCHOLAR and structure)
    3. Gazetteer matching (for BOOK, CONCEPT, PLACE)
    4. Structural patterns (for HADITH_REF)
    """

    def __init__(self, gazetteer_dir: str = "data/gazetteers/") -> None:
        # Load GazetteerMatcher from Phase 1
        # Load pattern rules
        self.normalizer = ArabicNormalizer()
        self.gazetteer = GazetteerMatcher(gazetteer_dir=gazetteer_dir)

        self._isnad_triggers = {"حدثنا", "اخبرنا", "عن", "قال", "رواه", "سمعت"}
        self._isnad_stop = {
            "عن",
            "ان",
            "قال",
            "اخبرنا",
            "حدثنا",
            "سمعت",
            "رواه",
            "ثم",
            "في",
            "الى",
            "على",
        }
        self._entity_stop = {"ان", "قال", "عن", "حدثنا", "اخبرنا", "رواه", "سمعت"}
        self._non_name_starters = {
            "هذا",
            "هذه",
            "ذلك",
            "ثم",
            "في",
            "على",
            "الى",
            "كتاب",
            "باب",
            "حديث",
            "رقم",
            "انه",
            "انها",
            "الربا",
        }
        self._name_prefixes = {"ابو", "ابي", "بن", "ابن", "عبد"}
        self._book_context_keywords = {"صحيح", "سنن", "مسند", "موطا"}
        self._punctuation_chars = set(".,،؛;:!?؟()[]{}\"'")

        self._hadith_ref_number_patterns = [
            re.compile(
                r"\u062d\u062f\u064a\u062b\s+\u0631\u0642\u0645\s+[0-9\u0660-\u0669]+"
            ),
            re.compile(r"\u0631\u0642\u0645\s+[0-9\u0660-\u0669]+"),
        ]

    def annotate_from_sanadset(self, tagged_text: str) -> List[Tuple[str, str]]:
        """
        Takes a Sanadset-format text with <NAR>, <SANAD>, <MATN> tags.
        Returns list of (token, BIO_label) pairs.

        Steps:
        1. Parse <NAR>...</NAR> spans  mark as SCHOLAR
        2. Remove all XML-style tags
        3. Tokenize (split on whitespace for Arabic)
        4. Assign BIO labels based on NAR span positions
        5. Run gazetteer matching on the full text for BOOK, CONCEPT, PLACE
        6. Run HADITH_REF pattern matching
        7. Merge labels (NAR tags take priority, then patterns, then gazetteer)
        """
        text, nar_entities = self._strip_tags_and_extract_nar(tagged_text)
        token_infos = self._tokenize_with_spans(text)
        token_list = [token_info["token"] for token_info in token_infos]

        pattern_entities = self._apply_isnad_patterns(text)
        pattern_entities.extend(self._apply_hadith_ref_patterns(text))
        pattern_entities = self._apply_book_context_rules(text, pattern_entities)

        gazetteer_entities = self._gazetteer_entities(
            text, allowed_types={"BOOK", "CONCEPT", "PLACE"}
        )

        nar_token_entities = self._char_entities_to_token_entities(
            token_infos, nar_entities
        )
        pattern_token_entities = self._char_entities_to_token_entities(
            token_infos, pattern_entities
        )
        gazetteer_token_entities = self._char_entities_to_token_entities(
            token_infos, gazetteer_entities
        )

        return self._merge_labels(
            token_list,
            nar_token_entities,
            pattern_token_entities,
            gazetteer_token_entities,
        )

    def annotate_from_raw(
        self, raw_text: str, *, is_normalized: bool = False
    ) -> List[Tuple[str, str]]:
        """
        Takes raw Arabic text (no tags)  used for hadith-json data.
        Returns list of (token, BIO_label) pairs.

        Steps:
        1. Normalize text using ArabicNormalizer
        2. Apply isnad pattern rules to find SCHOLAR candidates
        3. Run gazetteer matching for all entity types
        4. Apply HADITH_REF patterns
        5. Assign BIO labels
        """
        text = raw_text if is_normalized else self.normalizer.normalize(raw_text)
        token_infos = self._tokenize_with_spans(text)
        token_list = [token_info["token"] for token_info in token_infos]

        pattern_entities = self._apply_isnad_patterns(text)
        pattern_entities.extend(self._apply_hadith_ref_patterns(text))
        pattern_entities = self._apply_book_context_rules(text, pattern_entities)

        gazetteer_entities = self._gazetteer_entities(
            text,
            allowed_types={"SCHOLAR", "BOOK", "CONCEPT", "PLACE"},
        )
        gazetteer_entities = self._apply_book_context_rules(text, gazetteer_entities)

        pattern_token_entities = self._char_entities_to_token_entities(
            token_infos, pattern_entities
        )
        gazetteer_token_entities = self._char_entities_to_token_entities(
            token_infos, gazetteer_entities
        )

        return self._merge_labels(
            token_list, pattern_token_entities, gazetteer_token_entities
        )

    def _apply_isnad_patterns(self, text: str) -> List[Dict]:
        """
        Pattern rules for SCHOLAR extraction from raw text.

        Patterns to implement (in Arabic):
        - "حدثنا [X]"  X is SCHOLAR (narrated to us [X])
        - "أخبرنا [X]"  X is SCHOLAR (informed us [X])
        - "عن [X]"  X is SCHOLAR (from [X])  be careful, عن is very common
        - "قال [X]"  X is SCHOLAR (said [X])  only when X starts with proper name
        - "رواه [X]"  X is SCHOLAR or BOOK (narrated by [X])
        - "سمعت [X]"  X is SCHOLAR (I heard [X])

        For each pattern:
        - Match the trigger word
        - Extract the following 1-5 tokens as potential entity
        - Stop at the next trigger word, punctuation, or connector (عن أن قال)
        - Validate against gazetteer if possible

        Return list of {"text": str, "start": int, "end": int, "type": "SCHOLAR"}
        """
        token_infos = self._tokenize_with_spans(text)
        if not token_infos:
            return []

        normalized_tokens = [
            self.normalizer.normalize(token_info["token"]) for token_info in token_infos
        ]
        entities: List[Dict] = []

        for idx, trigger_norm in enumerate(normalized_tokens):
            if trigger_norm not in self._isnad_triggers:
                continue

            candidate_indexes: List[int] = []
            max_end = min(len(token_infos), idx + 6)

            for j in range(idx + 1, max_end):
                token_text = token_infos[j]["token"]
                token_norm = normalized_tokens[j]

                if not candidate_indexes and self._is_hard_boundary(
                    token_text, token_norm
                ):
                    break
                if candidate_indexes and (
                    token_norm in self._entity_stop
                    or self._is_pure_punctuation(token_text)
                ):
                    break

                candidate_indexes.append(j)
                if self._has_terminal_punctuation(token_text):
                    break

            if not candidate_indexes:
                continue

            start = token_infos[candidate_indexes[0]]["start"]
            end = token_infos[candidate_indexes[-1]]["end"]
            end = self._trim_right_punctuation(text, start, end)
            if end <= start:
                continue

            candidate_text = text[start:end]
            entity_type = self._classify_isnad_candidate(trigger_norm, candidate_text)
            if not entity_type:
                continue

            entities.append(
                {
                    "text": candidate_text,
                    "start": start,
                    "end": end,
                    "type": entity_type,
                }
            )

        return self._dedupe_entities(entities)

    def _apply_hadith_ref_patterns(self, text: str) -> List[Dict]:
        """
        Patterns for HADITH_REF extraction:
        - "كتاب [X]"  HADITH_REF (when used as chapter reference)
        - "باب [X]"  HADITH_REF (chapter/section)
        - "حديث رقم [N]"  HADITH_REF (hadith number)
        - "رقم [N]"  HADITH_REF (number reference)
        """
        token_infos = self._tokenize_with_spans(text)
        normalized_tokens = [
            self.normalizer.normalize(token_info["token"]) for token_info in token_infos
        ]
        entities: List[Dict] = []

        for idx, token_norm in enumerate(normalized_tokens):
            if token_norm not in {"كتاب", "باب"}:
                continue

            candidate_indexes = [idx]
            max_end = min(len(token_infos), idx + 6)
            for j in range(idx + 1, max_end):
                token_text = token_infos[j]["token"]
                next_norm = normalized_tokens[j]

                if self._is_pure_punctuation(token_text):
                    break
                if next_norm in self._entity_stop or next_norm in {
                    "كتاب",
                    "باب",
                    "حديث",
                    "رقم",
                }:
                    break

                candidate_indexes.append(j)
                if self._has_terminal_punctuation(token_text):
                    break

            if len(candidate_indexes) < 2:
                continue

            start = token_infos[candidate_indexes[0]]["start"]
            end = token_infos[candidate_indexes[-1]]["end"]
            end = self._trim_right_punctuation(text, start, end)
            if end <= start:
                continue

            entities.append(
                {
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "type": "HADITH_REF",
                }
            )

        for compiled_pattern in self._hadith_ref_number_patterns:
            for match in compiled_pattern.finditer(text):
                entities.append(
                    {
                        "text": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                        "type": "HADITH_REF",
                    }
                )

        return self._dedupe_entities(entities)

    def _apply_book_context_rules(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        Disambiguate SCHOLAR vs BOOK based on context.

        Rules:
        - If "صحيح" or "سنن" or "مسند" or "موطأ" precedes a SCHOLAR name
          that is also a known book author  relabel as BOOK
        - If "في" (in) + SCHOLAR name  likely BOOK context
        - If "رواه" + SCHOLAR name  could be either, keep as SCHOLAR
          (the book attribution is indirect)

        This is the disambiguation logic you identified in Phase 1.
        """
        if not entities:
            return []

        adjusted: List[Dict] = []
        for entity in entities:
            candidate = dict(entity)
            if candidate.get("type") != "SCHOLAR":
                adjusted.append(candidate)
                continue

            start = int(candidate.get("start", 0))
            prefix_text = text[max(0, start - 32) : start]
            prefix_norm_tokens = self.normalizer.normalize(prefix_text).split()

            if not prefix_norm_tokens:
                adjusted.append(candidate)
                continue

            last_token = prefix_norm_tokens[-1]
            prev_token = prefix_norm_tokens[-2] if len(prefix_norm_tokens) > 1 else ""

            if last_token == "رواه":
                adjusted.append(candidate)
                continue

            lookup_type = self._lookup_entity_type(candidate.get("text", ""))
            known_author = lookup_type in {"SCHOLAR", "BOOK"}

            preceded_by_book_keyword = last_token in self._book_context_keywords or (
                prev_token == "في" and last_token in self._book_context_keywords
            )
            preceded_by_fi = last_token == "في"

            if known_author and (preceded_by_book_keyword or preceded_by_fi):
                candidate["type"] = "BOOK"

            adjusted.append(candidate)

        return adjusted

    def _merge_labels(
        self,
        token_list: List[str],
        *label_sources: List[Dict],
    ) -> List[Tuple[str, str]]:
        """
        Merge multiple label sources with priority:
        1. Sanadset NAR tags (highest)
        2. Pattern rules
        3. Gazetteer matches

        When two sources disagree on the same span, higher priority wins.
        When two sources label different (non-overlapping) spans, keep both.
        """
        labels = ["O"] * len(token_list)
        occupied = [False] * len(token_list)

        for source in label_sources:
            if not source:
                continue

            sorted_source = sorted(
                source,
                key=lambda item: (
                    item.get("start_token", 0),
                    -(item.get("end_token", 0) - item.get("start_token", 0)),
                ),
            )
            for entity in sorted_source:
                start_token = entity.get("start_token")
                end_token = entity.get("end_token")
                entity_type = entity.get("type") or entity.get("entity_type")

                if start_token is None or end_token is None or not entity_type:
                    continue
                if start_token < 0 or end_token <= start_token:
                    continue
                if start_token >= len(token_list):
                    continue

                end_token = min(end_token, len(token_list))
                if any(occupied[start_token:end_token]):
                    continue

                labels[start_token] = f"B-{entity_type}"
                for idx in range(start_token + 1, end_token):
                    labels[idx] = f"I-{entity_type}"
                for idx in range(start_token, end_token):
                    occupied[idx] = True

        return list(zip(token_list, labels))

    def to_bio_format(self, annotations: List[Tuple[str, str]]) -> str:
        """Convert to CoNLL-style BIO format: token\tlabel per line"""
        return "\n".join(f"{token}\t{label}" for token, label in annotations)

    def to_json_format(
        self,
        annotations: List[Tuple[str, str]],
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Convert to JSON format with metadata for HuggingFace datasets"""
        return {
            "tokens": [token for token, _ in annotations],
            "ner_tags": [label for _, label in annotations],
            "metadata": metadata or {},
        }

    def _tokenize_with_spans(self, text: str) -> List[Dict]:
        token_infos: List[Dict] = []
        for match in _TOKEN_RE.finditer(text):
            token_infos.append(
                {
                    "token": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
        return token_infos

    def _strip_tags_and_extract_nar(self, tagged_text: str) -> Tuple[str, List[Dict]]:
        clean_parts: List[str] = []
        nar_stack: List[int] = []
        nar_spans: List[Dict] = []

        cursor = 0
        clean_length = 0

        def add_separator_if_needed() -> None:
            nonlocal clean_length
            if not clean_parts:
                return
            if clean_parts[-1] and clean_parts[-1][-1].isspace():
                return
            clean_parts.append(" ")
            clean_length += 1

        for match in _XML_TAG_RE.finditer(tagged_text):
            plain_text = tagged_text[cursor : match.start()]
            if plain_text:
                clean_parts.append(plain_text)
                clean_length += len(plain_text)

            tag_name = match.group(1).upper()
            is_closing_tag = tagged_text[match.start() : match.start() + 2] == "</"

            if tag_name == "NAR":
                if is_closing_tag:
                    if nar_stack:
                        start = nar_stack.pop()
                        if clean_length > start:
                            nar_spans.append(
                                {"start": start, "end": clean_length, "type": "SCHOLAR"}
                            )
                else:
                    nar_stack.append(clean_length)
            else:
                # Keep SANAD/MATN and other tag boundaries from collapsing adjacent tokens.
                add_separator_if_needed()

            cursor = match.end()

        tail = tagged_text[cursor:]
        if tail:
            clean_parts.append(tail)

        clean_text = "".join(clean_parts)
        for span in nar_spans:
            span["text"] = clean_text[span["start"] : span["end"]]

        return clean_text, self._dedupe_entities(nar_spans)

    def _gazetteer_entities(
        self, text: str, allowed_types: Optional[set] = None
    ) -> List[Dict]:
        entities: List[Dict] = []
        for match in self.gazetteer.match(text):
            entity_type = match.get("entity_type")
            if allowed_types is not None and entity_type not in allowed_types:
                continue
            entities.append(
                {
                    "text": match["text"],
                    "start": match["start"],
                    "end": match["end"],
                    "type": entity_type,
                }
            )
        return self._dedupe_entities(entities)

    def _char_entities_to_token_entities(
        self, token_infos: List[Dict], entities: List[Dict]
    ) -> List[Dict]:
        if not token_infos or not entities:
            return []

        converted: List[Dict] = []
        for entity in entities:
            start = int(entity.get("start", -1))
            end = int(entity.get("end", -1))
            entity_type = entity.get("type") or entity.get("entity_type")
            if start < 0 or end <= start or not entity_type:
                continue

            overlapping_indexes = [
                idx
                for idx, token_info in enumerate(token_infos)
                if token_info["start"] < end and token_info["end"] > start
            ]
            if not overlapping_indexes:
                continue

            start_token = overlapping_indexes[0]
            end_token = overlapping_indexes[-1] + 1
            converted.append(
                {
                    "text": entity.get("text", ""),
                    "type": entity_type,
                    "start_token": start_token,
                    "end_token": end_token,
                }
            )

        return self._dedupe_entities(converted, token_based=True)

    def _dedupe_entities(
        self, entities: List[Dict], token_based: bool = False
    ) -> List[Dict]:
        seen = set()
        deduped: List[Dict] = []
        for entity in entities:
            if token_based:
                key = (
                    entity.get("start_token"),
                    entity.get("end_token"),
                    entity.get("type") or entity.get("entity_type"),
                )
            else:
                key = (
                    entity.get("start"),
                    entity.get("end"),
                    entity.get("type") or entity.get("entity_type"),
                )

            if key in seen:
                continue
            seen.add(key)
            deduped.append(entity)
        return deduped

    def _lookup_entity_type(self, text: str) -> Optional[str]:
        normalized = self.normalizer.normalize(text)
        lookup = self.gazetteer.lookup.get(normalized)
        if not lookup:
            return None
        return lookup[1]

    def _classify_isnad_candidate(
        self, trigger_norm: str, candidate_text: str
    ) -> Optional[str]:
        lookup_type = self._lookup_entity_type(candidate_text)
        looks_like_name = self._is_probable_name(candidate_text)

        if trigger_norm == "رواه":
            if lookup_type == "BOOK":
                return "BOOK"
            if lookup_type == "SCHOLAR" or looks_like_name:
                return "SCHOLAR"
            return None

        if trigger_norm == "قال" and not self._starts_with_proper_name(candidate_text):
            return None

        if trigger_norm == "عن":
            if lookup_type == "SCHOLAR":
                return "SCHOLAR"
            if lookup_type == "BOOK":
                return None

            candidate_tokens = self.normalizer.normalize(candidate_text).split()
            if len(candidate_tokens) >= 2:
                return "SCHOLAR"
            if candidate_tokens and candidate_tokens[0] in self._name_prefixes:
                return "SCHOLAR"
            return None

        if lookup_type == "BOOK":
            return None
        if lookup_type == "SCHOLAR" or looks_like_name:
            return "SCHOLAR"
        return None

    def _is_probable_name(self, text: str) -> bool:
        normalized = self.normalizer.normalize(text)
        tokens = normalized.split()
        if not tokens or len(tokens) > 5:
            return False

        if any(token in self._entity_stop for token in tokens):
            return False
        if any(any(char.isdigit() for char in token) for token in tokens):
            return False

        first = tokens[0]
        if first in self._non_name_starters:
            return False
        if len(first) < 2:
            return False

        return bool(_ARABIC_LETTER_RE.search(first))

    def _starts_with_proper_name(self, text: str) -> bool:
        normalized = self.normalizer.normalize(text)
        tokens = normalized.split()
        if not tokens:
            return False
        first = tokens[0]
        if first in self._non_name_starters:
            return False
        if self._lookup_entity_type(first) == "SCHOLAR":
            return True
        return self._is_probable_name(first)

    def _is_hard_boundary(self, token_text: str, token_norm: str) -> bool:
        return token_norm in self._isnad_stop or self._is_pure_punctuation(token_text)

    def _is_pure_punctuation(self, token_text: str) -> bool:
        return bool(token_text) and all(
            char in self._punctuation_chars for char in token_text
        )

    def _has_terminal_punctuation(self, token_text: str) -> bool:
        return bool(token_text) and token_text[-1] in self._punctuation_chars

    def _trim_right_punctuation(self, text: str, start: int, end: int) -> int:
        while end > start and text[end - 1] in self._punctuation_chars:
            end -= 1
        return end
