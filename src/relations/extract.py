"""Rule-based relation extraction for Islamic NER output."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from src.preprocessing.normalize import ArabicNormalizer


class RelationExtractor:
    """
    Rule-based relation extraction from NER-annotated text.
    Takes NER output (tokens + BIO labels) and extracts relations.
    """

    def __init__(self, author_book_map: Dict[str, str] | None = None):
        """
        author_book_map: optional dict of {scholar_name: book_title}
        for known author-book pairs.
        """
        self.normalizer = ArabicNormalizer()
        default_pairs = {
            "البخاري": "صحيح البخاري",
            "مسلم": "صحيح مسلم",
            "أبو داود": "سنن أبي داود",
            "الترمذي": "سنن الترمذي",
            "النسائي": "سنن النسائي",
            "ابن ماجه": "سنن ابن ماجه",
            "مالك": "الموطأ",
            "أحمد": "المسند",
            "النووي": "رياض الصالحين",
        }
        raw_pairs = author_book_map or default_pairs
        self.author_book_map = {
            self.normalizer.normalize(author): book
            for author, book in raw_pairs.items()
        }

        self._narration_triggers = {
            self.normalizer.normalize(token) for token in ("حدثنا", "أخبرنا", "سمعت")
        }
        self._chain_connectors = {
            self.normalizer.normalize(token) for token in ("عن", "أن")
        }
        self._book_cues = {
            self.normalizer.normalize(token) for token in ("رواه", "أخرجه", "في")
        }

    def extract(
        self,
        tokens: List[str],
        labels: List[str],
        metadata: Dict | None = None,
    ) -> List[Dict]:
        """
        Main method. Takes tokenized text with BIO labels.
        Returns a list of extracted relation dictionaries.
        """
        entities = self._extract_entity_spans(tokens, labels)
        relations: List[Dict] = []
        relations.extend(self._extract_narration_chains(tokens, entities))
        relations.extend(self._extract_book_relations(tokens, entities, metadata))
        relations.extend(self._extract_concept_mentions(entities, metadata))
        relations.extend(self._extract_authorship(entities))
        return self._dedupe_relations(relations)

    def _extract_entity_spans(self, tokens: List[str], labels: List[str]) -> List[Dict]:
        """Convert BIO labels to entity spans."""
        if len(tokens) != len(labels):
            raise ValueError("tokens and labels must have same length")

        spans: List[Dict] = []
        current_start: Optional[int] = None
        current_type: Optional[str] = None

        for idx, label in enumerate(labels):
            if label == "O":
                if current_type is not None and current_start is not None:
                    spans.append(
                        {
                            "text": " ".join(tokens[current_start:idx]),
                            "type": current_type,
                            "start": current_start,
                            "end": idx,
                        }
                    )
                current_start = None
                current_type = None
                continue

            if "-" not in label:
                # Malformed tag, treat as boundary.
                if current_type is not None and current_start is not None:
                    spans.append(
                        {
                            "text": " ".join(tokens[current_start:idx]),
                            "type": current_type,
                            "start": current_start,
                            "end": idx,
                        }
                    )
                current_start = None
                current_type = None
                continue

            prefix, entity_type = label.split("-", 1)
            if prefix == "B":
                if current_type is not None and current_start is not None:
                    spans.append(
                        {
                            "text": " ".join(tokens[current_start:idx]),
                            "type": current_type,
                            "start": current_start,
                            "end": idx,
                        }
                    )
                current_start = idx
                current_type = entity_type
            elif prefix == "I":
                if current_type != entity_type or current_start is None:
                    # Repair broken BIO by starting a new span.
                    if current_type is not None and current_start is not None:
                        spans.append(
                            {
                                "text": " ".join(tokens[current_start:idx]),
                                "type": current_type,
                                "start": current_start,
                                "end": idx,
                            }
                        )
                    current_start = idx
                    current_type = entity_type
            else:
                if current_type is not None and current_start is not None:
                    spans.append(
                        {
                            "text": " ".join(tokens[current_start:idx]),
                            "type": current_type,
                            "start": current_start,
                            "end": idx,
                        }
                    )
                current_start = None
                current_type = None

        if current_type is not None and current_start is not None:
            spans.append(
                {
                    "text": " ".join(tokens[current_start : len(tokens)]),
                    "type": current_type,
                    "start": current_start,
                    "end": len(tokens),
                }
            )

        return spans

    def _extract_narration_chains(
        self, tokens: List[str], entities: List[Dict]
    ) -> List[Dict]:
        """
        Extract NARRATED_FROM relations from isnad-style patterns.
        """
        scholars = sorted(
            [entity for entity in entities if entity.get("type") == "SCHOLAR"],
            key=lambda entity: (int(entity["start"]), int(entity["end"])),
        )
        if len(scholars) < 2:
            return []

        normalized_tokens = [self.normalizer.normalize(token) for token in tokens]
        relations: List[Dict] = []

        for idx in range(len(scholars) - 1):
            source = scholars[idx]
            target = scholars[idx + 1]

            between_tokens = normalized_tokens[source["end"] : target["start"]]
            has_connector = any(
                token in self._chain_connectors for token in between_tokens
            )
            adjacent = source["end"] == target["start"]

            # Trigger words can indicate chain start but are not required for each edge.
            trigger_window_start = max(0, source["start"] - 3)
            has_trigger_before = any(
                token in self._narration_triggers
                for token in normalized_tokens[trigger_window_start : source["start"]]
            )

            if has_connector:
                confidence = 0.9
                connector = next(
                    (
                        token
                        for token in between_tokens
                        if token in self._chain_connectors
                    ),
                    "connector",
                )
                evidence = f"isnad connector ({connector}) between scholars"
            elif adjacent:
                confidence = 0.7
                evidence = "adjacent scholar entities in narration chain"
            elif has_trigger_before:
                confidence = 0.7
                evidence = "narration trigger before scholar pair"
            else:
                continue

            relations.append(
                {
                    "type": "NARRATED_FROM",
                    "source": self._copy_entity(source),
                    "target": self._copy_entity(target),
                    "confidence": confidence,
                    "evidence": evidence,
                }
            )

        return relations

    def _extract_book_relations(
        self,
        tokens: List[str],
        entities: List[Dict],
        metadata: Dict | None = None,
    ) -> List[Dict]:
        """
        Extract IN_BOOK relations using book/author cue patterns.
        """
        normalized_tokens = [self.normalizer.normalize(token) for token in tokens]
        hadith_source = self._build_hadith_source(metadata)
        relations: List[Dict] = []

        for entity in entities:
            entity_type = entity.get("type")
            start = int(entity["start"])
            if start <= 0:
                continue
            cue = normalized_tokens[start - 1]
            if cue not in self._book_cues:
                continue

            if entity_type == "BOOK":
                relations.append(
                    {
                        "type": "IN_BOOK",
                        "source": hadith_source,
                        "target": self._copy_entity(entity),
                        "confidence": 0.9,
                        "evidence": f"explicit cue ({tokens[start - 1]}) + BOOK",
                    }
                )
                continue

            if entity_type == "SCHOLAR":
                normalized_author = self.normalizer.normalize(entity["text"])
                mapped_book = self.author_book_map.get(normalized_author)
                if not mapped_book:
                    continue

                relations.append(
                    {
                        "type": "IN_BOOK",
                        "source": hadith_source,
                        "target": {
                            "text": mapped_book,
                            "type": "BOOK",
                            "start": -1,
                            "end": -1,
                        },
                        "confidence": 0.7,
                        "evidence": f"cue ({tokens[start - 1]}) + known author inference",
                    }
                )

        return relations

    def _extract_concept_mentions(
        self,
        entities: List[Dict],
        metadata: Dict | None,
    ) -> List[Dict]:
        """
        Extract MENTIONS_CONCEPT relation via co-occurrence.
        """
        concepts = [entity for entity in entities if entity.get("type") == "CONCEPT"]
        hadith_refs = [
            entity for entity in entities if entity.get("type") == "HADITH_REF"
        ]
        if not concepts:
            return []

        relations: List[Dict] = []
        if hadith_refs:
            for hadith_ref in hadith_refs:
                for concept in concepts:
                    relations.append(
                        {
                            "type": "MENTIONS_CONCEPT",
                            "source": self._copy_entity(hadith_ref),
                            "target": self._copy_entity(concept),
                            "confidence": 0.6,
                            "evidence": "co-occurrence of HADITH_REF and CONCEPT in sentence",
                        }
                    )
            return relations

        hadith_source = self._build_hadith_source(metadata)
        for concept in concepts:
            relations.append(
                {
                    "type": "MENTIONS_CONCEPT",
                    "source": hadith_source,
                    "target": self._copy_entity(concept),
                    "confidence": 0.6,
                    "evidence": "concept linked to sentence-level hadith metadata",
                }
            )
        return relations

    def _extract_authorship(self, entities: List[Dict]) -> List[Dict]:
        """
        Extract AUTHORED relations using known author-book pairs.
        """
        scholars = [entity for entity in entities if entity.get("type") == "SCHOLAR"]
        books = [entity for entity in entities if entity.get("type") == "BOOK"]

        normalized_books = {
            self.normalizer.normalize(book["text"]): book for book in books
        }

        relations: List[Dict] = []
        for scholar in scholars:
            normalized_scholar = self.normalizer.normalize(scholar["text"])
            mapped_book = self.author_book_map.get(normalized_scholar)
            if not mapped_book:
                continue

            normalized_mapped_book = self.normalizer.normalize(mapped_book)
            target_book = normalized_books.get(normalized_mapped_book)
            if target_book is None:
                target = {
                    "text": mapped_book,
                    "type": "BOOK",
                    "start": -1,
                    "end": -1,
                }
                evidence = "known scholar-book ground truth pair (inferred)"
            else:
                target = self._copy_entity(target_book)
                evidence = "known scholar-book pair with explicit BOOK mention"

            relations.append(
                {
                    "type": "AUTHORED",
                    "source": self._copy_entity(scholar),
                    "target": target,
                    "confidence": 1.0,
                    "evidence": evidence,
                }
            )

        return relations

    def _build_hadith_source(self, metadata: Dict | None) -> Dict:
        hadith_id = None
        if metadata:
            hadith_id = metadata.get("hadith_id") or metadata.get("id")
        hadith_text = str(hadith_id) if hadith_id is not None else "CURRENT_HADITH"
        return {
            "text": hadith_text,
            "type": "HADITH_REF",
            "start": -1,
            "end": -1,
        }

    def _copy_entity(self, entity: Dict) -> Dict:
        return {
            "text": str(entity.get("text", "")),
            "type": str(entity.get("type", "")),
            "start": int(entity.get("start", -1)),
            "end": int(entity.get("end", -1)),
        }

    def _dedupe_relations(self, relations: Iterable[Dict]) -> List[Dict]:
        deduped: Dict[Tuple, Dict] = {}
        for relation in relations:
            source = relation.get("source", {})
            target = relation.get("target", {})
            key = (
                relation.get("type"),
                source.get("type"),
                source.get("text"),
                int(source.get("start", -1)),
                int(source.get("end", -1)),
                target.get("type"),
                target.get("text"),
                int(target.get("start", -1)),
                int(target.get("end", -1)),
            )
            current = deduped.get(key)
            if current is None or float(relation.get("confidence", 0.0)) > float(
                current.get("confidence", 0.0)
            ):
                deduped[key] = relation

        return list(deduped.values())
