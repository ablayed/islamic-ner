"""Tests for rule-based relation extraction."""

from __future__ import annotations

from src.relations.extract import RelationExtractor


def _by_type(relations, relation_type: str):
    return [rel for rel in relations if rel.get("type") == relation_type]


def test_narration_chain_extraction() -> None:
    extractor = RelationExtractor()
    tokens = [
        "\u062d\u062f\u062b\u0646\u0627",
        "\u0639\u0628\u062f",
        "\u0627\u0644\u0644\u0647",
        "\u0639\u0646",
        "\u0645\u0627\u0644\u0643",
        "\u0639\u0646",
        "\u0646\u0627\u0641\u0639",
    ]
    labels = [
        "O",
        "B-SCHOLAR",
        "I-SCHOLAR",
        "O",
        "B-SCHOLAR",
        "O",
        "B-SCHOLAR",
    ]

    relations = extractor.extract(tokens, labels, metadata={"hadith_id": "h1"})
    narration = _by_type(relations, "NARRATED_FROM")

    assert len(narration) == 2
    assert narration[0]["source"]["text"] == "\u0639\u0628\u062f \u0627\u0644\u0644\u0647"
    assert narration[0]["target"]["text"] == "\u0645\u0627\u0644\u0643"
    assert narration[1]["source"]["text"] == "\u0645\u0627\u0644\u0643"
    assert narration[1]["target"]["text"] == "\u0646\u0627\u0641\u0639"
    assert all(rel["confidence"] == 0.9 for rel in narration)


def test_book_relation_extraction() -> None:
    extractor = RelationExtractor()
    tokens = [
        "\u0631\u0648\u0627\u0647",
        "\u0627\u0644\u0628\u062e\u0627\u0631\u064a",
    ]
    labels = [
        "O",
        "B-BOOK",
    ]

    relations = extractor.extract(tokens, labels, metadata={"hadith_id": "h2"})
    in_book = _by_type(relations, "IN_BOOK")

    assert len(in_book) == 1
    assert in_book[0]["target"]["text"] == "\u0627\u0644\u0628\u062e\u0627\u0631\u064a"
    assert in_book[0]["target"]["type"] == "BOOK"


def test_concept_mention_extraction() -> None:
    extractor = RelationExtractor()
    tokens = [
        "\u062d\u062f\u064a\u062b",
        "\u0631\u0642\u0645",
        "123",
        "\u0627\u0644\u0631\u0628\u0627",
    ]
    labels = [
        "B-HADITH_REF",
        "I-HADITH_REF",
        "I-HADITH_REF",
        "B-CONCEPT",
    ]

    relations = extractor.extract(tokens, labels, metadata={"hadith_id": "h3"})
    mentions = _by_type(relations, "MENTIONS_CONCEPT")

    assert len(mentions) == 1
    assert mentions[0]["source"]["type"] == "HADITH_REF"
    assert mentions[0]["target"]["type"] == "CONCEPT"
    assert mentions[0]["target"]["text"] == "\u0627\u0644\u0631\u0628\u0627"


def test_authorship_extraction() -> None:
    extractor = RelationExtractor()
    tokens = [
        "\u0642\u0627\u0644",
        "\u0627\u0644\u0628\u062e\u0627\u0631\u064a",
    ]
    labels = [
        "O",
        "B-SCHOLAR",
    ]

    relations = extractor.extract(tokens, labels, metadata={"hadith_id": "h4"})
    authored = _by_type(relations, "AUTHORED")

    assert len(authored) >= 1
    assert authored[0]["source"]["text"] == "\u0627\u0644\u0628\u062e\u0627\u0631\u064a"
    assert authored[0]["target"]["text"] == "\u0635\u062d\u064a\u062d \u0627\u0644\u0628\u062e\u0627\u0631\u064a"
    assert authored[0]["confidence"] == 1.0


def test_full_pipeline_extracts_multiple_relation_types() -> None:
    extractor = RelationExtractor()
    tokens = [
        "\u062d\u062f\u062b\u0646\u0627",
        "\u0639\u0628\u062f",
        "\u0627\u0644\u0644\u0647",
        "\u0639\u0646",
        "\u0645\u0627\u0644\u0643",
        "\u0642\u0627\u0644",
        "\u0627\u0644\u0628\u062e\u0627\u0631\u064a",
        "\u0641\u064a",
        "\u0635\u062d\u064a\u062d",
        "\u0627\u0644\u0628\u062e\u0627\u0631\u064a",
        "\u062d\u062f\u064a\u062b",
        "\u0631\u0642\u0645",
        "1",
        "\u0627\u0644\u0631\u0628\u0627",
    ]
    labels = [
        "O",
        "B-SCHOLAR",
        "I-SCHOLAR",
        "O",
        "B-SCHOLAR",
        "O",
        "B-SCHOLAR",
        "O",
        "B-BOOK",
        "I-BOOK",
        "B-HADITH_REF",
        "I-HADITH_REF",
        "I-HADITH_REF",
        "B-CONCEPT",
    ]

    relations = extractor.extract(tokens, labels, metadata={"hadith_id": "h5"})
    relation_types = {relation["type"] for relation in relations}

    assert "NARRATED_FROM" in relation_types
    assert "IN_BOOK" in relation_types
    assert "MENTIONS_CONCEPT" in relation_types
    assert "AUTHORED" in relation_types
