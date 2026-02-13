"""Tests for silver annotation generation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest

from src.ner.silver_annotator import SilverAnnotator


@pytest.fixture()
def sample_gazetteer_dir() -> Path:
    return Path("tests/fixtures/gazetteers")


def _find_subsequence_start(tokens: List[str], sequence: List[str]) -> int:
    for idx in range(0, len(tokens) - len(sequence) + 1):
        if tokens[idx : idx + len(sequence)] == sequence:
            return idx
    return -1


def _token_labels(annotations: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    tokens = [token for token, _ in annotations]
    labels = [label for _, label in annotations]
    return tokens, labels


def _find_label_for_token(annotations: List[Tuple[str, str]], token: str) -> str:
    for current_token, current_label in annotations:
        if current_token == token:
            return current_label
    raise AssertionError(f"Token not found in annotations: {token}")


def test_annotate_from_sanadset_marks_nar_span(sample_gazetteer_dir: Path) -> None:
    annotator = SilverAnnotator(gazetteer_dir=str(sample_gazetteer_dir))
    text = (
        "<SANAD>حدثنا <NAR>عبد الله بن يوسف</NAR> عن <NAR>مالك</NAR></SANAD>"
        "<MATN>...</MATN>"
    )

    annotations = annotator.annotate_from_sanadset(text)
    tokens, labels = _token_labels(annotations)

    span_tokens = ["عبد", "الله", "بن", "يوسف"]
    span_start = _find_subsequence_start(tokens, span_tokens)
    assert span_start >= 0
    assert labels[span_start : span_start + 4] == [
        "B-SCHOLAR",
        "I-SCHOLAR",
        "I-SCHOLAR",
        "I-SCHOLAR",
    ]
    assert _find_label_for_token(annotations, "مالك") == "B-SCHOLAR"


def test_annotate_from_raw_isnad_patterns(sample_gazetteer_dir: Path) -> None:
    annotator = SilverAnnotator(gazetteer_dir=str(sample_gazetteer_dir))
    text = "حدثنا محمد بن إسماعيل عن مالك"

    annotations = annotator.annotate_from_raw(text)
    tokens, labels = _token_labels(annotations)

    scholar_span = ["محمد", "بن", "اسماعيل"]
    span_start = _find_subsequence_start(tokens, scholar_span)
    assert span_start >= 0
    assert labels[span_start : span_start + 3] == ["B-SCHOLAR", "I-SCHOLAR", "I-SCHOLAR"]
    assert _find_label_for_token(annotations, "مالك") == "B-SCHOLAR"


def test_book_disambiguation_prefers_book_context(sample_gazetteer_dir: Path) -> None:
    annotator = SilverAnnotator(gazetteer_dir=str(sample_gazetteer_dir))
    text = "في صحيح البخاري"

    annotations = annotator.annotate_from_raw(text)
    bukhari_label = _find_label_for_token(annotations, "البخاري")

    assert bukhari_label.endswith("BOOK")
    assert not bukhari_label.endswith("SCHOLAR")


def test_concept_gazetteer_match_on_matn(sample_gazetteer_dir: Path) -> None:
    annotator = SilverAnnotator(gazetteer_dir=str(sample_gazetteer_dir))
    text = "نهى عن الربا"

    annotations = annotator.annotate_from_raw(text)
    assert _find_label_for_token(annotations, "الربا") == "B-CONCEPT"


def test_full_pipeline_merges_scholar_and_concept(sample_gazetteer_dir: Path) -> None:
    annotator = SilverAnnotator(gazetteer_dir=str(sample_gazetteer_dir))
    text = (
        "<SANAD>حدثنا <NAR>عبد الله بن يوسف</NAR> عن <NAR>مالك</NAR></SANAD>"
        "<MATN>نهى عن الربا</MATN>"
    )

    annotations = annotator.annotate_from_sanadset(text)
    tokens, labels = _token_labels(annotations)

    scholar_span = ["عبد", "الله", "بن", "يوسف"]
    scholar_start = _find_subsequence_start(tokens, scholar_span)
    assert scholar_start >= 0
    assert labels[scholar_start : scholar_start + 4] == [
        "B-SCHOLAR",
        "I-SCHOLAR",
        "I-SCHOLAR",
        "I-SCHOLAR",
    ]
    assert _find_label_for_token(annotations, "الربا") == "B-CONCEPT"


def test_to_bio_format_has_valid_prefixes(sample_gazetteer_dir: Path) -> None:
    annotator = SilverAnnotator(gazetteer_dir=str(sample_gazetteer_dir))
    annotations = annotator.annotate_from_raw("حدثنا محمد بن إسماعيل عن مالك")
    bio_text = annotator.to_bio_format(annotations)

    rows = [row for row in bio_text.splitlines() if row.strip()]
    mapping = dict(row.split("\t", 1) for row in rows)

    assert mapping["حدثنا"] == "O"
    assert mapping["محمد"] == "B-SCHOLAR"
    assert mapping["بن"] == "I-SCHOLAR"
