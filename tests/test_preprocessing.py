"""Tests for Arabic text preprocessing and normalization."""

from pathlib import Path

import pytest

from src.preprocessing.gazetteers import GazetteerMatcher
from src.preprocessing.normalize import ArabicNormalizer


def test_remove_tashkeel() -> None:
    """Diacritics should be removed while base letters are preserved."""
    normalizer = ArabicNormalizer()
    assert normalizer.remove_tashkeel("مُحَمَّد") == "محمد"


def test_normalize_alif() -> None:
    """Alif variants should normalize to bare Alif."""
    normalizer = ArabicNormalizer()
    assert normalizer.normalize_alif("أبو") == "ابو"
    assert normalizer.normalize_alif("إسلام") == "اسلام"
    assert normalizer.normalize_alif("آية") == "اية"
    assert normalizer.normalize("آية") == "ايه"


def test_normalize_taa_marbuta() -> None:
    """Taa marbuta should normalize to haa."""
    normalizer = ArabicNormalizer()
    assert normalizer.normalize_taa_marbuta("مكة") == "مكه"


def test_normalize_alif_maqsura() -> None:
    """Alif maqsura should normalize to yaa."""
    normalizer = ArabicNormalizer()
    assert normalizer.normalize_alif_maqsura("موسى") == "موسي"


def test_remove_tatweel() -> None:
    """Tatweel elongation marks should be removed."""
    normalizer = ArabicNormalizer()
    assert normalizer.remove_tatweel("مـحـمـد") == "محمد"


def test_remove_extra_whitespace() -> None:
    """Mixed whitespace should collapse into single spaces."""
    normalizer = ArabicNormalizer()
    assert normalizer.remove_extra_whitespace("  هذا\tنص\n\nعربي  ") == "هذا نص عربي"


def test_strip_html() -> None:
    """HTML tags should be stripped and entities decoded."""
    normalizer = ArabicNormalizer()
    text = "<p>السلام&nbsp;عليكم</p><br><b>ورحمة الله</b>"
    stripped = normalizer.strip_html(text)
    assert "<" not in stripped and ">" not in stripped
    assert "\xa0" in stripped
    assert "السلام" in stripped
    assert "ورحمة الله" in stripped


def test_full_pipeline_hadith_text() -> None:
    """Full pipeline should clean mixed Arabic orthographic noise."""
    normalizer = ArabicNormalizer()
    dirty_text = " <div>قَالَ&nbsp;رَسُولُ&nbsp;اللَّهِ: إِنَّمَا الأَعْمَالُ بِالنِّيَّاتِ.</div> "
    assert normalizer.normalize(dirty_text) == "قال رسول الله: انما الاعمال بالنيات."


def test_non_arabic_pass_through() -> None:
    """Non-Arabic text should remain unchanged except whitespace normalization."""
    normalizer = ArabicNormalizer()
    assert normalizer.normalize("Hello, world! 123") == "Hello, world! 123"


def test_empty_string_handling() -> None:
    """Empty input should remain empty through normalization."""
    normalizer = ArabicNormalizer()
    assert normalizer.normalize("") == ""


def _write_gazetteer(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def sample_gazetteer_dir(tmp_path: Path) -> Path:
    al_bukhari = "\u0627\u0644\u0628\u062e\u0627\u0631\u064a"
    sahih_al_bukhari = (
        "\u0635\u062d\u064a\u062d \u0627\u0644\u0628\u062e\u0627\u0631\u064a"
    )
    abu = "\u0623\u0628\u0648"
    abu_hurayrah = "\u0623\u0628\u0648 \u0647\u0631\u064a\u0631\u0629"
    abu_hurayrah_tashkeel = (
        "\u0623\u064e\u0628\u064f\u0648 "
        "\u0647\u064f\u0631\u064e\u064a\u0652\u0631\u064e\u0629\u064e"
    )

    _write_gazetteer(
        tmp_path / "scholars.txt",
        [
            f"{al_bukhari}|\u0645\u062d\u0645\u062f \u0628\u0646 \u0625\u0633\u0645\u0627\u0639\u064a\u0644 \u0627\u0644\u0628\u062e\u0627\u0631\u064a",
            abu,
            f"{abu_hurayrah}|{abu_hurayrah_tashkeel}",
        ],
    )
    _write_gazetteer(
        tmp_path / "books.txt",
        [
            f"{sahih_al_bukhari}|\u0627\u0644\u062c\u0627\u0645\u0639 \u0627\u0644\u0635\u062d\u064a\u062d",
        ],
    )
    _write_gazetteer(
        tmp_path / "concepts.txt",
        [
            "\u0627\u0644\u062a\u0648\u062d\u064a\u062f",
        ],
    )
    _write_gazetteer(
        tmp_path / "places.txt",
        [
            "\u0627\u0644\u0645\u062f\u064a\u0646\u0629|\u064a\u062b\u0631\u0628",
        ],
    )
    return tmp_path


def test_gazetteer_matches_bukhari_as_scholar(sample_gazetteer_dir: Path) -> None:
    matcher = GazetteerMatcher(gazetteer_dir=str(sample_gazetteer_dir))
    text = "\u0642\u0627\u0644 \u0627\u0644\u0628\u062e\u0627\u0631\u064a \u0641\u064a \u0643\u062a\u0627\u0628\u0647"

    matches = matcher.match(text)

    scholar_matches = [
        match
        for match in matches
        if match["text"] == "\u0627\u0644\u0628\u062e\u0627\u0631\u064a"
    ]
    assert scholar_matches
    assert scholar_matches[0]["entity_type"] == "SCHOLAR"


def test_gazetteer_matches_sahih_bukhari_as_book(sample_gazetteer_dir: Path) -> None:
    matcher = GazetteerMatcher(gazetteer_dir=str(sample_gazetteer_dir))
    text = "\u0647\u0630\u0627 \u0634\u0631\u062d \u0635\u062d\u064a\u062d \u0627\u0644\u0628\u062e\u0627\u0631\u064a"

    matches = matcher.match(text)

    assert any(
        match["text"]
        == "\u0635\u062d\u064a\u062d \u0627\u0644\u0628\u062e\u0627\u0631\u064a"
        and match["entity_type"] == "BOOK"
        for match in matches
    )
    assert not any(
        match["text"] == "\u0627\u0644\u0628\u062e\u0627\u0631\u064a"
        and match["entity_type"] == "SCHOLAR"
        for match in matches
    )


def test_gazetteer_longest_match_prefers_abu_hurayrah(
    sample_gazetteer_dir: Path,
) -> None:
    matcher = GazetteerMatcher(gazetteer_dir=str(sample_gazetteer_dir))
    text = "\u0639\u0646 \u0623\u0628\u0648 \u0647\u0631\u064a\u0631\u0629 \u0631\u0636\u064a \u0627\u0644\u0644\u0647 \u0639\u0646\u0647"

    matches = matcher.match(text)

    assert any(
        match["text"] == "\u0623\u0628\u0648 \u0647\u0631\u064a\u0631\u0629"
        and match["entity_type"] == "SCHOLAR"
        for match in matches
    )
    assert not any(
        match["text"] == "\u0623\u0628\u0648" and match["entity_type"] == "SCHOLAR"
        for match in matches
    )


def test_gazetteer_matches_tashkeel_variant(sample_gazetteer_dir: Path) -> None:
    matcher = GazetteerMatcher(gazetteer_dir=str(sample_gazetteer_dir))
    text = (
        "\u0631\u0648\u0649 "
        "\u0623\u064e\u0628\u064f\u0648 \u0647\u064f\u0631\u064e\u064a\u0652\u0631\u064e\u0629\u064e "
        "\u0627\u0644\u062d\u062f\u064a\u062b"
    )
    normalized_canonical = ArabicNormalizer().normalize(
        "\u0623\u0628\u0648 \u0647\u0631\u064a\u0631\u0629"
    )

    matches = matcher.match(text)

    matched = [
        match
        for match in matches
        if match["entity_type"] == "SCHOLAR"
        and match["canonical_name"] == normalized_canonical
    ]
    assert matched
    first_match = matched[0]
    assert text[first_match["start"] : first_match["end"]] == first_match["text"]
