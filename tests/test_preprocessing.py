"""Tests for Arabic text preprocessing and normalization."""

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
