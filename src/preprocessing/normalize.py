"""Arabic text normalization utilities for preprocessing.

This module exposes ``ArabicNormalizer`` which applies a consistent cleaning
pipeline for Arabic text before tokenization, NER, or downstream analysis.
"""

from __future__ import annotations

import html
import re


class ArabicNormalizer:
    """Normalize Arabic text into a canonical, model-friendly form.

    The class provides small focused methods for each normalization rule and a
    ``normalize`` orchestrator that applies them in a stable order.
    """

    _tashkeel_re = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")
    _alif_re = re.compile(r"[\u0622\u0623\u0625]")
    _taa_marbuta_re = re.compile(r"\u0629")
    _alif_maqsura_re = re.compile(r"\u0649")
    _tatweel_re = re.compile(r"\u0640")
    _whitespace_re = re.compile(r"\s+")
    _html_tag_re = re.compile(r"<[^>]+>")

    def normalize(self, text: str) -> str:
        """Run the full normalization pipeline in the required order.

        The order matters because HTML stripping, orthographic normalization,
        and whitespace cleanup each affect downstream steps.
        """

        text = self.strip_html(text)
        text = self.remove_tashkeel(text)
        text = self.normalize_alif(text)
        text = self.normalize_taa_marbuta(text)
        text = self.normalize_alif_maqsura(text)
        text = self.remove_tatweel(text)
        text = self.remove_extra_whitespace(text)
        return text

    def remove_tashkeel(self, text: str) -> str:
        """Remove Arabic diacritics (tashkeel/harakat) from text.

        This removes combining marks in:
        ``\\u0610-\\u061A``, ``\\u064B-\\u065F``, and ``\\u0670``.
        """

        return self._tashkeel_re.sub("", text)

    def normalize_alif(self, text: str) -> str:
        """Normalize Alif variants to bare Alif for lexical consistency.

        Replaces ``أ`` (U+0623), ``إ`` (U+0625), and ``آ`` (U+0622) with ``ا``.
        """

        return self._alif_re.sub("\u0627", text)

    def normalize_taa_marbuta(self, text: str) -> str:
        """Normalize Taa Marbuta to Haa to reduce orthographic variation.

        Replaces ``ة`` (U+0629) with ``ه`` (U+0647).
        """

        return self._taa_marbuta_re.sub("\u0647", text)

    def normalize_alif_maqsura(self, text: str) -> str:
        """Normalize Alif Maqsura to Yaa for canonical letter forms.

        Replaces ``ى`` (U+0649) with ``ي`` (U+064A).
        """

        return self._alif_maqsura_re.sub("\u064a", text)

    def remove_tatweel(self, text: str) -> str:
        """Remove Tatweel/Kashida elongation marks from Arabic text.

        This removes ``ـ`` (U+0640), a stylistic character not needed for NLP.
        """

        return self._tatweel_re.sub("", text)

    def remove_extra_whitespace(self, text: str) -> str:
        """Collapse repeated whitespace and trim text boundaries.

        Tabs, newlines, and multiple spaces are collapsed to a single space,
        then leading/trailing whitespace is stripped.
        """

        return self._whitespace_re.sub(" ", text).strip()

    def strip_html(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities.

        HTML tags are replaced with spaces to avoid accidental word joins.
        Entities are decoded so escaped content becomes plain text.
        """

        decoded = html.unescape(text)
        no_tags = self._html_tag_re.sub(" ", decoded)
        return html.unescape(no_tags)
