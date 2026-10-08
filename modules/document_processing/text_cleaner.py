"""
Text Cleaner Module.
Cleans and normalizes resume and job description text while strictly preserving technical terms.
"""

import re
import unicodedata


class TextCleaner:
    """Cleans extracted raw text while protecting technical terms, punctuation, and structure."""

    # Patterns to temporarily protect from aggressive regexes if needed
    _TECH_PRESERVE_PATTERNS = [
        r"\bC\+\+",
        r"\bC\#",
        r"\.NET\b",
        r"\bNode\.js\b",
        r"\bReact\.js\b",
        r"\bVue\.js\b",
        r"\bCI/CD\b",
        r"\bTCP/IP\b",
        r"\bScikit-learn\b",
    ]

    @classmethod
    def clean(cls, text: str) -> str:
        """
        Cleans and normalizes raw text.

        Args:
            text: Uncleaned extracted text.

        Returns:
            Normalized clean text.
        """
        if not text:
            return ""

        # Normalize unicode (NFKC)
        normalized = unicodedata.normalize("NFKC", text)

        # Replace non-standard bullets and typographic symbols with standard delimiters
        bullet_chars = r"[•●▪■◆★\u2022\u2023\u25E6\u2043\u2219]"
        normalized = re.sub(bullet_chars, "\n- ", normalized)

        # Replace unusual quote characters and hyphens
        normalized = normalized.replace("“", '"').replace("”", '"')
        normalized = normalized.replace("‘", "'").replace("’", "'")
        normalized = normalized.replace("—", " - ").replace("–", " - ")

        # Replace control characters but retain newlines and tabs
        cleaned_chars = []
        for ch in normalized:
            if ch in ("\n", "\r", "\t") or (unicodedata.category(ch)[0] != "C" and ord(ch) >= 32):
                cleaned_chars.append(ch)
            else:
                cleaned_chars.append(" ")
        cleaned = "".join(cleaned_chars)

        # Fix multiple newlines and spaces
        cleaned = re.sub(r"\r\n|\r", "\n", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)

        return cleaned.strip()

    @classmethod
    def extract_lines(cls, text: str) -> list[str]:
        """Splits clean text into non-empty stripped lines."""
        return [line.strip() for line in text.splitlines() if line.strip()]
