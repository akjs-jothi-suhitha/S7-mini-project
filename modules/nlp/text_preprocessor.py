"""
Text Preprocessor Module.
Provides tokenization, sentence extraction, lemmatization, and clean section segmentation using spaCy / fallback.
"""

import re
from typing import List, Dict, Optional, Tuple
from config import SPACY_MODEL
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)

_SPACY_NLP = None


def get_spacy_nlp():
    """Loads spaCy model lazily with fallback."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy
            try:
                _SPACY_NLP = spacy.load(SPACY_MODEL)
            except Exception:
                try:
                    _SPACY_NLP = spacy.load("en_core_web_sm")
                except Exception:
                    _SPACY_NLP = spacy.blank("en")
        except ImportError:
            logger.info("spaCy not installed, using built-in NLP preprocessor.")
            _SPACY_NLP = None
    return _SPACY_NLP


class TextPreprocessor:
    """Provides NLP sentence segmentation, tokenization, and section parsing."""

    # Common standard resume sections
    SECTION_HEADERS = [
        "summary", "objective", "professional summary", "about me",
        "skills", "technical skills", "core competencies", "technologies", "key skills",
        "work experience", "experience", "employment history", "professional experience", "work history",
        "education", "academic background", "academic qualifications",
        "projects", "academic projects", "key projects", "personal projects",
        "certifications", "licenses & certifications", "courses", "certificates",
        "achievements", "awards", "publications", "languages"
    ]

    @classmethod
    def segment_sentences(cls, text: str) -> List[str]:
        """Splits text into meaningful, clean sentences for evidence extraction."""
        if not text:
            return []

        nlp = get_spacy_nlp()
        if nlp is not None and "parser" in nlp.pipe_names or (nlp and "senter" in nlp.pipe_names):
            try:
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                if sentences:
                    return sentences
            except Exception:
                pass

        # Robust regex-based sentence segmentation fallback
        # Split on newline bullets, periods, semicolons
        raw_chunks = re.split(r"(?:\n+|(?:(?<=[.!?])\s+(?=[A-Z0-9]))|•|\*|- )", text)
        sentences = []
        for chunk in raw_chunks:
            cleaned = chunk.strip()
            if cleaned and len(cleaned) > 3:
                sentences.append(cleaned)
        return sentences

    @classmethod
    def extract_sections(cls, text: str) -> Dict[str, str]:
        """
        Segments resume text into distinct functional sections.

        Returns:
            Dictionary mapping section names (e.g. 'skills', 'experience', 'projects', 'education', 'certifications', 'general') to text.
        """
        if not text:
            return {"general": ""}

        lines = text.splitlines()
        sections: Dict[str, List[str]] = {"general": []}
        current_section = "general"

        header_pattern = re.compile(
            r"^(?:[0-9IVX]+\.\s*)?([A-Za-z\s&/]{2,35})(?::)?$",
            re.IGNORECASE
        )

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if line matches a known section header
            matched_header = None
            clean_line_lower = re.sub(r"[^a-zA-Z\s]", "", line_str).strip().lower()

            for header in cls.SECTION_HEADERS:
                if clean_line_lower == header or (len(line_str) <= 35 and clean_line_lower.startswith(header)):
                    matched_header = header
                    break

            if matched_header:
                # Map to standard canonical section bucket
                if any(k in matched_header for k in ["skill", "technolog", "competenc"]):
                    current_section = "skills"
                elif any(k in matched_header for k in ["experience", "employment", "history", "work"]):
                    current_section = "experience"
                elif any(k in matched_header for k in ["project"]):
                    current_section = "projects"
                elif any(k in matched_header for k in ["education", "academic", "qualification"]):
                    current_section = "education"
                elif any(k in matched_header for k in ["certif", "course", "license"]):
                    current_section = "certifications"
                else:
                    current_section = matched_header.replace(" ", "_")

                if current_section not in sections:
                    sections[current_section] = []
            else:
                sections[current_section].append(line_str)

        return {k: "\n".join(v).strip() for k, v in sections.items() if v}
