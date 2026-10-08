"""
PII Detector Module.
Identifies Personally Identifiable Information (PII) including Name, Email, Phone, URL, Address, and DOB.
"""

import re
from typing import Dict, List, Tuple


class PIIDetector:
    """Detects PII entities and spans within resume text."""

    # Regex for email addresses
    _EMAIL_REGEX = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
    )

    # Regex for international and domestic phone numbers
    _PHONE_REGEX = re.compile(
        r"(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4})"
    )

    # Regex for URLs (LinkedIn, GitHub, portfolios, http/https/www)
    _URL_REGEX = re.compile(
        r"\b(?:https?://|www\.)[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|/))|\b(?:linkedin\.com/in/|github\.com/)[A-Za-z0-9_-]+\b",
        re.IGNORECASE,
    )

    # Regex for Date of Birth / DOB patterns
    _DOB_REGEX = re.compile(
        r"\b(?:DOB|D\.O\.B\.|Date of Birth|Birth Date)[\s:]*([0-9]{1,2}[-/.][0-9]{1,2}[-/.][0-9]{2,4}|[A-Za-z]+\s+[0-9]{1,2},?\s+[0-9]{4})\b",
        re.IGNORECASE,
    )

    # Regex for ZIP / Postal Codes & Street addresses
    _POSTAL_REGEX = re.compile(
        r"\b(?:[A-Z]{1,2}\s*[0-9]{5}(?:-[0-9]{4})?|[0-9]{6})\b"
    )
    _STREET_REGEX = re.compile(
        r"\b\d{1,5}\s+[A-Za-z0-9.,\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way)\b",
        re.IGNORECASE,
    )

    @classmethod
    def detect_pii(cls, text: str) -> Dict[str, List[str]]:
        """
        Scans text and returns a dictionary of detected PII values by category.

        Args:
            text: Cleaned or raw resume text.

        Returns:
            Dictionary mapping entity types to lists of detected strings.
        """
        if not text:
            return {
                "names": [],
                "emails": [],
                "phones": [],
                "urls": [],
                "addresses": [],
                "dates": [],
            }

        emails = list(set(cls._EMAIL_REGEX.findall(text)))

        # Find phone numbers (filter out short digit matches)
        raw_phones = cls._PHONE_REGEX.findall(text)
        phones = list(
            set(
                p.strip()
                for p in raw_phones
                if sum(c.isdigit() for c in p) >= 10 and not any(p in em for em in emails)
            )
        )

        urls = list(set(cls._URL_REGEX.findall(text)))
        # Normalize findall output if it returned tuples
        normalized_urls = []
        for match in re.finditer(cls._URL_REGEX, text):
            normalized_urls.append(match.group(0).strip())
        urls = list(set(normalized_urls))

        # DOB / Dates
        dates = []
        for match in re.finditer(cls._DOB_REGEX, text):
            dates.append(match.group(0).strip())

        # Addresses
        addresses = []
        for match in re.finditer(cls._STREET_REGEX, text):
            addresses.append(match.group(0).strip())

        # Name detection (first few lines heuristic + title casing)
        names = cls._detect_candidate_names(text)

        return {
            "names": names,
            "emails": emails,
            "phones": phones,
            "urls": urls,
            "addresses": addresses,
            "dates": dates,
        }

    @classmethod
    def _detect_candidate_names(cls, text: str) -> List[str]:
        """Heuristic detection of candidate name from the top section of resume."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        names = []
        # Typically the candidate name appears in the first 3 lines
        candidate_header = lines[:3] if len(lines) >= 3 else lines
        name_blacklist = {
            "resume", "curriculum vitae", "cv", "profile", "summary",
            "education", "experience", "skills", "projects", "contact",
            "contact information", "personal details", "phone", "email"
        }

        for line in candidate_header:
            clean_line = re.sub(r"[,|•\-–—].*$", "", line).strip()
            # If line is 2-4 words, starts with capital letter, contains only letters/spaces
            words = clean_line.split()
            if 1 <= len(words) <= 4:
                if clean_line.lower() not in name_blacklist and not any(c.isdigit() for c in clean_line):
                    if all(w[0].isupper() for w in words if w and w.isalpha()):
                        names.append(clean_line)
                        break

        return names
