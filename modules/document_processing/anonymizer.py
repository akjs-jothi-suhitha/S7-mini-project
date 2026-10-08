"""
Anonymizer Module.
Masks Personally Identifiable Information (PII) to generate privacy-safe, anonymized resume representations.
"""

import re
import uuid
from typing import Dict, List, Tuple
from config import ANONYMIZATION_PREFIX
from modules.document_processing.pii_detector import PIIDetector


class Anonymizer:
    """Masks detected PII and generates unique candidate identifiers."""

    @staticmethod
    def generate_candidate_id() -> str:
        """Generates an internal UUID for the candidate."""
        return f"CAND-{uuid.uuid4().hex[:8].upper()}"

    @staticmethod
    def generate_anonymized_id() -> str:
        """Generates a recruiter-facing anonymized ID, e.g. ANON-A1B2."""
        return f"{ANONYMIZATION_PREFIX}-{uuid.uuid4().hex[:4].upper()}"

    @classmethod
    def anonymize_text(
        cls,
        text: str,
        pii_entities: Dict[str, List[str]] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Replaces identified PII in text with standardized masking placeholders.

        Args:
            text: Text to anonymize.
            pii_entities: Pre-detected PII dictionary or None (will run detector).

        Returns:
            Tuple of (anonymized_text, summary_counts_dict).
        """
        if not text:
            return "", {}

        if pii_entities is None:
            pii_entities = PIIDetector.detect_pii(text)

        anonymized = text
        counts: Dict[str, int] = {
            "names": 0,
            "emails": 0,
            "phones": 0,
            "urls": 0,
            "addresses": 0,
            "dates": 0,
        }

        # 1. Mask Emails
        for email in pii_entities.get("emails", []):
            if email:
                pattern = re.escape(email)
                anonymized, count = re.subn(pattern, "[ANONYMIZED_EMAIL]", anonymized, flags=re.IGNORECASE)
                counts["emails"] += count

        # 2. Mask Phones
        for phone in pii_entities.get("phones", []):
            if phone:
                pattern = re.escape(phone)
                anonymized, count = re.subn(pattern, "[ANONYMIZED_PHONE]", anonymized)
                counts["phones"] += count

        # 3. Mask URLs
        for url in pii_entities.get("urls", []):
            if url:
                pattern = re.escape(url)
                anonymized, count = re.subn(pattern, "[ANONYMIZED_URL]", anonymized, flags=re.IGNORECASE)
                counts["urls"] += count

        # 4. Mask Addresses
        for addr in pii_entities.get("addresses", []):
            if addr:
                pattern = re.escape(addr)
                anonymized, count = re.subn(pattern, "[ANONYMIZED_ADDRESS]", anonymized, flags=re.IGNORECASE)
                counts["addresses"] += count

        # 5. Mask Dates / DOB
        for dt in pii_entities.get("dates", []):
            if dt:
                pattern = re.escape(dt)
                anonymized, count = re.subn(pattern, "[ANONYMIZED_DATE]", anonymized, flags=re.IGNORECASE)
                counts["dates"] += count

        # 6. Mask Names
        for name in pii_entities.get("names", []):
            if name and len(name.strip()) > 1:
                pattern = r"\b" + re.escape(name.strip()) + r"\b"
                anonymized, count = re.subn(pattern, "[ANONYMIZED_NAME]", anonymized, flags=re.IGNORECASE)
                counts["names"] += count

        return anonymized, counts
