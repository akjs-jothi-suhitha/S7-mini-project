"""
Logging utility for the Resume Screening System.
Ensures privacy-compliant logging where PII and raw resume texts are never logged.
"""

import logging
import sys
import re
from typing import Optional

_LOGGER: Optional[logging.Logger] = None

# Regex patterns to sanitize in case accidental PII is passed to logs
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

class PrivacyFilter(logging.Filter):
    """Filter that masks any inadvertent email or phone patterns in log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _EMAIL_PATTERN.sub("[MASKED_EMAIL]", record.msg)
            record.msg = _PHONE_PATTERN.sub("[MASKED_PHONE]", record.msg)
        return True


def get_logger(name: str = "ResumeScreening") -> logging.Logger:
    """Returns a privacy-aware logger instance."""
    global _LOGGER
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.addFilter(PrivacyFilter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger
