"""
Candidate data models.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RawCandidateDoc:
    """Represents a loaded candidate document before extraction."""
    candidate_id: str
    anonymized_id: str
    file_name: str
    file_type: str
    raw_text: str
    cleaned_text: str = ""
    anonymized_text: str = ""
    pii_entities: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class CandidateProfile:
    """Structured candidate profile extracted from resume."""
    candidate_id: str
    anonymized_id: str
    candidate_name: str = ""
    skills: List[Dict[str, Any]] = field(default_factory=list)  # list of {"skill_name", "normalized_skill", "category"}
    education: List[Dict[str, str]] = field(default_factory=list)  # [{"degree", "institution", "year"}]
    experience_years: float = 0.0
    experience_entries: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    technical_tools: List[str] = field(default_factory=list)
    cleaned_text: str = ""
    anonymized_text: str = ""
    extracted_keywords: List[str] = field(default_factory=list)
