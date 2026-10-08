"""
Job description data models.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class JobRequirements:
    """Structured representation of extracted job requirements."""
    job_id: str
    title: str = "Job Position"
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    education: List[str] = field(default_factory=list)
    experience_years: float = 0.0
    experience_details: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    raw_text: str = ""
    cleaned_text: str = ""
    job_keywords: List[str] = field(default_factory=list)
