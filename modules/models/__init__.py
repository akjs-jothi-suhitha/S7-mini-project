"""Domain models package."""
from .candidate import CandidateProfile, RawCandidateDoc
from .job import JobRequirements
from .result import (
    SkillMatchItem,
    EvidenceItem,
    CandidateScoreBreakdown,
    CandidateScreeningResult,
    BatchScreeningResult,
)

__all__ = [
    "CandidateProfile",
    "RawCandidateDoc",
    "JobRequirements",
    "SkillMatchItem",
    "EvidenceItem",
    "CandidateScoreBreakdown",
    "CandidateScreeningResult",
    "BatchScreeningResult",
]
