"""
Screening and evaluation result models.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class SkillMatchItem:
    """Individual skill matching outcome."""
    job_skill: str
    status: str  # "matched", "partially_matched", "missing"
    matched_candidate_skill: Optional[str] = None
    similarity_score: float = 0.0
    evidence_text: str = ""
    evidence_level: int = 0  # 0 to 3
    evidence_confidence: str = "Missing"  # "Missing", "Low", "Moderate", "High"


@dataclass
class EvidenceItem:
    """Detailed evidence item found for a skill."""
    skill_name: str
    evidence_level: int
    confidence_label: str
    source_section: str  # "Projects", "Experience", "Certifications", "Skills"
    snippet: str


@dataclass
class CandidateScoreBreakdown:
    """Detailed transparent breakdown of scoring components (all normalized 0-100)."""
    semantic_skill_match_score: float
    skill_evidence_score: float
    experience_relevance_score: float
    education_relevance_score: float
    projects_certs_score: float
    overall_score: float


@dataclass
class CandidateScreeningResult:
    """Complete screening result for an individual candidate."""
    candidate_id: str
    anonymized_id: str
    job_id: str
    candidate_name: str
    rank: int = 0
    overall_score: float = 0.0
    score_breakdown: Optional[CandidateScoreBreakdown] = None
    matched_skills: List[SkillMatchItem] = field(default_factory=list)
    partial_skills: List[SkillMatchItem] = field(default_factory=list)
    missing_skills: List[SkillMatchItem] = field(default_factory=list)
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    strong_evidence: List[str] = field(default_factory=list)
    weak_evidence: List[str] = field(default_factory=list)
    relevant_experience: List[str] = field(default_factory=list)
    relevant_projects: List[str] = field(default_factory=list)
    recommendation: str = "LOW_RELEVANCE"  # "HIGH_RELEVANCE", "RELEVANT", "MODERATE_RELEVANCE", "LOW_RELEVANCE"
    explanation: str = ""
    pii_summary: Dict[str, int] = field(default_factory=dict)


@dataclass
class BatchScreeningResult:
    """Aggregated batch screening output with ranking and fairness audit."""
    job_id: str
    job_title: str
    total_candidates: int
    ranked_candidates: List[CandidateScreeningResult] = field(default_factory=list)
    fairness_metrics: Dict[str, Any] = field(default_factory=dict)
