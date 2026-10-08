"""
Scoring Engine Module.
Calculates transparent weighted multi-factor candidate scores (0-100) using configurable weights.
"""

from typing import Dict, Any, Tuple
from config import SCORING_WEIGHTS, RECOMMENDATION_THRESHOLDS
from modules.models.candidate import CandidateProfile
from modules.models.job import JobRequirements
from modules.models.result import CandidateScoreBreakdown
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ScoringEngine:
    """Computes transparent, explainable scores across five core qualification pillars."""

    @classmethod
    def calculate_score(
        cls,
        candidate_profile: CandidateProfile,
        job_requirements: JobRequirements,
        semantic_skill_score: float,
        evidence_score: float,
    ) -> Tuple[float, CandidateScoreBreakdown, str]:
        """
        Calculates normalized scores across all pillars and maps to a recommendation category.

        Args:
            candidate_profile: Candidate profile.
            job_requirements: Job requirements.
            semantic_skill_score: Score from semantic skill matching (0-100).
            evidence_score: Score from evidence validation (0-100).

        Returns:
            Tuple of:
                - overall_score: float (0.0 to 100.0)
                - breakdown: CandidateScoreBreakdown
                - recommendation: str ("HIGH_RELEVANCE", "RELEVANT", "MODERATE_RELEVANCE", "LOW_RELEVANCE")
        """
        # 1. Experience Relevance Score (0 - 100)
        exp_score = cls._score_experience(candidate_profile, job_requirements)

        # 2. Education Relevance Score (0 - 100)
        edu_score = cls._score_education(candidate_profile, job_requirements)

        # 3. Projects & Certifications Score (0 - 100)
        proj_certs_score = cls._score_projects_certs(candidate_profile, job_requirements)

        # 4. Overall Weighted Score Calculation
        w_skill = SCORING_WEIGHTS.get("semantic_skill_match", 0.35)
        w_ev = SCORING_WEIGHTS.get("skill_evidence", 0.25)
        w_exp = SCORING_WEIGHTS.get("experience_relevance", 0.20)
        w_edu = SCORING_WEIGHTS.get("education_relevance", 0.10)
        w_proj = SCORING_WEIGHTS.get("projects_certs", 0.10)

        overall = (
            (semantic_skill_score * w_skill)
            + (evidence_score * w_ev)
            + (exp_score * w_exp)
            + (edu_score * w_edu)
            + (proj_certs_score * w_proj)
        )

        overall = round(min(100.0, max(0.0, overall)), 2)

        breakdown = CandidateScoreBreakdown(
            semantic_skill_match_score=round(semantic_skill_score, 2),
            skill_evidence_score=round(evidence_score, 2),
            experience_relevance_score=round(exp_score, 2),
            education_relevance_score=round(edu_score, 2),
            projects_certs_score=round(proj_certs_score, 2),
            overall_score=overall,
        )

        recommendation = cls._determine_recommendation(overall)

        return overall, breakdown, recommendation

    @classmethod
    def _score_experience(cls, profile: CandidateProfile, job: JobRequirements) -> float:
        """Evaluates candidate years of experience against job requirement."""
        req_years = job.experience_years
        cand_years = profile.experience_years

        if req_years <= 0.0:
            # Entry level position
            return 90.0 if cand_years > 0 else 75.0

        if cand_years >= req_years:
            return 100.0

        ratio = cand_years / req_years
        return round(ratio * 100.0, 2)

    @classmethod
    def _score_education(cls, profile: CandidateProfile, job: JobRequirements) -> float:
        """Evaluates candidate degree level and relevance."""
        degrees = [e.get("degree", "").lower() for e in profile.education]
        if not degrees:
            return 50.0  # Basic default if not explicitly extracted

        # Check for Master's / PhD
        if any("m.tech" in d or "m.s" in d or "master" in d or "ph.d" in d or "phd" in d for d in degrees):
            return 100.0

        # Check for Bachelor's / B.Tech / B.E. / BCA / BS
        if any("b.tech" in d or "b.e" in d or "bachelor" in d or "bca" in d or "b.s" in d or "bsc" in d for d in degrees):
            return 90.0

        return 75.0

    @classmethod
    def _score_projects_certs(cls, profile: CandidateProfile, job: JobRequirements) -> float:
        """Evaluates depth of candidate projects and professional certifications."""
        num_projs = len(profile.projects)
        num_certs = len(profile.certifications)

        score = 0.0
        # Projects component (up to 60 points)
        if num_projs >= 3:
            score += 60.0
        elif num_projs == 2:
            score += 45.0
        elif num_projs == 1:
            score += 30.0

        # Certifications component (up to 40 points)
        if num_certs >= 2:
            score += 40.0
        elif num_certs == 1:
            score += 25.0

        # If projects were rich, give bonus
        if score == 0.0 and len(profile.technical_tools) > 3:
            score = 50.0

        return min(100.0, score)

    @classmethod
    def _determine_recommendation(cls, overall_score: float) -> str:
        """Maps overall score to recommendation string."""
        if overall_score >= RECOMMENDATION_THRESHOLDS["HIGH_RELEVANCE"]:
            return "HIGH RELEVANCE"
        elif overall_score >= RECOMMENDATION_THRESHOLDS["RELEVANT"]:
            return "RELEVANT"
        elif overall_score >= RECOMMENDATION_THRESHOLDS["MODERATE_RELEVANCE"]:
            return "MODERATE RELEVANCE"
        else:
            return "LOW RELEVANCE"
