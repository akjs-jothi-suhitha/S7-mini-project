"""
Skill Validator Module.
Validates matched skills against candidate evidence and calculates confidence scores.
"""

from typing import List, Tuple
from config import EVIDENCE_LEVEL_WEIGHTS, EVIDENCE_LEVEL_LABELS
from modules.skill_validation.evidence_extractor import EvidenceExtractor
from modules.models.candidate import CandidateProfile
from modules.models.result import SkillMatchItem, EvidenceItem


class SkillValidator:
    """Validates skill claims with contextual evidence and computes evidence confidence scores."""

    @classmethod
    def validate_skills(
        cls,
        matched_skills: List[SkillMatchItem],
        partial_skills: List[SkillMatchItem],
        missing_skills: List[SkillMatchItem],
        candidate_profile: CandidateProfile,
    ) -> Tuple[List[SkillMatchItem], List[SkillMatchItem], List[SkillMatchItem], List[EvidenceItem], List[str], List[str], float]:
        """
        Enriches skill match items with evidence data and calculates normalized evidence score.

        Args:
            matched_skills: Matched skills list.
            partial_skills: Partially matched skills list.
            missing_skills: Missing skills list.
            candidate_profile: Candidate profile.

        Returns:
            Tuple of:
                - enriched_matched: List[SkillMatchItem]
                - enriched_partial: List[SkillMatchItem]
                - enriched_missing: List[SkillMatchItem]
                - evidence_items: List[EvidenceItem]
                - strong_evidence: List[str]
                - weak_evidence: List[str]
                - evidence_score: float (0.0 to 100.0)
        """
        all_items = matched_skills + partial_skills + missing_skills
        evidence_items: List[EvidenceItem] = []
        strong_evidence: List[str] = []
        weak_evidence: List[str] = []

        total_evidence_weight = 0.0

        for item in all_items:
            if item.status == "missing":
                item.evidence_level = 0
                item.evidence_confidence = EVIDENCE_LEVEL_LABELS[0]
                item.evidence_text = "No supporting evidence found in resume."
                continue

            level, label, section, snippet = EvidenceExtractor.find_evidence_for_skill(
                item.job_skill, candidate_profile
            )
            item.evidence_level = level
            item.evidence_confidence = label
            item.evidence_text = snippet

            weight = EVIDENCE_LEVEL_WEIGHTS.get(level, 0.0)
            total_evidence_weight += weight

            ev_obj = EvidenceItem(
                skill_name=item.job_skill,
                evidence_level=level,
                confidence_label=label,
                source_section=section,
                snippet=snippet,
            )
            evidence_items.append(ev_obj)

            if level >= 2:
                strong_evidence.append(f"{item.job_skill} → {section}: \"{snippet[:100]}...\"")
            elif level == 1:
                weak_evidence.append(f"{item.job_skill} → Mentioned in skills list only without contextual projects/experience.")

        # Calculate normalized score based on total skills evaluated
        total_eval_skills = len(matched_skills) + len(partial_skills) + len(missing_skills)
        if total_eval_skills > 0:
            evidence_score = (total_evidence_weight / total_eval_skills) * 100.0
        else:
            evidence_score = 0.0

        evidence_score = round(min(100.0, max(0.0, evidence_score)), 2)

        return (
            matched_skills,
            partial_skills,
            missing_skills,
            evidence_items,
            strong_evidence,
            weak_evidence,
            evidence_score,
        )
