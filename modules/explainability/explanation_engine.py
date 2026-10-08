"""
Explanation Engine Module.
Synthesizes transparent, data-driven explanations and recommendations from candidate evaluation metrics.
"""

from typing import List, Dict, Any
from modules.models.result import CandidateScreeningResult, SkillMatchItem, EvidenceItem
from modules.models.candidate import CandidateProfile
from modules.models.job import JobRequirements


class ExplanationEngine:
    """Generates human-readable, auditable explanations for candidate screening outcomes."""

    @classmethod
    def generate_explanation(
        cls,
        candidate_profile: CandidateProfile,
        job_requirements: JobRequirements,
        matched_skills: List[SkillMatchItem],
        partial_skills: List[SkillMatchItem],
        missing_skills: List[SkillMatchItem],
        evidence_items: List[EvidenceItem],
        overall_score: float,
        recommendation: str,
    ) -> str:
        """
        Builds a comprehensive natural-language explanation derived from actual evaluation metrics.

        Args:
            candidate_profile: Candidate profile.
            job_requirements: Job requirements.
            matched_skills: Matched skills list.
            partial_skills: Partially matched skills list.
            missing_skills: Missing skills list.
            evidence_items: Evidence items list.
            overall_score: Overall score (0-100).
            recommendation: Recommendation string.

        Returns:
            Structured multi-paragraph explanation string.
        """
        matched_names = [m.job_skill for m in matched_skills]
        partial_names = [p.job_skill for p in partial_skills]
        missing_names = [m.job_skill for m in missing_skills]

        high_evidence = [e for e in evidence_items if e.evidence_level >= 2]
        low_evidence = [e for e in evidence_items if e.evidence_level == 1]

        paragraphs = []

        # 1. Summary statement
        paragraphs.append(
            f"Candidate achieved an overall evaluation score of {overall_score:.1f}% "
            f"resulting in a recommendation of '{recommendation}'."
        )

        # 2. Skill Alignment Breakdown
        if matched_names:
            matched_str = ", ".join(matched_names[:6])
            paragraphs.append(
                f"Demonstrates strong technical alignment with key required skills including {matched_str}."
            )
        else:
            paragraphs.append("No primary required skills demonstrated strong direct alignment.")

        if partial_names:
            partial_str = ", ".join(partial_names[:4])
            paragraphs.append(f"Partially aligned competencies detected in {partial_str} through semantic similarity.")

        # 3. Evidence Validation Breakdown
        if high_evidence:
            top_ev = high_evidence[:3]
            ev_summaries = [f"{ev.skill_name} (in {ev.source_section})" for ev in top_ev]
            paragraphs.append(
                f"Skill claims are supported by concrete contextual evidence in projects/experience for: {', '.join(ev_summaries)}."
            )
        elif low_evidence:
            paragraphs.append(
                "Several skills were mentioned in the skills listing but lack extensive corroborating project or work experience details."
            )

        # 4. Gap Analysis
        if missing_names:
            miss_str = ", ".join(missing_names[:5])
            paragraphs.append(f"Gaps identified: no significant evidence or mention found for {miss_str}.")

        # 5. Experience Alignment
        cand_exp = candidate_profile.experience_years
        req_exp = job_requirements.experience_years
        if req_exp > 0:
            if cand_exp >= req_exp:
                paragraphs.append(
                    f"Candidate possesses {cand_exp:.1f} years of professional experience, meeting or exceeding the {req_exp:.1f} years required."
                )
            else:
                paragraphs.append(
                    f"Candidate possesses {cand_exp:.1f} years of relevant experience compared to the requested {req_exp:.1f} years."
                )

        return " ".join(paragraphs)
