"""
Bias Reduction Module.
Enforces privacy masking, demographic feature removal, and ensures scoring functions receive only job-relevant attributes.
"""

from typing import Dict, Any, List
from modules.models.candidate import CandidateProfile, RawCandidateDoc
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BiasReductionPipeline:
    """
    Enforces bias mitigation protocols by stripping PII and validating that scoring engines
    operate strictly on anonymized, merit-based qualifications.
    """

    DISALLOWED_SCORING_FEATURES = {
        "candidate_name",
        "name",
        "email",
        "phone",
        "address",
        "gender",
        "age",
        "dob",
        "nationality",
        "photo",
        "marital_status",
    }

    @classmethod
    def sanitize_for_scoring(cls, profile: CandidateProfile) -> Dict[str, Any]:
        """
        Extracts only merit-based features for evaluation, isolating sensitive identity attributes.

        Args:
            profile: Extracted CandidateProfile.

        Returns:
            Sanitized feature dictionary containing strictly merit attributes.
        """
        # Ensure only skill, experience, education, and project features are returned
        return {
            "candidate_id": profile.candidate_id,
            "anonymized_id": profile.anonymized_id,
            "skills": [s["normalized_skill"] for s in profile.skills],
            "experience_years": profile.experience_years,
            "education_degrees": [e.get("degree", "") for e in profile.education],
            "certifications": profile.certifications,
            "projects": profile.projects,
            "technical_tools": profile.technical_tools,
        }

    @classmethod
    def get_bias_mitigation_statement(cls) -> Dict[str, Any]:
        """Returns standard academic disclosures on bias reduction methodology."""
        return {
            "methodology": "PII Masking & Anonymized Merit Evaluation",
            "masked_attributes": ["Name", "Email", "Phone", "Physical Address", "URLs/Social Links", "Date of Birth"],
            "disclaimer": (
                "PII detection and masking is a best-effort bias-reduction technique designed to mitigate "
                "unconscious cognitive bias during initial candidate screening. It does not guarantee complete "
                "elimination of systemic or societal bias."
            ),
            "scoring_basis": "Semantic skill alignment, verified contextual evidence, experience relevance, and project credentials.",
        }
