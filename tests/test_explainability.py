"""
Tests for Explanation Engine.
"""

import unittest
from modules.explainability.explanation_engine import ExplanationEngine
from modules.models.candidate import CandidateProfile
from modules.models.job import JobRequirements
from modules.models.result import SkillMatchItem, EvidenceItem


class TestExplainability(unittest.TestCase):

    def test_explanation_generation(self):
        profile = CandidateProfile(
            candidate_id="CAND-01",
            anonymized_id="ANON-01",
            experience_years=4.0,
        )
        job = JobRequirements(
            job_id="JOB-01",
            experience_years=3.0,
        )
        matched = [SkillMatchItem(job_skill="Python", status="matched")]
        partial = []
        missing = [SkillMatchItem(job_skill="Kubernetes", status="missing")]
        evidence = [EvidenceItem(skill_name="Python", evidence_level=3, confidence_label="High", source_section="Projects", snippet="Built pipeline.")]

        explanation = ExplanationEngine.generate_explanation(
            candidate_profile=profile,
            job_requirements=job,
            matched_skills=matched,
            partial_skills=partial,
            missing_skills=missing,
            evidence_items=evidence,
            overall_score=85.5,
            recommendation="HIGH RELEVANCE",
        )

        self.assertIn("85.5%", explanation)
        self.assertIn("HIGH RELEVANCE", explanation)
        self.assertIn("Python", explanation)
        self.assertIn("Kubernetes", explanation)


if __name__ == "__main__":
    unittest.main()
