"""
Tests for Skill Evidence Extractor and Validator.
"""

import unittest
from modules.skill_validation.evidence_extractor import EvidenceExtractor
from modules.skill_validation.skill_validator import SkillValidator
from modules.models.candidate import CandidateProfile
from modules.models.result import SkillMatchItem


class TestSkillValidation(unittest.TestCase):

    def test_evidence_extraction_levels(self):
        cand_profile = CandidateProfile(
            candidate_id="CAND-01",
            anonymized_id="ANON-01",
            skills=[
                {"skill_name": "Python", "normalized_skill": "Python", "category": "Languages"},
                {"skill_name": "Docker", "normalized_skill": "Docker", "category": "DevOps"},
            ],
            experience_entries=[
                "Architected backend microservices using Python and FastAPI for 3 years."
            ],
            projects=[
                "Automated Screening System: built using Python and Streamlit."
            ],
            certifications=[],
            cleaned_text="Architected backend microservices using Python and FastAPI for 3 years. Listed in skills: Docker",
        )

        # Python should have high evidence (Level 3)
        level_py, label_py, sec_py, _ = EvidenceExtractor.find_evidence_for_skill("Python", cand_profile)
        self.assertEqual(level_py, 3)
        self.assertTrue(label_py.startswith("High"))

        # Docker only in skills list (Level 1)
        level_doc, label_doc, sec_doc, _ = EvidenceExtractor.find_evidence_for_skill("Docker", cand_profile)
        self.assertEqual(level_doc, 1)

    def test_skill_validator(self):
        matched = [
            SkillMatchItem(job_skill="Python", status="matched"),
            SkillMatchItem(job_skill="Docker", status="matched"),
        ]
        cand_profile = CandidateProfile(
            candidate_id="CAND-01",
            anonymized_id="ANON-01",
            skills=[
                {"skill_name": "Python", "normalized_skill": "Python", "category": "Languages"},
                {"skill_name": "Docker", "normalized_skill": "Docker", "category": "DevOps"},
            ],
            experience_entries=["Developed Python web services."],
            cleaned_text="Developed Python web services. Skills: Docker",
        )

        (
            en_matched,
            en_partial,
            en_missing,
            ev_items,
            strong,
            weak,
            ev_score,
        ) = SkillValidator.validate_skills(matched, [], [], cand_profile)

        self.assertGreater(ev_score, 0.0)
        self.assertEqual(len(ev_items), 2)
        self.assertGreater(len(strong), 0)


if __name__ == "__main__":
    unittest.main()
