"""
Tests for Skill Extractor and Information Extractor.
"""

import unittest
from modules.nlp.skill_extractor import SkillExtractor
from modules.nlp.information_extractor import InformationExtractor
from modules.models.candidate import RawCandidateDoc


class TestNLPExtraction(unittest.TestCase):

    def test_skill_extraction(self):
        text = "Proficient in Python, SQL, Docker, React, and Machine Learning algorithms."
        skills = SkillExtractor.extract_skills(text)
        canon_names = [s["normalized_skill"] for s in skills]

        self.assertIn("Python", canon_names)
        self.assertIn("SQL", canon_names)
        self.assertIn("Docker", canon_names)
        self.assertIn("React", canon_names)
        self.assertIn("Machine Learning", canon_names)

    def test_information_extraction(self):
        sample_resume = """
        Jane Doe
        Email: jane@test.com
        
        EDUCATION
        Master of Science in Computer Science | Stanford University, 2020
        
        EXPERIENCE
        3.5 years of experience in AI Engineering.
        - Designed scalable deep learning architectures using PyTorch and FastAPI.
        
        PROJECTS
        - Transformer Based Question Answering System: Developed high throughput model.
        
        CERTIFICATIONS
        - AWS Certified Solutions Architect
        """
        raw_doc = RawCandidateDoc(
            candidate_id="CAND-001",
            anonymized_id="ANON-001",
            file_name="resume.txt",
            file_type="txt",
            raw_text=sample_resume,
            cleaned_text=sample_resume,
            anonymized_text=sample_resume,
            pii_entities={"names": ["Jane Doe"], "emails": ["jane@test.com"]},
        )

        profile = InformationExtractor.extract_profile(raw_doc)

        self.assertGreaterEqual(profile.experience_years, 3.0)
        self.assertGreater(len(profile.education), 0)
        self.assertTrue(any("master" in e["degree"].lower() for e in profile.education))
        self.assertGreater(len(profile.certifications), 0)
        self.assertGreater(len(profile.projects), 0)


if __name__ == "__main__":
    unittest.main()
