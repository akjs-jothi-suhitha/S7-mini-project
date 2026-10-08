"""
Tests for Semantic Skill Matcher and Cosine Similarity.
"""

import unittest
from modules.semantic_matching.matcher import SemanticSkillMatcher
from modules.semantic_matching.similarity import compute_cosine_similarity
from modules.models.candidate import CandidateProfile
from modules.models.job import JobRequirements


class TestSemanticMatching(unittest.TestCase):

    def test_cosine_similarity(self):
        vec1 = [1.0, 0.0, 0.5]
        vec2 = [1.0, 0.0, 0.5]
        self.assertGreater(compute_cosine_similarity(vec1, vec2), 0.99)

    def test_semantic_skill_matcher(self):
        matcher = SemanticSkillMatcher()

        cand_profile = CandidateProfile(
            candidate_id="CAND-01",
            anonymized_id="ANON-01",
            skills=[
                {"skill_name": "Python", "normalized_skill": "Python", "category": "Languages"},
                {"skill_name": "PyTorch", "normalized_skill": "PyTorch", "category": "Frameworks"},
                {"skill_name": "NLP", "normalized_skill": "Natural Language Processing", "category": "AI"},
            ],
        )

        job_req = JobRequirements(
            job_id="JOB-01",
            title="ML Engineer",
            required_skills=["Python", "Natural Language Processing", "Docker"],
            preferred_skills=[],
        )

        matched, partial, missing, score = matcher.match_skills(cand_profile, job_req)

        matched_names = [m.job_skill for m in matched]
        missing_names = [m.job_skill for m in missing]

        self.assertIn("Python", matched_names)
        self.assertIn("Natural Language Processing", matched_names)
        self.assertIn("Docker", missing_names)
        self.assertGreater(score, 50.0)


if __name__ == "__main__":
    unittest.main()
