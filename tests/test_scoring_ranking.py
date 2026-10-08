"""
Tests for Scoring Engine and Ranking Engine.
"""

import unittest
from modules.scoring.scoring_engine import ScoringEngine
from modules.scoring.ranking_engine import RankingEngine
from modules.models.candidate import CandidateProfile
from modules.models.job import JobRequirements
from modules.models.result import CandidateScreeningResult, CandidateScoreBreakdown


class TestScoringRanking(unittest.TestCase):

    def test_scoring_engine_calculation(self):
        profile = CandidateProfile(
            candidate_id="CAND-01",
            anonymized_id="ANON-01",
            skills=[{"normalized_skill": "Python"}, {"normalized_skill": "Docker"}],
            education=[{"degree": "Bachelor of Technology", "institution": "Tech Univ", "year": "2020"}],
            experience_years=4.0,
            projects=["Built AI system using Python and Docker."],
            certifications=["AWS Certified"],
        )
        job = JobRequirements(
            job_id="JOB-01",
            required_skills=["Python", "Docker"],
            experience_years=3.0,
        )

        overall, breakdown, rec = ScoringEngine.calculate_score(
            candidate_profile=profile,
            job_requirements=job,
            semantic_skill_score=90.0,
            evidence_score=85.0,
        )

        self.assertGreaterEqual(overall, 0.0)
        self.assertLessEqual(overall, 100.0)
        self.assertEqual(breakdown.semantic_skill_match_score, 90.0)
        self.assertIn(rec, ["HIGH RELEVANCE", "RELEVANT", "MODERATE RELEVANCE", "LOW RELEVANCE"])

    def test_ranking_engine_ordering(self):
        c1 = CandidateScreeningResult(
            candidate_id="C1", anonymized_id="A1", job_id="J1", candidate_name="Alice",
            overall_score=70.0,
            score_breakdown=CandidateScoreBreakdown(70, 70, 70, 70, 70, 70)
        )
        c2 = CandidateScreeningResult(
            candidate_id="C2", anonymized_id="A2", job_id="J1", candidate_name="Bob",
            overall_score=90.0,
            score_breakdown=CandidateScoreBreakdown(90, 90, 90, 90, 90, 90)
        )

        ranked = RankingEngine.rank_candidates([c1, c2])

        self.assertEqual(ranked[0].candidate_id, "C2")
        self.assertEqual(ranked[0].rank, 1)
        self.assertEqual(ranked[1].candidate_id, "C1")
        self.assertEqual(ranked[1].rank, 2)


if __name__ == "__main__":
    unittest.main()
