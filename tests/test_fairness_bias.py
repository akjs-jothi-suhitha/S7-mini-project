"""
Tests for Bias Reduction Pipeline and Fairness Auditor.
"""

import unittest
from modules.fairness.bias_reduction import BiasReductionPipeline
from modules.fairness.fairness_metrics import FairnessAuditor
from modules.models.candidate import CandidateProfile
from modules.models.result import CandidateScreeningResult, CandidateScoreBreakdown


class TestFairnessBias(unittest.TestCase):

    def test_bias_reduction_sanitizes_pii(self):
        profile = CandidateProfile(
            candidate_id="CAND-001",
            anonymized_id="ANON-001",
            candidate_name="John Doe",
            skills=[{"normalized_skill": "Python"}],
            experience_years=3.5,
        )
        sanitized = BiasReductionPipeline.sanitize_for_scoring(profile)

        self.assertNotIn("candidate_name", sanitized)
        self.assertNotIn("email", sanitized)
        self.assertNotIn("phone", sanitized)
        self.assertIn("Python", sanitized["skills"])

    def test_fairness_auditor(self):
        res1 = CandidateScreeningResult(
            candidate_id="CAND-A1",
            anonymized_id="ANON-A1",
            job_id="JOB-1",
            candidate_name="Candidate 1",
            overall_score=85.0,
        )
        res2 = CandidateScreeningResult(
            candidate_id="CAND-B2",
            anonymized_id="ANON-B2",
            job_id="JOB-1",
            candidate_name="Candidate 2",
            overall_score=60.0,
        )

        metrics = FairnessAuditor.audit_batch_fairness([res1, res2])

        self.assertEqual(metrics["total_candidates_audited"], 2)
        self.assertIn("score_mean", metrics)
        self.assertIn("selection_rate_pct", metrics)


if __name__ == "__main__":
    unittest.main()
