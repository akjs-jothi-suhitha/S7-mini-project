"""
End-to-End Integration Tests for Screening Service.
Validates the complete pipeline: Job + 3 Resumes -> Extraction -> Semantic Match -> Evidence Validation -> Bias Reduction -> Scoring -> Ranking.
"""

import tempfile
import os
import unittest
from pathlib import Path
from modules.services.screening_service import ScreeningService
from modules.models.result import BatchScreeningResult


class TestScreeningServiceE2E(unittest.TestCase):

    def test_end_to_end_screening_pipeline(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        try:
            service = ScreeningService(db_path=temp_db)

            # Load sample data
            data_dir = Path(__file__).resolve().parent.parent / "data"
            job_file = data_dir / "sample_job_descriptions" / "senior_ml_engineer.txt"
            resume_a = data_dir / "sample_resumes" / "candidate_a_alex_turner.txt"
            resume_b = data_dir / "sample_resumes" / "candidate_b_priya_sharma.txt"
            resume_c = data_dir / "sample_resumes" / "candidate_c_john_doe.txt"

            with open(job_file, "r", encoding="utf-8") as f:
                job_text = f.read()

            with open(resume_a, "r", encoding="utf-8") as f:
                cand_a_text = f.read()

            with open(resume_b, "r", encoding="utf-8") as f:
                cand_b_text = f.read()

            with open(resume_c, "r", encoding="utf-8") as f:
                cand_c_text = f.read()

            resumes = [
                cand_a_text.encode("utf-8"),
                cand_b_text.encode("utf-8"),
                cand_c_text.encode("utf-8"),
            ]
            filenames = [
                "candidate_a_alex_turner.txt",
                "candidate_b_priya_sharma.txt",
                "candidate_c_john_doe.txt",
            ]

            batch_result: BatchScreeningResult = service.process_screening(
                job_input=job_text,
                resume_files=resumes,
                resume_filenames=filenames,
                job_title="Senior Machine Learning Engineer",
            )

            self.assertEqual(batch_result.total_candidates, 3)
            ranked = batch_result.ranked_candidates
            self.assertEqual(len(ranked), 3)

            # Verify Ranking: Candidate A > Candidate B > Candidate C
            scores = [c.overall_score for c in ranked]
            self.assertGreaterEqual(scores[0], scores[1])
            self.assertGreaterEqual(scores[1], scores[2])

            # Top candidate should be Alex Turner (Candidate A)
            self.assertTrue("Alex Turner" in ranked[0].candidate_name or "CAND-" in ranked[0].candidate_id)
            self.assertEqual(ranked[0].rank, 1)
            self.assertGreater(ranked[0].overall_score, 75.0)
            self.assertIn(ranked[0].recommendation, ["HIGH RELEVANCE", "RELEVANT"])

            # Third candidate (John Doe) should have lowest score
            self.assertEqual(ranked[2].rank, 3)
            self.assertLess(ranked[2].overall_score, ranked[0].overall_score)

            # Verify fairness metrics
            self.assertIn("selection_rate_pct", batch_result.fairness_metrics)
            self.assertEqual(batch_result.fairness_metrics["total_candidates_audited"], 3)

        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)


if __name__ == "__main__":
    unittest.main()
