"""
End-to-End Integration Test with actual DOCX resume files and Job Description.
"""

import tempfile
import os
import unittest
from pathlib import Path
from modules.services.screening_service import ScreeningService
from modules.models.result import BatchScreeningResult


class TestDocxPDFScreeningE2E(unittest.TestCase):

    def test_docx_end_to_end_screening(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        try:
            service = ScreeningService(db_path=temp_db)

            data_dir = Path(__file__).resolve().parent.parent / "data"
            job_file = data_dir / "sample_job_descriptions" / "senior_ml_engineer.docx"
            resume_a = data_dir / "sample_resumes" / "candidate_a_alex_turner.docx"
            resume_b = data_dir / "sample_resumes" / "candidate_b_priya_sharma.docx"
            resume_c = data_dir / "sample_resumes" / "candidate_c_john_doe.docx"

            self.assertTrue(job_file.exists(), f"Missing {job_file}")
            self.assertTrue(resume_a.exists(), f"Missing {resume_a}")
            self.assertTrue(resume_b.exists(), f"Missing {resume_b}")
            self.assertTrue(resume_c.exists(), f"Missing {resume_c}")

            batch_result: BatchScreeningResult = service.process_screening(
                job_input=job_file,
                resume_files=[resume_a, resume_b, resume_c],
                job_title="Senior Machine Learning Engineer",
            )

            self.assertEqual(batch_result.total_candidates, 3)
            ranked = batch_result.ranked_candidates
            self.assertEqual(len(ranked), 3)

            # Check dynamic calculation
            self.assertGreater(ranked[0].overall_score, ranked[1].overall_score)
            self.assertGreater(ranked[1].overall_score, ranked[2].overall_score)

            # Ranks
            self.assertEqual(ranked[0].rank, 1)
            self.assertEqual(ranked[1].rank, 2)
            self.assertEqual(ranked[2].rank, 3)

            # Candidates
            self.assertIn("Alex Turner", ranked[0].candidate_name)
            self.assertEqual(ranked[0].recommendation, "HIGH RELEVANCE")
            self.assertEqual(ranked[2].recommendation, "LOW RELEVANCE")

        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)


if __name__ == "__main__":
    unittest.main()
