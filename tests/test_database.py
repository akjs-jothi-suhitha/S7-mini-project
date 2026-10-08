"""
Tests for Database persistence module.
"""

import tempfile
import os
import unittest
import database


class TestDatabase(unittest.TestCase):

    def test_database_lifecycle(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        try:
            # Initialize
            database.init_db(temp_db)

            # Save candidate
            database.save_candidate("CAND-TEST", "ANON-TEST", "resume.pdf", ["email", "phone"], db_path=temp_db)

            # Save profile
            database.save_candidate_profile(
                "CAND-TEST", "ANON-TEST",
                education="B.Tech CS", experience_years=3.0,
                certifications=["AWS"], projects=["AI App"], technical_tools=["Docker"],
                cleaned_text="Sample", anonymized_text="Sample Anon",
                db_path=temp_db
            )

            # Save Job
            database.save_job_description(
                "JOB-TEST", "ML Engineer", "Raw text",
                required_skills=["Python", "SQL"], preferred_skills=["AWS"],
                education_req="B.Tech", experience_req=3.0,
                db_path=temp_db
            )

            # Save Screening Result
            database.save_screening_result(
                candidate_id="CAND-TEST",
                anonymized_id="ANON-TEST",
                job_id="JOB-TEST",
                rank=1,
                overall_score=88.5,
                semantic_score=90.0,
                evidence_score=85.0,
                experience_score=90.0,
                education_score=90.0,
                projects_score=80.0,
                recommendation="HIGH RELEVANCE",
                explanation="Strong candidate",
                db_path=temp_db,
            )

            results = database.get_screening_results_for_job("JOB-TEST", db_path=temp_db)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["overall_score"], 88.5)
            self.assertEqual(results[0]["recommendation"], "HIGH RELEVANCE")

        finally:
            if os.path.exists(temp_db):
                os.remove(temp_db)


if __name__ == "__main__":
    unittest.main()
