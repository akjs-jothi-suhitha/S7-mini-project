"""
Tests for Job Description Parser and Requirement Extractor.
"""

import unittest
from modules.job_processing.job_parser import JobParser
from modules.job_processing.requirement_extractor import RequirementExtractor


class TestJobProcessing(unittest.TestCase):

    def test_job_requirement_extraction(self):
        job_text = """
        Senior Data Engineer
        
        Requirements:
        - 4+ years of experience in data engineering.
        - Strong skills in Python, SQL, PostgreSQL, and Docker.
        - Bachelor's degree in Computer Science.
        
        Preferred Skills:
        - Experience with Kubernetes and AWS.
        
        Responsibilities:
        - Build scalable ETL data pipelines.
        """
        cleaned = JobParser.parse(job_text)
        reqs = RequirementExtractor.extract_requirements(cleaned, job_title="Senior Data Engineer")

        self.assertEqual(reqs.title, "Senior Data Engineer")
        self.assertGreaterEqual(reqs.experience_years, 4.0)
        self.assertTrue("Python" in reqs.required_skills or "Python" in reqs.job_keywords)
        self.assertTrue("SQL" in reqs.required_skills or "SQL" in reqs.job_keywords)
        self.assertGreater(len(reqs.responsibilities), 0)


if __name__ == "__main__":
    unittest.main()
