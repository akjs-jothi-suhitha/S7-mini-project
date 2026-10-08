"""
Tests for Text Cleaner and Document Processor.
"""

import unittest
from modules.document_processing.text_cleaner import TextCleaner
from modules.document_processing.document_processor import DocumentProcessor
from modules.models.candidate import RawCandidateDoc


class TestDocumentProcessing(unittest.TestCase):

    def test_text_cleaner_preserves_tech_terms(self):
        raw = "Skills: • C++ • C# • .NET • Node.js • Scikit-learn • CI/CD • RESTful APIs"
        cleaned = TextCleaner.clean(raw)

        self.assertIn("C++", cleaned)
        self.assertIn("C#", cleaned)
        self.assertIn(".NET", cleaned)
        self.assertIn("Node.js", cleaned)
        self.assertIn("Scikit-learn", cleaned)

    def test_document_processor_txt(self):
        sample_resume = """
        Alex Turner
        Email: alex@example.com | Phone: 555-019-2834
        
        Skills: Python, Machine Learning, Docker, SQL
        Experience: 4 years of experience as ML engineer.
        """
        doc: RawCandidateDoc = DocumentProcessor.process_file(
            sample_resume.encode("utf-8"), filename="test_resume.txt"
        )

        self.assertTrue(doc.candidate_id.startswith("CAND-"))
        self.assertTrue(doc.anonymized_id.startswith("ANON-"))
        self.assertNotIn("alex@example.com", doc.anonymized_text)
        self.assertIn("[ANONYMIZED_EMAIL]", doc.anonymized_text)


if __name__ == "__main__":
    unittest.main()
