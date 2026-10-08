"""
Tests for PII Detector and Anonymizer.
"""

import unittest
from modules.document_processing.pii_detector import PIIDetector
from modules.document_processing.anonymizer import Anonymizer


class TestPIIAnonymizer(unittest.TestCase):

    def test_pii_detection(self):
        sample_text = """
        Johnathan Smith
        Email: jsmith.dev@testdomain.com
        Phone: +1-202-555-0143
        Address: 456 Elm Street, Seattle, WA 98101
        DOB: 15/06/1992
        LinkedIn: https://linkedin.com/in/johnsmith-ai
        """
        pii = PIIDetector.detect_pii(sample_text)

        self.assertIn("jsmith.dev@testdomain.com", pii["emails"])
        self.assertTrue(any("555" in p for p in pii["phones"]))
        self.assertTrue(any("linkedin.com" in u for u in pii["urls"]))
        self.assertTrue(any("Elm Street" in a for a in pii["addresses"]))
        self.assertTrue(len(pii["dates"]) > 0)

    def test_pii_anonymization(self):
        sample_text = """
        Jane Doe
        Contact: jane.doe@acmecorp.org
        Phone: (555) 234-5678
        Home: 123 Maple Avenue, Boston
        DOB: 01/01/1990
        """
        anonymized, counts = Anonymizer.anonymize_text(sample_text)

        self.assertNotIn("jane.doe@acmecorp.org", anonymized)
        self.assertIn("[ANONYMIZED_EMAIL]", anonymized)
        self.assertNotIn("(555) 234-5678", anonymized)
        self.assertIn("[ANONYMIZED_PHONE]", anonymized)
        self.assertIn("[ANONYMIZED_ADDRESS]", anonymized)
        self.assertIn("[ANONYMIZED_DATE]", anonymized)

    def test_anonymized_candidate_id_generation(self):
        cid = Anonymizer.generate_candidate_id()
        aid = Anonymizer.generate_anonymized_id()

        self.assertTrue(cid.startswith("CAND-"))
        self.assertTrue(aid.startswith("ANON-"))


if __name__ == "__main__":
    unittest.main()
