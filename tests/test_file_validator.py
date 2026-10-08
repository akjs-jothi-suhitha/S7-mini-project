"""
Tests for file validator module.
"""

import io
import unittest
from modules.document_processing.file_validator import FileValidator, ValidationError


class TestFileValidator(unittest.TestCase):

    def test_file_validator_valid_bytes(self):
        valid_bytes = b"Sample resume content for validation testing."
        self.assertTrue(FileValidator.validate_file(valid_bytes, filename="resume.pdf"))
        self.assertTrue(FileValidator.validate_file(valid_bytes, filename="resume.docx"))

    def test_file_validator_empty_file(self):
        with self.assertRaises(ValidationError):
            FileValidator.validate_file(b"", filename="empty.pdf")

    def test_file_validator_invalid_extension(self):
        with self.assertRaises(ValidationError):
            FileValidator.validate_file(b"some content", filename="malicious.exe")

    def test_file_validator_stream(self):
        stream = io.BytesIO(b"Valid resume document in memory stream")
        stream.name = "candidate_resume.pdf"
        self.assertTrue(FileValidator.validate_file(stream))


if __name__ == "__main__":
    unittest.main()
