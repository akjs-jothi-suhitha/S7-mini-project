"""
Document Processor Module.
Coordinates file validation, text extraction (PDF/DOCX), cleaning, PII detection, and anonymization.
"""

from pathlib import Path
from typing import Union, BinaryIO, Tuple
import io

from modules.document_processing.file_validator import FileValidator, ValidationError
from modules.document_processing.pdf_parser import PDFParser
from modules.document_processing.docx_parser import DocxParser
from modules.document_processing.text_cleaner import TextCleaner
from modules.document_processing.pii_detector import PIIDetector
from modules.document_processing.anonymizer import Anonymizer
from modules.models.candidate import RawCandidateDoc
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """End-to-end ingestion and preprocessing for resume and job description documents."""

    @classmethod
    def process_file(
        cls,
        file_obj: Union[str, Path, BinaryIO, bytes, io.BytesIO],
        filename: str = "",
        candidate_id: str = None,
        anonymized_id: str = None,
    ) -> RawCandidateDoc:
        """
        Validates, extracts, cleans, detects PII, and anonymizes an uploaded file.

        Args:
            file_obj: File path, stream, or bytes.
            filename: Name of the file (required for streams/bytes).
            candidate_id: Optional existing ID.
            anonymized_id: Optional existing anonymized ID.

        Returns:
            RawCandidateDoc object containing all text representations.
        """
        if isinstance(file_obj, (str, Path)):
            filename = filename or Path(file_obj).name

        # 1. Validate file
        FileValidator.validate_file(file_obj, filename=filename)

        ext = Path(filename).suffix.lstrip(".").lower()
        logger.info("Processing file: %s (Type: %s)", filename, ext.upper())

        # 2. Extract text according to format
        if ext == "pdf":
            raw_text = PDFParser.extract_text(file_obj)
        elif ext in ("docx", "doc"):
            raw_text = DocxParser.extract_text(file_obj)
        elif ext in ("txt", "md"):
            if isinstance(file_obj, (str, Path)):
                with open(file_obj, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
            elif isinstance(file_obj, bytes):
                raw_text = file_obj.decode("utf-8", errors="ignore")
            elif hasattr(file_obj, "read"):
                content = file_obj.read()
                raw_text = content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else str(content)
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
            else:
                raw_text = ""
        else:
            raise ValidationError(f"Unsupported file extension: .{ext}")

        if not raw_text.strip():
            raise ValidationError(f"No text could be extracted from '{filename}'.")

        # 3. Clean and normalize text
        cleaned_text = TextCleaner.clean(raw_text)

        # 4. Detect PII
        pii_entities = PIIDetector.detect_pii(cleaned_text)

        # 5. Generate Anonymized Representation
        anonymized_text, _ = Anonymizer.anonymize_text(cleaned_text, pii_entities)

        cand_id = candidate_id or Anonymizer.generate_candidate_id()
        anon_id = anonymized_id or Anonymizer.generate_anonymized_id()

        return RawCandidateDoc(
            candidate_id=cand_id,
            anonymized_id=anon_id,
            file_name=filename,
            file_type=ext,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            anonymized_text=anonymized_text,
            pii_entities=pii_entities,
        )
