"""
Job Parser Module.
Parses job descriptions from direct text input, PDF, or DOCX files.
"""

from pathlib import Path
from typing import Union, BinaryIO
import io

from modules.document_processing.file_validator import FileValidator, ValidationError
from modules.document_processing.pdf_parser import PDFParser
from modules.document_processing.docx_parser import DocxParser
from modules.document_processing.text_cleaner import TextCleaner
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)


class JobParser:
    """Parses and cleans Job Descriptions from text or files."""

    @classmethod
    def parse(
        cls,
        input_data: Union[str, Path, BinaryIO, bytes, io.BytesIO],
        filename: str = "",
    ) -> str:
        """
        Parses raw job description input into clean normalized text.

        Args:
            input_data: Text string, file path, stream, or bytes.
            filename: Optional filename.

        Returns:
            Clean normalized job description text.
        """
        if isinstance(input_data, str) and not filename and len(input_data.strip()) > 0 and not Path(input_data).exists():
            # Direct text input
            cleaned = TextCleaner.clean(input_data)
            if not cleaned:
                raise ValidationError("Job description text is empty.")
            return cleaned

        # File input (Path, stream, bytes)
        if isinstance(input_data, (str, Path)) and Path(input_data).exists():
            filename = filename or Path(input_data).name

        ext = Path(filename).suffix.lstrip(".").lower() if filename else ""

        if ext == "pdf":
            raw_text = PDFParser.extract_text(input_data)
        elif ext in ("docx", "doc"):
            raw_text = DocxParser.extract_text(input_data)
        elif isinstance(input_data, str):
            raw_text = input_data
        elif isinstance(input_data, bytes):
            raw_text = input_data.decode("utf-8", errors="ignore")
        elif hasattr(input_data, "read"):
            content = input_data.read()
            raw_text = content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else str(content)
            if hasattr(input_data, "seek"):
                input_data.seek(0)
        else:
            raise ValidationError("Invalid job description input format.")

        cleaned = TextCleaner.clean(raw_text)
        if not cleaned:
            raise ValidationError("No readable text found in the provided job description.")

        return cleaned
