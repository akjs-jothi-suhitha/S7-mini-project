"""
PDF Parser Module using PyMuPDF (fitz) with transparent fallback to other installed PDF extractors.
Extracts text from PDF resumes and job documents with error and scanned document detection.
"""

import io
from pathlib import Path
from typing import Union, BinaryIO
from modules.document_processing.file_validator import ValidationError
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PDFParser:
    """Extracts raw text content from PDF files using PyMuPDF / fallback libraries."""

    @staticmethod
    def extract_text(file_obj: Union[str, Path, BinaryIO, bytes, io.BytesIO]) -> str:
        """
        Extracts clean text from a PDF file.

        Args:
            file_obj: File path, stream, or bytes.

        Returns:
            Extracted text content as a single string.

        Raises:
            ValidationError: If PDF is invalid, corrupted, or scanned without extractable text.
        """
        full_text = ""
        # 1. Try PyMuPDF (fitz)
        try:
            import fitz  # PyMuPDF
            doc = None
            if isinstance(file_obj, (str, Path)):
                doc = fitz.open(str(file_obj))
            elif isinstance(file_obj, (bytes, bytearray)):
                doc = fitz.open(stream=file_obj, filetype="pdf")
            elif hasattr(file_obj, "read"):
                content = file_obj.read()
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                doc = fitz.open(stream=content, filetype="pdf")

            if doc:
                pages = [doc.load_page(i).get_text("text") for i in range(doc.page_count)]
                doc.close()
                full_text = "\n".join(pages).strip()
                if full_text:
                    return full_text
        except ImportError:
            pass
        except Exception as e:
            logger.warning("PyMuPDF failed: %s, attempting fallback parser.", e)

        # 2. Try pdfplumber
        try:
            import pdfplumber
            if isinstance(file_obj, (bytes, bytearray)):
                stream = io.BytesIO(file_obj)
            elif isinstance(file_obj, (str, Path)):
                stream = str(file_obj)
            else:
                content = file_obj.read()
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                stream = io.BytesIO(content)

            with pdfplumber.open(stream) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                full_text = "\n".join(pages).strip()
                if full_text:
                    return full_text
        except ImportError:
            pass
        except Exception as e:
            logger.warning("pdfplumber failed: %s, attempting fallback parser.", e)

        # 3. Try PyPDF2 / pypdf
        try:
            import pypdf
            reader_cls = pypdf.PdfReader
        except ImportError:
            try:
                import PyPDF2
                reader_cls = PyPDF2.PdfReader
            except ImportError:
                reader_cls = None

        if reader_cls:
            try:
                if isinstance(file_obj, (bytes, bytearray)):
                    stream = io.BytesIO(file_obj)
                elif isinstance(file_obj, (str, Path)):
                    stream = open(str(file_obj), "rb")
                else:
                    content = file_obj.read()
                    if hasattr(file_obj, "seek"):
                        file_obj.seek(0)
                    stream = io.BytesIO(content)

                reader = reader_cls(stream)
                pages = [page.extract_text() or "" for page in reader.pages]
                full_text = "\n".join(pages).strip()
                if full_text:
                    return full_text
            except Exception as e:
                logger.warning("PyPDF parser failed: %s", e)

        # 4. Try pypdfium2
        try:
            import pypdfium2 as pdfium
            if isinstance(file_obj, (str, Path)):
                pdf = pdfium.PdfDocument(str(file_obj))
            elif isinstance(file_obj, (bytes, bytearray)):
                pdf = pdfium.PdfDocument(file_obj)
            else:
                content = file_obj.read()
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                pdf = pdfium.PdfDocument(content)

            pages = []
            for i in range(len(pdf)):
                page = pdf[i]
                textpage = page.get_textpage()
                pages.append(textpage.get_text_range())
            full_text = "\n".join(pages).strip()
            if full_text:
                return full_text
        except ImportError:
            pass
        except Exception as e:
            logger.warning("pypdfium2 failed: %s", e)

        if not full_text:
            raise ValidationError(
                "No extractable text found in PDF. The document may be scanned, image-only, or corrupted."
            )

        return full_text
