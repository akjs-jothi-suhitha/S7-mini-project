"""
File Validator Module.
Validates uploaded resume and job description files for type, extension, size, and integrity.
"""

import os
import io
from pathlib import Path
from typing import Union, BinaryIO
from config import MAX_FILE_SIZE_BYTES, ALLOWED_EXTENSIONS
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception raised when file validation fails."""
    pass


class FileValidator:
    """Validates files before ingestion."""

    @staticmethod
    def validate_file(
        file_obj: Union[str, Path, BinaryIO, bytes, io.BytesIO],
        filename: str = "",
    ) -> bool:
        """
        Validates file existence, size, extension, and non-emptiness.

        Args:
            file_obj: File path or file-like object / bytes.
            filename: Name of the file (required if file_obj is bytes/stream).

        Returns:
            True if valid.

        Raises:
            ValidationError: If any check fails.
        """
        if isinstance(file_obj, (str, Path)):
            path = Path(file_obj)
            if not path.exists():
                raise ValidationError(f"File does not exist: {path.name}")
            filename = filename or path.name
            size = path.stat().st_size
        elif isinstance(file_obj, (bytes, bytearray)):
            size = len(file_obj)
            if not filename:
                raise ValidationError("Filename is required for raw bytes validation.")
        elif hasattr(file_obj, "read"):
            # Stream / BytesIO
            if hasattr(file_obj, "name") and not filename:
                filename = file_obj.name
            if not filename:
                raise ValidationError("Filename could not be determined for stream.")
            
            # Check size
            if hasattr(file_obj, "size"):
                size = file_obj.size
            else:
                curr_pos = file_obj.tell() if hasattr(file_obj, "tell") else 0
                file_obj.seek(0, os.SEEK_END)
                size = file_obj.tell()
                file_obj.seek(curr_pos)
        else:
            raise ValidationError("Unsupported file object type.")

        # Check empty file
        if size == 0:
            raise ValidationError(f"File '{filename}' is empty.")

        # Check max size
        if size > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
            raise ValidationError(
                f"File '{filename}' exceeds maximum allowed size of {max_mb:.1f} MB (size: {size / (1024*1024):.2f} MB)."
            )

        # Check extension
        ext = Path(filename).suffix.lstrip(".").lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file format '.{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        return True
