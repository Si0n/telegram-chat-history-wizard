"""
Handle upload and validation of new chat exports.
"""
import zipfile
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

import config
from db import Database


@dataclass
class ValidationResult:
    """Result of export validation."""
    valid: bool
    error: Optional[str] = None
    chat_id: Optional[int] = None
    chat_name: Optional[str] = None
    message_count: Optional[int] = None


class ExportUploader:
    """Handle new export uploads with validation."""

    def __init__(self, db: Database):
        self.db = db

    def validate_zip(self, zip_path: Path) -> ValidationResult:
        """
        Validate a zip file is a proper Telegram export.
        Returns ValidationResult with details.
        """
        # Check file exists
        if not zip_path.exists():
            return ValidationResult(valid=False, error="File not found")

        # Check it's a valid zip
        if not zipfile.is_zipfile(zip_path):
            return ValidationResult(valid=False, error="Not a valid ZIP file")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # List contents
                names = zf.namelist()

                # Find result.json (might be in root or subdirectory)
                result_json_path = None
                for name in names:
                    if name.endswith("result.json"):
                        result_json_path = name
                        break

                if not result_json_path:
                    return ValidationResult(
                        valid=False,
                        error="No result.json found in archive"
                    )

                # Read and parse result.json
                with zf.open(result_json_path) as f:
                    # Read first 10KB to get metadata
                    content = f.read(10240).decode("utf-8")

                    # Try to find the key fields in the beginning
                    # This is a quick check without parsing full JSON
                    if '"messages"' not in content and '"type"' not in content:
                        return ValidationResult(
                            valid=False,
                            error="Invalid Telegram export format"
                        )

                # Now parse properly to get chat_id
                with zf.open(result_json_path) as f:
                    data = json.load(f)

                chat_id = data.get("id")
                chat_name = data.get("name")
                messages = data.get("messages", [])

                if not chat_id:
                    return ValidationResult(
                        valid=False,
                        error="No chat ID found in export"
                    )

                # Check if chat_id matches existing exports
                expected_chat_id = self.db.get_expected_chat_id()
                if expected_chat_id and chat_id != expected_chat_id:
                    return ValidationResult(
                        valid=False,
                        error=f"Wrong chat! Expected chat_id: {expected_chat_id}, got: {chat_id}"
                    )

                return ValidationResult(
                    valid=True,
                    chat_id=chat_id,
                    chat_name=chat_name,
                    message_count=len(messages)
                )

        except zipfile.BadZipFile:
            return ValidationResult(valid=False, error="Corrupted ZIP file")
        except json.JSONDecodeError as e:
            return ValidationResult(valid=False, error=f"Invalid JSON: {e}")
        except Exception as e:
            return ValidationResult(valid=False, error=f"Validation error: {e}")

    def extract_export(self, zip_path: Path) -> Path:
        """
        Extract zip to chat_exports directory.
        Returns path to extracted directory.
        """
        # Generate unique directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extract_name = f"ChatExport_{timestamp}"
        extract_path = config.CHAT_EXPORTS_DIR / extract_name

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Check if contents are in a subdirectory
            names = zf.namelist()
            has_subdir = all("/" in name for name in names if name != names[0])

            if has_subdir:
                # Extract to temp, then move contents up
                temp_path = config.CHAT_EXPORTS_DIR / f"_temp_{timestamp}"
                zf.extractall(temp_path)

                # Find the actual export directory
                subdirs = [d for d in temp_path.iterdir() if d.is_dir()]
                if subdirs:
                    shutil.move(str(subdirs[0]), str(extract_path))
                    shutil.rmtree(temp_path)
                else:
                    shutil.move(str(temp_path), str(extract_path))
            else:
                # Extract directly
                extract_path.mkdir(parents=True, exist_ok=True)
                zf.extractall(extract_path)

        return extract_path

    async def process_upload(
        self,
        zip_path: Path,
        delete_zip_after: bool = True
    ) -> tuple[bool, str, Optional[dict]]:
        """
        Full upload processing pipeline.
        Returns (success, message, stats).
        """
        # Validate
        validation = self.validate_zip(zip_path)
        if not validation.valid:
            return False, f"❌ {validation.error}", None

        # Extract
        try:
            export_path = self.extract_export(zip_path)
        except Exception as e:
            return False, f"❌ Extraction failed: {e}", None

        # Import messages (will be handled by indexer)
        stats = {
            "chat_id": validation.chat_id,
            "chat_name": validation.chat_name,
            "estimated_messages": validation.message_count,
            "export_path": str(export_path)
        }

        # Cleanup
        if delete_zip_after and zip_path.exists():
            zip_path.unlink()

        return True, f"✅ Export extracted to {export_path.name}", stats
