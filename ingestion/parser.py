"""
Telegram chat export parser.
Handles large JSON files using streaming to avoid memory issues.
"""
import json
import ijson
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass

import config


@dataclass
class ParsedMessage:
    """Structured message from Telegram export."""
    message_id: int
    user_id: Optional[int]
    username: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    text: str
    timestamp: datetime
    timestamp_unix: int
    reply_to_message_id: Optional[int]
    # Forwarded message metadata
    is_forwarded: bool = False
    forward_from: Optional[str] = None
    forward_date: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dict for database insertion."""
        return {
            "message_id": self.message_id,
            "user_id": self.user_id,
            "username": self.username,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "text": self.text,
            "timestamp": self.timestamp,
            "timestamp_unix": self.timestamp_unix,
            "reply_to_message_id": self.reply_to_message_id,
            "is_forwarded": self.is_forwarded,
            "forward_from": self.forward_from,
            "forward_date": self.forward_date,
        }


@dataclass
class ChatMetadata:
    """Chat metadata from export."""
    chat_id: int
    chat_name: str
    chat_type: str


class TelegramExportParser:
    """Parse Telegram JSON exports with streaming support."""

    def __init__(self, export_path: Path):
        self.export_path = export_path
        self.result_json = export_path / "result.json"

        if not self.result_json.exists():
            raise FileNotFoundError(f"result.json not found in {export_path}")

    def get_metadata(self) -> ChatMetadata:
        """Extract chat metadata without loading full file."""
        with open(self.result_json, "r", encoding="utf-8") as f:
            # Read just the first part to get metadata
            parser = ijson.parse(f)
            chat_id = None
            chat_name = None
            chat_type = None

            for prefix, event, value in parser:
                if prefix == "id" and chat_id is None:
                    chat_id = value
                elif prefix == "name" and chat_name is None:
                    chat_name = value
                elif prefix == "type" and chat_type is None:
                    chat_type = value

                # Stop once we have all metadata
                if all([chat_id, chat_name, chat_type]):
                    break

        return ChatMetadata(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type
        )

    def _extract_text(self, text_field) -> str:
        """
        Extract plain text from Telegram's text field.
        Can be string or list of text entities.
        """
        if text_field is None:
            return ""

        if isinstance(text_field, str):
            return text_field

        if isinstance(text_field, list):
            parts = []
            for item in text_field:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    # Text entity with formatting
                    parts.append(item.get("text", ""))
            return "".join(parts)

        return str(text_field)

    def _parse_message(self, msg: dict) -> Optional[ParsedMessage]:
        """Parse a single message dict into ParsedMessage."""
        # Skip service messages (joins, leaves, etc.)
        if msg.get("type") != "message":
            return None

        # Skip messages without text
        text = self._extract_text(msg.get("text"))
        if not text or not text.strip():
            return None

        # Don't truncate - long messages will be chunked during embedding

        # Parse user info
        from_field = msg.get("from")
        from_id = msg.get("from_id")

        # Handle different from_id formats
        user_id = None
        if from_id:
            if isinstance(from_id, str) and from_id.startswith("user"):
                user_id = int(from_id.replace("user", ""))
            elif isinstance(from_id, int):
                user_id = from_id

        # Parse timestamp
        date_str = msg.get("date")
        timestamp = datetime.fromisoformat(date_str) if date_str else None
        timestamp_unix = msg.get("date_unixtime")
        if timestamp_unix:
            timestamp_unix = int(timestamp_unix)

        # Parse reply
        reply_to = msg.get("reply_to_message_id")

        # Detect forwarded messages
        is_forwarded = False
        forward_from = None
        forward_date = None

        # Check for forwarded_from (Telegram export format)
        forwarded_from = msg.get("forwarded_from")
        if forwarded_from:
            is_forwarded = True
            forward_from = forwarded_from

        # Alternative format: forward_from
        if not is_forwarded:
            fwd_from = msg.get("forward_from")
            if fwd_from:
                is_forwarded = True
                forward_from = fwd_from

        # Check for forward_date
        fwd_date_str = msg.get("forward_date")
        if fwd_date_str:
            is_forwarded = True
            try:
                forward_date = datetime.fromisoformat(fwd_date_str)
            except (ValueError, TypeError):
                pass

        return ParsedMessage(
            message_id=msg.get("id"),
            user_id=user_id,
            username=from_field if isinstance(from_field, str) else None,
            first_name=None,  # Not always in export
            last_name=None,
            text=text.strip(),
            timestamp=timestamp,
            timestamp_unix=timestamp_unix,
            reply_to_message_id=reply_to,
            is_forwarded=is_forwarded,
            forward_from=forward_from,
            forward_date=forward_date,
        )

    def stream_messages(self) -> Iterator[ParsedMessage]:
        """
        Stream messages from export file.
        Memory-efficient for large files.
        """
        with open(self.result_json, "r", encoding="utf-8") as f:
            # Stream through messages array
            messages = ijson.items(f, "messages.item")

            for msg in messages:
                parsed = self._parse_message(msg)
                if parsed:
                    yield parsed

    def count_messages(self) -> int:
        """Count total messages (for progress tracking)."""
        count = 0
        with open(self.result_json, "r", encoding="utf-8") as f:
            messages = ijson.items(f, "messages.item")
            for msg in messages:
                if msg.get("type") == "message" and msg.get("text"):
                    count += 1
        return count

    def parse_all(self) -> list[ParsedMessage]:
        """
        Parse all messages into memory.
        Use only for smaller exports or when needed.
        """
        return list(self.stream_messages())


def find_exports(base_dir: Path = None) -> list[Path]:
    """Find all export directories containing result.json."""
    if base_dir is None:
        base_dir = config.CHAT_EXPORTS_DIR

    exports = []
    for path in base_dir.iterdir():
        if path.is_dir() and (path / "result.json").exists():
            exports.append(path)

    return sorted(exports)
