from datetime import datetime

from db.database import Database


class DialogueWindow:
    """Fetch and navigate a window of messages around an anchor message."""

    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def _ts_from_unix(unix_ts: int) -> datetime:
        """Convert unix timestamp to UTC datetime for DB queries."""
        return datetime.utcfromtimestamp(unix_ts)

    def open(self, message_id: int) -> tuple[list[dict], int, int]:
        """
        Open a dialogue window around a message.

        Returns: (messages, anchor_id, anchor_chat_id)
        - messages: list of dicts, 2 before + anchor + 2 after
        """
        msg = self.db.get_message_by_db_id(message_id)  # Returns dict
        if not msg:
            return [], message_id, 0

        ts = datetime.fromisoformat(msg["timestamp"]) if msg["timestamp"] else None
        if not ts:
            return [msg], msg["id"], msg["chat_id"]

        before_msgs, after_msgs = self.db.get_messages_around(
            chat_id=msg["chat_id"],
            timestamp=ts,
            before=2,
            after=2,
        )

        window = before_msgs + [msg] + after_msgs  # All already dicts
        return window, msg["id"], msg["chat_id"]

    def scroll_back(self, chat_id: int, first_timestamp_unix: int) -> list[dict]:
        """
        Scroll back: fetch 3 earlier messages.

        The caller keeps the first message from the current window as overlap.
        """
        ts = self._ts_from_unix(first_timestamp_unix)
        earlier, _ = self.db.get_messages_around(
            chat_id=chat_id, timestamp=ts, before=3, after=0
        )
        return earlier

    def scroll_forward(self, chat_id: int, last_timestamp_unix: int) -> list[dict]:
        """
        Scroll forward: fetch 3 later messages.

        The caller keeps the last message from the current window as overlap.
        """
        ts = self._ts_from_unix(last_timestamp_unix)
        _, later = self.db.get_messages_around(
            chat_id=chat_id, timestamp=ts, before=0, after=3
        )
        return later

    def has_earlier(self, chat_id: int, first_timestamp_unix: int) -> bool:
        """Check if there are messages before the current window."""
        ts = self._ts_from_unix(first_timestamp_unix)
        earlier, _ = self.db.get_messages_around(
            chat_id=chat_id, timestamp=ts, before=1, after=0
        )
        return len(earlier) > 0

    def has_later(self, chat_id: int, last_timestamp_unix: int) -> bool:
        """Check if there are messages after the current window."""
        ts = self._ts_from_unix(last_timestamp_unix)
        _, later = self.db.get_messages_around(
            chat_id=chat_id, timestamp=ts, before=0, after=1
        )
        return len(later) > 0
