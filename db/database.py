import re
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

import config
from .models import Base, Message


def _msg_to_dict(msg: Message) -> dict:
    """Convert a Message ORM object to a plain dict (avoids detached session issues)."""
    return {
        "id": msg.id,
        "message_id": msg.message_id,
        "chat_id": msg.chat_id,
        "user_id": msg.user_id,
        "username": msg.username,
        "first_name": msg.first_name,
        "last_name": msg.last_name,
        "text": msg.text,
        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
        "timestamp_unix": msg.timestamp_unix,
        "reply_to_message_id": msg.reply_to_message_id,
        "is_forwarded": msg.is_forwarded,
        "forward_from": msg.forward_from,
        "display_name": msg.display_name,
        "formatted_date": msg.formatted_date,
    }


class Database:
    """SQLite database for message metadata (read-only after indexing)."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def get_message_by_db_id(self, db_id: int) -> dict | None:
        """Get a single message by its internal DB id. Returns a plain dict."""
        with self.get_session() as session:
            msg = session.query(Message).filter(Message.id == db_id).first()
            return _msg_to_dict(msg) if msg else None

    def get_messages_by_db_ids(self, db_ids: list[int]) -> list[dict]:
        """Get multiple messages by DB ids. Returns plain dicts."""
        with self.get_session() as session:
            msgs = session.query(Message).filter(Message.id.in_(db_ids)).all()
            return [_msg_to_dict(m) for m in msgs]

    def get_messages_around(
        self,
        chat_id: int,
        timestamp: datetime,
        before: int = 2,
        after: int = 2,
    ) -> tuple[list[dict], list[dict]]:
        """Get messages before and after a timestamp within the same chat. Returns plain dicts."""
        with self.get_session() as session:
            before_msgs = (
                session.query(Message)
                .filter(Message.chat_id == chat_id, Message.timestamp < timestamp)
                .order_by(Message.timestamp.desc())
                .limit(before)
                .all()
            )
            after_msgs = (
                session.query(Message)
                .filter(Message.chat_id == chat_id, Message.timestamp > timestamp)
                .order_by(Message.timestamp.asc())
                .limit(after)
                .all()
            )
            return (
                [_msg_to_dict(m) for m in reversed(before_msgs)],
                [_msg_to_dict(m) for m in after_msgs],
            )

    def execute_safe_sql(self, sql: str) -> list[dict]:
        """Execute a read-only SQL query with timeout. Only SELECT/WITH allowed."""
        # Strip comments and trailing semicolons
        cleaned = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        cleaned = re.sub(r"--.*$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip().rstrip(";").strip()
        first_word = cleaned.split()[0].upper() if cleaned else ""

        if first_word not in ("SELECT", "WITH"):
            raise ValueError("Only SELECT queries are allowed")

        # Wrap cleaned SQL to enforce result limit
        wrapped = f"SELECT * FROM ({cleaned}) LIMIT 50"

        def _run():
            with self.engine.connect() as conn:
                result = conn.execute(text(wrapped))
                columns = list(result.keys())
                return [dict(zip(columns, row)) for row in result.fetchall()]

        # Execute with timeout
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            try:
                return future.result(timeout=config.AGENT_QUERY_TIMEOUT)
            except FuturesTimeoutError:
                raise TimeoutError("Query took too long. Try a more specific search.")
