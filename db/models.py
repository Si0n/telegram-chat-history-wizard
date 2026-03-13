from datetime import datetime
from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, DateTime,
    Index, Boolean, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Message(Base):
    """Indexed messages from chat exports."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    message_id = Column(BigInteger, nullable=False)
    chat_id = Column(BigInteger, nullable=False)

    user_id = Column(BigInteger)
    username = Column(String(255))
    first_name = Column(String(255))
    last_name = Column(String(255))

    text = Column(Text)
    timestamp = Column(DateTime, nullable=False)
    timestamp_unix = Column(BigInteger)

    reply_to_message_id = Column(BigInteger)

    is_forwarded = Column(Boolean, default=False)
    forward_from = Column(String(255), nullable=True)
    forward_date = Column(DateTime, nullable=True)

    vector_id = Column(String(255))
    is_embedded = Column(Boolean, default=False)

    __table_args__ = (
        UniqueConstraint("message_id", "chat_id", name="uq_message_chat"),
        Index("idx_messages_user_id", "user_id"),
        Index("idx_messages_timestamp", "timestamp"),
        Index("idx_messages_message_id", "message_id"),
        Index("idx_messages_chat_id", "chat_id"),
        Index("idx_messages_reply_to", "reply_to_message_id"),
    )

    @property
    def display_name(self) -> str:
        if self.first_name:
            return self.first_name
        if self.username:
            return f"@{self.username}"
        return f"User {self.user_id}"

    @property
    def formatted_date(self) -> str:
        if self.timestamp:
            return self.timestamp.strftime("%d.%m.%Y %H:%M")
        return "Unknown date"
