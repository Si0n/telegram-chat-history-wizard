from datetime import datetime
from sqlalchemy import (
    Column, Integer, BigInteger, String, Text, DateTime,
    ForeignKey, Index, Boolean, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class UserAlias(Base):
    """User nickname/alias mapping for natural language queries."""
    __tablename__ = "user_aliases"

    id = Column(Integer, primary_key=True)
    user_id = Column(BigInteger, nullable=False)
    username = Column(String(255), nullable=False)  # Actual username
    alias = Column(String(255), nullable=False)     # Nickname (lowercase for matching)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("alias", name="uq_alias"),
        Index("idx_alias_user_id", "user_id"),
        Index("idx_alias_alias", "alias"),
    )


class Export(Base):
    """Track processed chat exports for deduplication."""
    __tablename__ = "exports"

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    chat_id = Column(BigInteger, nullable=False)
    chat_name = Column(String(255))
    message_count = Column(Integer, default=0)
    min_message_id = Column(BigInteger)
    max_message_id = Column(BigInteger)
    date_range_start = Column(DateTime)
    date_range_end = Column(DateTime)
    processed_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="export")


class Message(Base):
    """Indexed messages from chat exports."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    message_id = Column(BigInteger, nullable=False)  # Telegram's message ID
    chat_id = Column(BigInteger, nullable=False)     # Chat ID (for uniqueness across chats)
    export_id = Column(Integer, ForeignKey("exports.id"), nullable=False)

    # User info
    user_id = Column(BigInteger)
    username = Column(String(255))
    first_name = Column(String(255))
    last_name = Column(String(255))

    # Message content
    text = Column(Text)
    timestamp = Column(DateTime, nullable=False)
    timestamp_unix = Column(BigInteger)

    # Threading
    reply_to_message_id = Column(BigInteger)

    # Vector reference
    vector_id = Column(String(255))  # ChromaDB document ID
    is_embedded = Column(Boolean, default=False)

    export = relationship("Export", back_populates="messages")

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
        """Get best available display name for user."""
        if self.username:
            return f"@{self.username}"
        parts = []
        if self.first_name:
            parts.append(self.first_name)
        if self.last_name:
            parts.append(self.last_name)
        return " ".join(parts) if parts else f"User {self.user_id}"

    @property
    def formatted_date(self) -> str:
        """Format timestamp for display."""
        if self.timestamp:
            return self.timestamp.strftime("%d.%m.%Y %H:%M")
        return "Unknown date"
