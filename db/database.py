from datetime import datetime
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker, Session

from .models import Base, Export, Message, UserAlias


class Database:
    """SQLite database for message metadata and deduplication."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    # === Export Management ===

    def get_expected_chat_id(self) -> Optional[int]:
        """Get chat_id from first export (for validation)."""
        with self.get_session() as session:
            export = session.query(Export).first()
            return export.chat_id if export else None

    def create_export(
        self,
        filename: str,
        chat_id: int,
        chat_name: str = None
    ) -> Export:
        """Create a new export record."""
        with self.get_session() as session:
            export = Export(
                filename=filename,
                chat_id=chat_id,
                chat_name=chat_name,
                processed_at=datetime.utcnow()
            )
            session.add(export)
            session.commit()
            session.refresh(export)
            return export

    def update_export_stats(self, export_id: int):
        """Update export statistics after processing."""
        with self.get_session() as session:
            export = session.get(Export, export_id)
            if not export:
                return

            stats = session.query(
                func.count(Message.id).label("count"),
                func.min(Message.message_id).label("min_id"),
                func.max(Message.message_id).label("max_id"),
                func.min(Message.timestamp).label("min_date"),
                func.max(Message.timestamp).label("max_date"),
            ).filter(Message.export_id == export_id).first()

            export.message_count = stats.count or 0
            export.min_message_id = stats.min_id
            export.max_message_id = stats.max_id
            export.date_range_start = stats.min_date
            export.date_range_end = stats.max_date
            session.commit()

    def list_exports(self) -> list[Export]:
        """List all processed exports."""
        with self.get_session() as session:
            return session.query(Export).order_by(Export.processed_at.desc()).all()

    # === Message Management ===

    def message_exists(self, message_id: int, chat_id: int = None) -> bool:
        """Check if message already indexed (for deduplication)."""
        with self.get_session() as session:
            query = session.query(Message).filter(Message.message_id == message_id)
            if chat_id:
                query = query.filter(Message.chat_id == chat_id)
            return session.query(query.exists()).scalar()

    def get_existing_message_ids(self, message_ids: list[int]) -> set[int]:
        """Batch check which message IDs already exist."""
        with self.get_session() as session:
            existing = session.query(Message.message_id).filter(
                Message.message_id.in_(message_ids)
            ).all()
            return {m.message_id for m in existing}

    def bulk_insert_messages(self, messages: list[dict], export_id: int) -> int:
        """Insert multiple messages, returns count inserted."""
        if not messages:
            return 0

        with self.get_session() as session:
            msg_objects = [
                Message(export_id=export_id, **msg) for msg in messages
            ]
            session.bulk_save_objects(msg_objects)
            session.commit()
            return len(msg_objects)

    def get_messages_without_embeddings(
        self,
        limit: int = 100,
        exclude_ids: set[int] = None
    ) -> list[Message]:
        """Get messages that haven't been embedded yet."""
        with self.get_session() as session:
            query = session.query(Message).filter(
                Message.is_embedded == False,
                Message.text.isnot(None),
                Message.text != ""
            )
            if exclude_ids:
                query = query.filter(Message.id.notin_(exclude_ids))
            return query.limit(limit).all()

    def mark_messages_embedded(self, message_ids: list[int], vector_ids: list[str]):
        """Mark messages as embedded with their vector IDs."""
        with self.get_session() as session:
            for msg_id, vec_id in zip(message_ids, vector_ids):
                session.query(Message).filter(Message.id == msg_id).update({
                    "is_embedded": True,
                    "vector_id": vec_id
                })
            session.commit()

    def get_message_by_telegram_id(self, message_id: int) -> Optional[Message]:
        """Get message by Telegram's message_id."""
        with self.get_session() as session:
            return session.query(Message).filter(
                Message.message_id == message_id
            ).first()

    def get_messages_by_ids(self, db_ids: list[int]) -> list[Message]:
        """Get messages by internal DB IDs."""
        with self.get_session() as session:
            return session.query(Message).filter(Message.id.in_(db_ids)).all()

    def get_user_messages(
        self,
        user_id: int = None,
        username: str = None,
        limit: int = 100
    ) -> list[Message]:
        """Get messages from a specific user."""
        with self.get_session() as session:
            query = session.query(Message)
            if user_id:
                query = query.filter(Message.user_id == user_id)
            elif username:
                # Handle with or without @
                username = username.lstrip("@")
                query = query.filter(Message.username == username)
            return query.order_by(Message.timestamp).limit(limit).all()

    def get_thread_context(
        self,
        message_id: int,
        before: int = 5,
        after: int = 5
    ) -> list[Message]:
        """Get messages around a specific message for context."""
        with self.get_session() as session:
            target = session.query(Message).filter(
                Message.message_id == message_id
            ).first()
            if not target:
                return []

            # Get messages within range of the target message_id
            return session.query(Message).filter(
                Message.message_id >= message_id - before,
                Message.message_id <= message_id + after
            ).order_by(Message.message_id).all()

    # === Stats ===

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self.get_session() as session:
            total_messages = session.query(func.count(Message.id)).scalar()
            embedded_messages = session.query(func.count(Message.id)).filter(
                Message.is_embedded == True
            ).scalar()
            unique_users = session.query(func.count(func.distinct(Message.user_id))).scalar()
            date_range = session.query(
                func.min(Message.timestamp),
                func.max(Message.timestamp)
            ).first()

            return {
                "total_messages": total_messages or 0,
                "embedded_messages": embedded_messages or 0,
                "unique_users": unique_users or 0,
                "date_start": date_range[0],
                "date_end": date_range[1],
                "exports_count": session.query(func.count(Export.id)).scalar() or 0
            }

    # === User Alias Management ===

    def add_alias(self, user_id: int, username: str, alias: str) -> Optional[UserAlias]:
        """Add a nickname/alias for a user. Returns None if alias already exists."""
        with self.get_session() as session:
            existing = session.query(UserAlias).filter(
                UserAlias.alias == alias.lower()
            ).first()
            if existing:
                return None

            user_alias = UserAlias(
                user_id=user_id,
                username=username,
                alias=alias.lower()
            )
            session.add(user_alias)
            session.commit()
            session.refresh(user_alias)
            return user_alias

    def remove_alias(self, alias: str) -> bool:
        """Remove an alias. Returns True if removed, False if not found."""
        with self.get_session() as session:
            result = session.query(UserAlias).filter(
                UserAlias.alias == alias.lower()
            ).delete()
            session.commit()
            return result > 0

    def get_user_by_alias(self, alias: str) -> Optional[tuple[int, str]]:
        """Get (user_id, username) by alias. Returns None if not found."""
        with self.get_session() as session:
            user_alias = session.query(UserAlias).filter(
                UserAlias.alias == alias.lower()
            ).first()
            if user_alias:
                return (user_alias.user_id, user_alias.username)
            return None

    def get_aliases_for_user(self, user_id: int) -> list[str]:
        """Get all aliases for a specific user."""
        with self.get_session() as session:
            aliases = session.query(UserAlias).filter(
                UserAlias.user_id == user_id
            ).all()
            return [a.alias for a in aliases]

    def get_all_aliases(self) -> list[UserAlias]:
        """Get all aliases grouped by user."""
        with self.get_session() as session:
            return session.query(UserAlias).order_by(
                UserAlias.user_id, UserAlias.alias
            ).all()

    def get_aliases_dict(self) -> dict[str, tuple[int, str]]:
        """Get all aliases as dict: {alias: (user_id, username)}."""
        with self.get_session() as session:
            aliases = session.query(UserAlias).all()
            return {a.alias: (a.user_id, a.username) for a in aliases}

    def get_all_users(self) -> list[tuple[int, str]]:
        """Get all unique users from messages as (user_id, username)."""
        with self.get_session() as session:
            users = session.query(
                Message.user_id,
                Message.username
            ).distinct().all()
            return [(u.user_id, u.username) for u in users if u.user_id]

    # === User Statistics (Feature 8) ===

    def get_user_stats(self, limit: int = 10) -> list[dict]:
        """
        Get user statistics for the dashboard.

        Returns list of dicts with:
        - user_id, username, display_name
        - message_count
        - first_message_date, last_message_date
        """
        with self.get_session() as session:
            from sqlalchemy import desc

            results = session.query(
                Message.user_id,
                Message.username,
                func.count(Message.id).label("message_count"),
                func.min(Message.timestamp).label("first_message"),
                func.max(Message.timestamp).label("last_message")
            ).filter(
                Message.user_id.isnot(None)
            ).group_by(
                Message.user_id
            ).order_by(
                desc("message_count")
            ).limit(limit).all()

            stats = []
            for r in results:
                # Get display name (prefer username, fallback to user_id)
                display = r.username or f"User#{r.user_id}"
                stats.append({
                    "user_id": r.user_id,
                    "username": r.username,
                    "display_name": display,
                    "message_count": r.message_count,
                    "first_message": r.first_message,
                    "last_message": r.last_message
                })
            return stats

    def get_hourly_distribution(self) -> dict[int, int]:
        """
        Get message count by hour of day.

        Returns dict: {hour: count} for hours 0-23
        """
        with self.get_session() as session:
            # SQLite-specific: extract hour from timestamp
            # Using strftime for SQLite compatibility
            results = session.execute(
                """
                SELECT
                    CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                    COUNT(*) as count
                FROM messages
                WHERE timestamp IS NOT NULL
                GROUP BY hour
                ORDER BY hour
                """
            ).fetchall()

            # Initialize all hours with 0
            distribution = {h: 0 for h in range(24)}
            for row in results:
                if row[0] is not None:
                    distribution[row[0]] = row[1]

            return distribution

    def get_daily_distribution(self, days: int = 30) -> list[dict]:
        """
        Get message count by day for the last N days.

        Returns list of dicts: [{date: "YYYY-MM-DD", count: N}, ...]
        """
        with self.get_session() as session:
            results = session.execute(
                f"""
                SELECT
                    DATE(timestamp) as date,
                    COUNT(*) as count
                FROM messages
                WHERE timestamp >= DATE('now', '-{days} days')
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                """
            ).fetchall()

            return [{"date": row[0], "count": row[1]} for row in results if row[0]]

    def get_most_active_users_in_period(
        self,
        date_from: str = None,
        date_to: str = None,
        limit: int = 10
    ) -> list[dict]:
        """Get most active users in a date range."""
        with self.get_session() as session:
            from sqlalchemy import desc

            query = session.query(
                Message.user_id,
                Message.username,
                func.count(Message.id).label("message_count")
            ).filter(
                Message.user_id.isnot(None)
            )

            if date_from:
                query = query.filter(Message.timestamp >= date_from)
            if date_to:
                query = query.filter(Message.timestamp <= date_to)

            results = query.group_by(
                Message.user_id
            ).order_by(
                desc("message_count")
            ).limit(limit).all()

            return [{
                "user_id": r.user_id,
                "username": r.username or f"User#{r.user_id}",
                "message_count": r.message_count
            } for r in results]

    # === Conversation Threading (Feature 9) ===

    def get_reply_chain_backward(
        self,
        message_id: int,
        max_depth: int = 20
    ) -> list[Message]:
        """
        Walk the reply chain backward to find the root message.

        Returns messages in chronological order (oldest first).
        """
        with self.get_session() as session:
            chain = []
            current_id = message_id
            seen_ids = set()

            for _ in range(max_depth):
                if current_id in seen_ids:
                    break  # Prevent infinite loops
                seen_ids.add(current_id)

                msg = session.query(Message).filter(
                    Message.message_id == current_id
                ).first()

                if not msg:
                    break

                chain.append(msg)

                # Check if this message is a reply to another
                if not msg.reply_to_message_id:
                    break

                current_id = msg.reply_to_message_id

            # Return in chronological order (oldest first)
            return list(reversed(chain))

    def get_reply_chain_forward(
        self,
        message_id: int,
        max_results: int = 50
    ) -> list[Message]:
        """
        Find all direct and indirect replies to a message (BFS).

        Returns messages in chronological order.
        """
        with self.get_session() as session:
            replies = []
            queue = [message_id]
            seen_ids = {message_id}

            while queue and len(replies) < max_results:
                current_id = queue.pop(0)

                # Find all messages that reply to this one
                direct_replies = session.query(Message).filter(
                    Message.reply_to_message_id == current_id
                ).order_by(Message.message_id).all()

                for reply in direct_replies:
                    if reply.message_id not in seen_ids:
                        seen_ids.add(reply.message_id)
                        replies.append(reply)
                        queue.append(reply.message_id)

            # Sort by message_id for chronological order
            replies.sort(key=lambda m: m.message_id)
            return replies[:max_results]

    def get_full_thread(
        self,
        message_id: int,
        max_backward: int = 20,
        max_forward: int = 50
    ) -> dict:
        """
        Get the full conversation thread around a message.

        Returns dict with:
        - messages: All thread messages in chronological order
        - root_id: ID of the thread root message
        - participants: Set of unique usernames
        - duration_minutes: Thread duration in minutes
        """
        # Get backward chain (to root)
        backward = self.get_reply_chain_backward(message_id, max_backward)

        # Get root message ID
        root_id = backward[0].message_id if backward else message_id

        # Get forward chain from root
        forward = self.get_reply_chain_forward(root_id, max_forward)

        # Combine and deduplicate
        all_messages = {m.message_id: m for m in backward}
        for m in forward:
            if m.message_id not in all_messages:
                all_messages[m.message_id] = m

        # Sort by message_id
        messages = sorted(all_messages.values(), key=lambda m: m.message_id)

        # Calculate thread metadata
        participants = set()
        for m in messages:
            if m.username:
                participants.add(m.username)
            elif m.user_id:
                participants.add(f"User#{m.user_id}")

        # Calculate duration
        duration_minutes = 0
        if len(messages) >= 2:
            first = messages[0].timestamp
            last = messages[-1].timestamp
            if first and last:
                delta = last - first
                duration_minutes = int(delta.total_seconds() / 60)

        return {
            "messages": messages,
            "root_id": root_id,
            "participants": participants,
            "duration_minutes": duration_minutes,
            "message_count": len(messages)
        }
