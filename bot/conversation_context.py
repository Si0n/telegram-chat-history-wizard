"""
Track conversation context for follow-up questions.
"""
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConversationState:
    """State of a conversation for follow-up questions."""
    chat_id: int
    original_question: str
    search_query: str
    mentioned_users: list[tuple[int, str]]
    results: list[dict]
    bot_message_id: int
    created_at: float = field(default_factory=time.time)


class ConversationContext:
    """Track conversation context for follow-up questions."""

    TTL_SECONDS = 3600  # 1 hour

    def __init__(self):
        # Map: (chat_id, bot_message_id) -> ConversationState
        self._contexts: dict[tuple[int, int], ConversationState] = {}

    def store(
        self,
        chat_id: int,
        bot_message_id: int,
        original_question: str,
        search_query: str,
        mentioned_users: list[tuple[int, str]],
        results: list[dict]
    ) -> None:
        """Store conversation context."""
        self._cleanup_old()

        state = ConversationState(
            chat_id=chat_id,
            original_question=original_question,
            search_query=search_query,
            mentioned_users=mentioned_users,
            results=results,
            bot_message_id=bot_message_id
        )
        self._contexts[(chat_id, bot_message_id)] = state

    def get(self, chat_id: int, reply_to_message_id: int) -> Optional[ConversationState]:
        """Get conversation context by reply_to_message_id."""
        key = (chat_id, reply_to_message_id)
        state = self._contexts.get(key)

        if state and time.time() - state.created_at > self.TTL_SECONDS:
            del self._contexts[key]
            return None

        return state

    def _cleanup_old(self) -> None:
        """Remove expired contexts."""
        now = time.time()
        expired = [
            key for key, state in self._contexts.items()
            if now - state.created_at > self.TTL_SECONDS
        ]
        for key in expired:
            del self._contexts[key]

    def clear(self, chat_id: int = None) -> None:
        """Clear contexts, optionally for specific chat."""
        if chat_id is None:
            self._contexts.clear()
        else:
            to_remove = [k for k in self._contexts if k[0] == chat_id]
            for key in to_remove:
                del self._contexts[key]
