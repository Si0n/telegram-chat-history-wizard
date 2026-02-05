"""
Track conversation context for follow-up questions.
"""
import hashlib
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

    # Pagination state (Feature 2)
    all_results: list[dict] = field(default_factory=list)  # Full result set
    current_page: int = 1
    results_per_page: int = 5

    # Date filter state (Feature 5)
    active_date_filter: Optional[str] = None  # "today", "week", "month", "2024", etc.

    # Suggestions state (Feature 4)
    suggestions: list[str] = field(default_factory=list)
    suggestion_hashes: dict = field(default_factory=dict)  # hash -> full query

    @property
    def total_pages(self) -> int:
        """Calculate total pages based on all results."""
        if not self.all_results:
            return 1
        return max(1, (len(self.all_results) + self.results_per_page - 1) // self.results_per_page)

    def get_page_results(self, page: int = None) -> list[dict]:
        """Get results for current or specified page."""
        if page is None:
            page = self.current_page
        start = (page - 1) * self.results_per_page
        end = start + self.results_per_page
        return self.all_results[start:end]

    def add_suggestion(self, query: str) -> str:
        """Add a suggestion and return its hash."""
        h = hashlib.md5(query.encode()).hexdigest()[:8]
        self.suggestion_hashes[h] = query
        self.suggestions.append(query)
        return h

    def get_suggestion_by_hash(self, hash_str: str) -> Optional[str]:
        """Get suggestion query by hash."""
        return self.suggestion_hashes.get(hash_str)


class ConversationContext:
    """Track conversation context for follow-up questions."""

    TTL_SECONDS = 1800  # 30 minutes (reduced from 1 hour to save memory)
    MAX_CONTEXTS = 50   # Max stored contexts (prevent unbounded growth)
    MAX_RESULTS_PER_CONTEXT = 50  # Limit results stored per context

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
        results: list[dict],
        all_results: list[dict] = None,
        current_page: int = 1,
        active_date_filter: str = None
    ) -> ConversationState:
        """Store conversation context and return the state."""
        self._cleanup_old()

        # Limit stored results to save memory
        limited_all_results = (all_results if all_results is not None else results)
        if len(limited_all_results) > self.MAX_RESULTS_PER_CONTEXT:
            limited_all_results = limited_all_results[:self.MAX_RESULTS_PER_CONTEXT]

        state = ConversationState(
            chat_id=chat_id,
            original_question=original_question,
            search_query=search_query,
            mentioned_users=mentioned_users,
            results=results[:10] if results else [],  # Limit display results too
            bot_message_id=bot_message_id,
            all_results=limited_all_results,
            current_page=current_page,
            active_date_filter=active_date_filter
        )
        self._contexts[(chat_id, bot_message_id)] = state

        # Enforce max contexts limit (remove oldest if exceeded)
        if len(self._contexts) > self.MAX_CONTEXTS:
            oldest_key = min(self._contexts.keys(), key=lambda k: self._contexts[k].created_at)
            del self._contexts[oldest_key]

        return state

    def get(self, chat_id: int, reply_to_message_id: int) -> Optional[ConversationState]:
        """Get conversation context by reply_to_message_id."""
        key = (chat_id, reply_to_message_id)
        state = self._contexts.get(key)

        if state and time.time() - state.created_at > self.TTL_SECONDS:
            del self._contexts[key]
            return None

        return state

    def get_by_context_key(self, context_key: str) -> Optional[ConversationState]:
        """Get conversation context by context key (chat_id_msg_id format)."""
        try:
            chat_id, msg_id = context_key.split("_")
            return self.get(int(chat_id), int(msg_id))
        except (ValueError, AttributeError):
            return None

    def make_context_key(self, chat_id: int, msg_id: int) -> str:
        """Create a context key from chat_id and message_id."""
        return f"{chat_id}_{msg_id}"

    def update_state(self, state: ConversationState) -> None:
        """Update an existing state in the context store."""
        key = (state.chat_id, state.bot_message_id)
        self._contexts[key] = state

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
