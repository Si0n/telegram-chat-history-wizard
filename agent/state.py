import time
from dataclasses import dataclass, field

import config

# Module-level for easy mocking in tests
STATE_TTL_MINUTES = config.STATE_TTL_MINUTES
STATE_MAX_CONCURRENT = config.STATE_MAX_CONCURRENT


@dataclass
class SearchState:
    all_results: list[dict]
    original_query: str
    current_page: int = 0
    sort_order: str = "asc"
    highlight_terms: list[str] = field(default_factory=list)
    explanation: str = ""
    last_accessed: float = field(default_factory=time.time)


@dataclass
class DialogueState:
    anchor_message_id: int
    anchor_chat_id: int
    current_window: list[dict]
    highlight_terms: list[str] = field(default_factory=list)
    saved_search_state: "SearchState | None" = None
    last_accessed: float = field(default_factory=time.time)


class StateManager:
    def __init__(self):
        self._states: dict[tuple, SearchState | DialogueState] = {}

    def _evict_expired(self):
        now = time.time()
        ttl = STATE_TTL_MINUTES * 60
        expired = [k for k, v in self._states.items() if now - v.last_accessed > ttl]
        for k in expired:
            del self._states[k]

    def _evict_lru(self):
        if len(self._states) >= STATE_MAX_CONCURRENT:
            oldest = min(self._states, key=lambda k: self._states[k].last_accessed)
            del self._states[oldest]

    def set(self, chat_id: int, message_id: int, state: SearchState | DialogueState):
        self._evict_expired()
        self._evict_lru()
        self._states[(chat_id, message_id)] = state

    def get(self, chat_id: int, message_id: int) -> SearchState | DialogueState | None:
        self._evict_expired()
        key = (chat_id, message_id)
        state = self._states.get(key)
        if state:
            state.last_accessed = time.time()
        return state
