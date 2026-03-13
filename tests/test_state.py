import time
import pytest
from unittest.mock import patch
from agent.state import StateManager, SearchState, DialogueState


def test_set_and_get():
    sm = StateManager()
    state = SearchState(all_results=[{"id": 1}], original_query="test")
    sm.set(100, 200, state)
    got = sm.get(100, 200)
    assert got is state
    assert got.all_results == [{"id": 1}]


def test_get_missing_returns_none():
    sm = StateManager()
    assert sm.get(100, 200) is None


def test_ttl_eviction():
    sm = StateManager()
    state = SearchState(all_results=[], original_query="test")
    state.last_accessed = time.time() - 3600  # 1 hour ago
    sm._states[(100, 200)] = state

    # Accessing triggers eviction
    assert sm.get(100, 200) is None


def test_lru_eviction():
    with patch("agent.state.STATE_MAX_CONCURRENT", 2):
        sm = StateManager()
        s1 = SearchState(all_results=[], original_query="q1")
        s1.last_accessed = time.time() - 10
        sm._states[(1, 1)] = s1

        s2 = SearchState(all_results=[], original_query="q2")
        s2.last_accessed = time.time() - 5
        sm._states[(2, 2)] = s2

        # Adding a 3rd should evict the oldest (1,1)
        s3 = SearchState(all_results=[], original_query="q3")
        sm.set(3, 3, s3)

        assert sm.get(1, 1) is None
        assert sm.get(2, 2) is not None
        assert sm.get(3, 3) is not None


def test_get_updates_last_accessed():
    sm = StateManager()
    state = SearchState(all_results=[], original_query="test")
    old_time = time.time() - 100
    state.last_accessed = old_time
    sm._states[(100, 200)] = state

    sm.get(100, 200)
    assert state.last_accessed > old_time
