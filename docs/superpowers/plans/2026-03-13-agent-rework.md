# Agent-Based Bot Rework — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the complex vector search pipeline with a simple LLM-powered agent that writes SQL/vector queries, shows actual messages with highlighting, and supports pagination + dialogue browsing.

**Architecture:** GPT-5 mini agent with two tools (vector_search + run_sql). Sync agent loop called via asyncio.to_thread from async Telegram handlers. In-memory state with LRU eviction for pagination/dialogue.

**Tech Stack:** python-telegram-bot 21+, openai SDK, chromadb, sqlalchemy, sqlite3

**Spec:** `docs/superpowers/specs/2026-03-13-agent-rework-design.md`

---

## File Structure

### Files to Delete
```
search/                          # Entire directory (13 files)
bot/formatters.py
bot/conversation_context.py
bot/upload_wizard.py
indexer.py
ingestion/                       # Entire directory
migrations/                      # Entire directory
scripts/                         # Entire directory
```

### Files to Modify
```
config.py                        # Strip to essentials + add agent config
db/models.py                     # Keep Message only, remove 5 other models
db/database.py                   # Strip to: init, get_message_by_db_id, get_messages_around, execute_safe_sql
db/__init__.py                   # Remove Export, other model imports
bot/__init__.py                  # Remove MessageFormatter import
bot/handlers.py                  # Complete rewrite — thin routing
main.py                          # Simplified — only "bot" command
requirements.txt                 # Remove ijson, aiohttp, aiofiles, psutil; add pytest
```

### Files to Create
```
agent/__init__.py                # Exports AgentLoop, StateManager, Formatter, DialogueWindow
agent/state.py                   # SearchState, DialogueState, StateManager (LRU + TTL)
agent/prompts.py                 # SYSTEM_PROMPT, TOOL_DEFINITIONS for OpenAI
agent/loop.py                    # AgentLoop: LLM ↔ tool execution orchestrator
agent/formatter.py               # Format messages for Telegram (highlight, truncate, keyboard)
agent/dialogue.py                # DialogueWindow: fetch surrounding messages, navigate
tests/test_state.py              # State manager tests
tests/test_formatter.py          # Formatter tests
tests/test_sql_safety.py         # SQL whitelist tests
```

---

## Chunk 1: Cleanup & Foundation

### Task 1: Delete old files and directories

**Files:**
- Delete: `search/`, `bot/formatters.py`, `bot/conversation_context.py`, `bot/upload_wizard.py`, `indexer.py`, `ingestion/`, `migrations/`, `scripts/`

- [ ] **Step 1: Remove old directories and files**

```bash
git rm -r search/
git rm -r ingestion/
git rm -r migrations/
git rm -r scripts/
git rm indexer.py
git rm bot/formatters.py
git rm bot/conversation_context.py
git rm bot/upload_wizard.py
```

- [ ] **Step 2: Commit cleanup**

```bash
git commit -m "chore: remove old search pipeline, ingestion, and helper files"
```

---

### Task 2: Rewrite config.py

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Replace config.py contents**

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Credentials
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Database
SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", DATA_DIR / "metadata.db"))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", DATA_DIR / "chroma"))

# Models
CHAT_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-large"  # Must match existing index

# Agent
AGENT_MAX_ITERATIONS = 3
AGENT_QUERY_TIMEOUT = 5        # seconds
AGENT_MAX_RESULTS = 50

# Display
RESULTS_PER_PAGE = 3
DIALOGUE_INITIAL_WINDOW = 5    # 2 before + selected + 2 after
DIALOGUE_SCROLL_SIZE = 3
MESSAGE_CHAR_LIMIT = 4096

# State
STATE_TTL_MINUTES = 30
STATE_MAX_CONCURRENT = 100
```

- [ ] **Step 2: Commit**

```bash
git add config.py
git commit -m "refactor: strip config.py to agent essentials"
```

---

### Task 3: Rewrite db/models.py and db/__init__.py

**Files:**
- Modify: `db/models.py`
- Modify: `db/__init__.py`

- [ ] **Step 1: Replace db/models.py — keep only Message model**

```python
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
```

- [ ] **Step 2: Replace db/__init__.py**

```python
from .database import Database
from .models import Base, Message

__all__ = ["Database", "Base", "Message"]
```

- [ ] **Step 3: Commit**

```bash
git add db/models.py db/__init__.py
git commit -m "refactor: strip db models to Message only"
```

---

### Task 4: Rewrite db/database.py

**Files:**
- Modify: `db/database.py`
- Create: `tests/test_sql_safety.py`

- [ ] **Step 1: Write SQL safety test**

```python
# tests/test_sql_safety.py
import pytest
import tempfile
from pathlib import Path
from db.database import Database


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        return Database(Path(f.name))


def test_select_allowed(db):
    # Should not raise (table may not exist, but validation passes)
    try:
        db.execute_safe_sql("SELECT 1")
    except Exception as e:
        # OperationalError from SQLite is OK (no table), ValueError is not
        assert not isinstance(e, ValueError)


def test_insert_rejected(db):
    with pytest.raises(ValueError, match="Only SELECT"):
        db.execute_safe_sql("INSERT INTO messages VALUES (1)")


def test_update_rejected(db):
    with pytest.raises(ValueError, match="Only SELECT"):
        db.execute_safe_sql("UPDATE messages SET text='x'")


def test_delete_rejected(db):
    with pytest.raises(ValueError, match="Only SELECT"):
        db.execute_safe_sql("DELETE FROM messages")


def test_drop_rejected(db):
    with pytest.raises(ValueError, match="Only SELECT"):
        db.execute_safe_sql("DROP TABLE messages")


def test_pragma_rejected(db):
    with pytest.raises(ValueError, match="Only SELECT"):
        db.execute_safe_sql("PRAGMA table_info(messages)")


def test_with_cte_allowed(db):
    result = db.execute_safe_sql("WITH cte AS (SELECT 1 AS v) SELECT * FROM cte")
    assert result[0]["v"] == 1


def test_limit_enforced(db):
    result = db.execute_safe_sql("SELECT 1 AS val")
    assert len(result) == 1
    assert result[0]["val"] == 1


def test_lowercase_select_allowed(db):
    result = db.execute_safe_sql("select 1 as val")
    assert result[0]["val"] == 1


def test_leading_whitespace_allowed(db):
    result = db.execute_safe_sql("   SELECT 1 AS val")
    assert result[0]["val"] == 1


def test_semicolon_injection_rejected(db):
    # Semicolons are stripped; multi-statement should fail or be harmless
    result = db.execute_safe_sql("SELECT 1 AS val;")
    assert result[0]["val"] == 1


def test_trailing_semicolon_stripped(db):
    result = db.execute_safe_sql("SELECT 1 AS val ;  ")
    assert result[0]["val"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/sion/telegram-chat-history-wizard && python -m pytest tests/test_sql_safety.py -v
```
Expected: FAIL (database.py doesn't have execute_safe_sql yet)

- [ ] **Step 3: Replace db/database.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/sion/telegram-chat-history-wizard && python -m pytest tests/test_sql_safety.py -v
```
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add db/database.py tests/test_sql_safety.py
git commit -m "refactor: strip database.py to essentials + SQL safety whitelist"
```

---

### Task 5: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Replace requirements.txt**

```
# Telegram Bot
python-telegram-bot>=21.0

# OpenAI
openai>=1.0

# Vector Database
chromadb>=0.4

# Database
sqlalchemy>=2.0

# Environment
python-dotenv>=1.0

# Testing
pytest>=8.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: remove unused deps, add pytest"
```

---

## Chunk 2: Agent Core

### Task 6: Create agent/state.py with tests

**Files:**
- Create: `agent/__init__.py`
- Create: `agent/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write state manager tests**

```python
# tests/test_state.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/sion/telegram-chat-history-wizard && python -m pytest tests/test_state.py -v
```
Expected: FAIL (module doesn't exist yet)

- [ ] **Step 3: Create agent/__init__.py (empty for now)**

```python
# agent/__init__.py
```

- [ ] **Step 4: Create agent/state.py**

```python
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
    last_accessed: float = field(default_factory=time.time)


@dataclass
class DialogueState:
    anchor_message_id: int
    anchor_chat_id: int
    current_window: list[dict]
    highlight_terms: list[str] = field(default_factory=list)
    saved_search_state: object = None  # Embedded SearchState copy for "back to results"
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
```

- [ ] **Step 5: Run tests**

```bash
cd /home/sion/telegram-chat-history-wizard && python -m pytest tests/test_state.py -v
```
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add agent/__init__.py agent/state.py tests/test_state.py
git commit -m "feat: add agent state manager with TTL and LRU eviction"
```

---

### Task 7: Create agent/prompts.py

**Files:**
- Create: `agent/prompts.py`

- [ ] **Step 1: Create agent/prompts.py**

```python
SYSTEM_PROMPT = """You are a chat history search agent. Your job is to find messages in a chat history database.

You have three tools:

1. vector_search(query, n_results) — semantic search in the vector database.
   Returns messages similar in meaning to the query.
   Best for: fuzzy topics, phrases with unknown exact wording, morphology variations.
   Results include a similarity score.

2. run_sql(sql) — execute a read-only SQL query against the messages table.
   Table schema:
     messages(id, message_id, chat_id, user_id, username, first_name, last_name,
              text, timestamp, timestamp_unix, reply_to_message_id,
              is_forwarded, forward_from, forward_date)
   Rules:
   - ALWAYS sort by timestamp, NEVER by id (two chats were merged, IDs are not chronological)
   - Use LIKE for text matching (case-insensitive by default in SQLite)
   - All queries search across all chats (no chat_id filter needed)
   Best for: exact phrases, date/time filters, counting, aggregations, user-specific queries.

3. submit_results(result_ids, highlight_terms, sort_order, explanation) — call this when you have found sufficient results.
   - result_ids: list of message 'id' values to display
   - highlight_terms: phrases to bold in the displayed messages
   - sort_order: "asc" (oldest first, DEFAULT) or "desc" (newest first)
   - explanation: brief description of what was found

Rules:
- Default sort order is oldest first (ASC) unless the user asks for recent/latest messages.
- You have a maximum of 3 tool calls total. Use them wisely.
- If the user says "exact/точно/дословно", use run_sql with LIKE only (no vector search).
- Otherwise, choose the best tool for the question.
- You MUST call submit_results when done. Never respond with plain text.
- When using vector_search results: the 'id' field in each result is the database ID to use in submit_results.
- When using run_sql results: the 'id' field in each result is the database ID to use in submit_results.
- Include all relevant highlight_terms — these are bolded in the displayed messages."""


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Semantic search in the chat history vector database. Returns messages similar in meaning. Best for fuzzy topics, unknown exact wording, morphology variations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query text"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (max 50)",
                        "default": 20
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a read-only SQL SELECT query against the messages table. Best for exact phrases (LIKE), date filters, counting, aggregations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL SELECT query to execute"
                    }
                },
                "required": ["sql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_results",
            "description": "Submit the final search results. Call this when you have found sufficient messages to answer the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of message database IDs to display"
                    },
                    "highlight_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Phrases to highlight (bold) in displayed messages"
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort order: asc=oldest first (default), desc=newest first"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation of what was found"
                    }
                },
                "required": ["result_ids", "highlight_terms", "sort_order"]
            }
        }
    }
]
```

- [ ] **Step 2: Commit**

```bash
git add agent/prompts.py
git commit -m "feat: add agent system prompt and OpenAI tool definitions"
```

---

### Task 8: Create agent/loop.py

**Files:**
- Create: `agent/loop.py`

- [ ] **Step 1: Create agent/loop.py**

```python
import json
import logging

import chromadb
from chromadb.config import Settings
from openai import OpenAI

import config
from db.database import Database
from .prompts import SYSTEM_PROMPT, TOOL_DEFINITIONS

logger = logging.getLogger(__name__)


class AgentLoop:
    """Orchestrates LLM ↔ tool execution for chat history search."""

    def __init__(self, db: Database):
        self.db = db
        self.openai = OpenAI(api_key=config.OPENAI_API_KEY)
        self.collection = None
        try:
            chroma = chromadb.PersistentClient(
                path=str(config.CHROMA_DB_PATH),
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = chroma.get_collection("messages")
        except Exception as e:
            logger.warning(f"ChromaDB unavailable, SQL-only mode: {e}")

    def process_query(self, user_message: str) -> dict:
        """
        Run the agent loop synchronously.

        Returns:
            {
                "results": [list of message dicts with id, text, first_name, timestamp, chat_id, ...],
                "highlight_terms": [...],
                "sort_order": "asc" | "desc",
                "explanation": "...",
                "error": None | "error message"
            }
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        collected_results: dict[int, dict] = {}  # id -> message dict

        for iteration in range(config.AGENT_MAX_ITERATIONS):
            try:
                response = self.openai.chat.completions.create(
                    model=config.CHAT_MODEL,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                )
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return self._error_response("Search service is temporarily unavailable. Try again later.")

            choice = response.choices[0]

            # No tool calls — LLM responded with text (fallback)
            if not choice.message.tool_calls:
                return self._fallback_response(collected_results, choice.message.content)

            # Process tool calls
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if name == "submit_results":
                    return self._handle_submit(args, collected_results)
                elif name == "vector_search":
                    result = self._exec_vector_search(args)
                elif name == "run_sql":
                    result = self._exec_sql(args)
                else:
                    result = {"error": f"Unknown tool: {name}"}

                # Accumulate results
                if isinstance(result, list):
                    for r in result:
                        if "id" in r and r["id"] not in collected_results:
                            collected_results[r["id"]] = r

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })

        # Max iterations exhausted
        return self._fallback_response(collected_results)

    # --- Tool Executors ---

    def _exec_vector_search(self, args: dict) -> list[dict] | dict:
        if not self.collection:
            return {"error": "Vector search unavailable. Use run_sql instead."}

        query = args.get("query", "")
        n_results = min(args.get("n_results", 20), config.AGENT_MAX_RESULTS)

        try:
            # Embed query
            emb_response = self.openai.embeddings.create(
                model=config.EMBEDDING_MODEL, input=query
            )
            query_embedding = emb_response.data[0].embedding

            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return [{"error": f"Vector search failed: {e}"}]

        if not results["ids"] or not results["ids"][0]:
            return []

        # Map vector IDs to DB ids, fetch full messages
        matches = []
        for i, vec_id in enumerate(results["ids"][0]):
            try:
                db_id = int(vec_id.split("_")[1])
            except (IndexError, ValueError):
                continue

            similarity = round(1 - results["distances"][0][i], 3)
            msg = self.db.get_message_by_db_id(db_id)  # Returns dict
            if not msg:
                continue

            msg["similarity"] = similarity
            matches.append(msg)

        return matches

    def _exec_sql(self, args: dict) -> list[dict] | dict:
        sql = args.get("sql", "")
        try:
            return self.db.execute_safe_sql(sql)
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return {"error": f"SQL error: {e}"}

    # --- Response Builders ---

    def _handle_submit(self, args: dict, collected: dict) -> dict:
        result_ids = args.get("result_ids", [])
        highlight_terms = args.get("highlight_terms", [])
        sort_order = args.get("sort_order", "asc")
        explanation = args.get("explanation", "")

        # Fetch messages that were in collected results
        results = [collected[rid] for rid in result_ids if rid in collected]

        # If some IDs weren't in collected (e.g., from SQL), fetch from DB
        missing_ids = [rid for rid in result_ids if rid not in collected]
        if missing_ids:
            db_msgs = self.db.get_messages_by_db_ids(missing_ids)
            results.extend(db_msgs)  # Already plain dicts from _msg_to_dict

        # Sort by timestamp
        reverse = sort_order == "desc"
        results.sort(key=lambda r: r.get("timestamp") or "", reverse=reverse)

        return {
            "results": results,
            "highlight_terms": highlight_terms,
            "sort_order": sort_order,
            "explanation": explanation,
            "error": None,
        }

    def _fallback_response(self, collected: dict, explanation: str = "") -> dict:
        results = list(collected.values())
        results.sort(key=lambda r: r.get("timestamp") or "")
        return {
            "results": results,
            "highlight_terms": [],
            "sort_order": "asc",
            "explanation": explanation or "Search completed",
            "error": None,
        }

    def _error_response(self, message: str) -> dict:
        return {
            "results": [],
            "highlight_terms": [],
            "sort_order": "asc",
            "explanation": "",
            "error": message,
        }
```

- [ ] **Step 2: Commit**

```bash
git add agent/loop.py
git commit -m "feat: add agent loop with vector_search + run_sql tools"
```

---

### Task 9: Update agent/__init__.py

**Files:**
- Modify: `agent/__init__.py`

- [ ] **Step 1: Update agent/__init__.py with exports**

```python
from .loop import AgentLoop
from .state import StateManager, SearchState, DialogueState

__all__ = ["AgentLoop", "StateManager", "SearchState", "DialogueState"]
```

- [ ] **Step 2: Commit**

```bash
git add agent/__init__.py
git commit -m "feat: add agent package exports"
```

---

## Chunk 3: Display, Handlers & Integration

### Task 10: Create agent/formatter.py with tests

**Files:**
- Create: `agent/formatter.py`
- Create: `tests/test_formatter.py`

- [ ] **Step 1: Write formatter tests**

```python
# tests/test_formatter.py
import pytest
from agent.formatter import Formatter


def test_highlight_single_term():
    f = Formatter()
    escaped = f.escape_html("Я разбираюсь в интеллекте лучше всех")
    result = f.highlight(escaped, ["интеллекте"])
    assert "<b>интеллекте</b>" in result


def test_highlight_case_insensitive():
    f = Formatter()
    escaped = f.escape_html("Привет МИР привет")
    result = f.highlight(escaped, ["мир"])
    assert "<b>МИР</b>" in result


def test_highlight_multiple_terms():
    f = Formatter()
    escaped = f.escape_html("альфа бета гамма")
    result = f.highlight(escaped, ["альфа", "гамма"])
    assert "<b>альфа</b>" in result
    assert "<b>гамма</b>" in result


def test_truncate_html_short():
    f = Formatter()
    result = f.truncate_html("Short text", 100)
    assert result == "Short text"


def test_truncate_html_long():
    f = Formatter()
    result = f.truncate_html("A" * 200, 100)
    assert result.endswith("...")
    # Count visible chars (excluding "...")
    visible = sum(1 for c in result if c != ".")
    assert visible <= 103


def test_truncate_html_preserves_tags():
    f = Formatter()
    text = "Hello <b>world</b> this is a test"
    result = f.truncate_html(text, 11)
    assert "<b>" in result


def test_escape_html():
    f = Formatter()
    result = f.escape_html("2 < 3 & 4 > 1")
    assert "&lt;" in result
    assert "&amp;" in result
    assert "&gt;" in result


def test_format_search_results_empty():
    f = Formatter()
    text, keyboard = f.format_search_results([], 0, 0, [], "asc")
    assert "Nothing found" in text


def test_format_search_results_page():
    f = Formatter()
    results = [
        {"id": 1, "first_name": "Леха", "text": "Тестовое сообщение", "timestamp": "2021-03-15T14:32:00", "chat_id": 100},
        {"id": 2, "first_name": "Саша", "text": "Еще сообщение", "timestamp": "2021-03-15T14:33:00", "chat_id": 100},
        {"id": 3, "first_name": "Дима", "text": "Третье сообщение", "timestamp": "2021-03-15T14:34:00", "chat_id": 100},
    ]
    text, keyboard = f.format_search_results(results, total=3, page=0, highlight_terms=[], sort_order="asc")
    assert "Леха" in text
    assert "Саша" in text
    assert "1-3" in text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/sion/telegram-chat-history-wizard && python -m pytest tests/test_formatter.py -v
```
Expected: FAIL

- [ ] **Step 3: Create agent/formatter.py**

```python
import re
from datetime import datetime

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

import config


class Formatter:
    """Format search results and dialogue windows for Telegram."""

    NUMBER_EMOJIS = ["1\u20e3", "2\u20e3", "3\u20e3"]

    def escape_html(self, text: str) -> str:
        """Escape HTML special characters for Telegram HTML mode."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def highlight(self, text: str, terms: list[str]) -> str:
        """Bold highlight terms in already-escaped HTML text (case-insensitive)."""
        for term in terms:
            escaped_term = self.escape_html(term)
            pattern = re.compile(re.escape(escaped_term), re.IGNORECASE)
            text = pattern.sub(lambda m: f"<b>{m.group()}</b>", text)
        return text

    def truncate_html(self, text: str, max_chars: int) -> str:
        """Truncate HTML-escaped text at max_chars of visible content, preserving tags."""
        visible = 0
        result = []
        i = 0
        while i < len(text):
            if text[i] == "<":
                # Skip HTML tags (don't count towards visible chars)
                end = text.find(">", i)
                if end == -1:
                    break
                result.append(text[i : end + 1])
                i = end + 1
            elif text[i] == "&":
                # HTML entity counts as 1 visible char
                end = text.find(";", i)
                if end == -1:
                    break
                result.append(text[i : end + 1])
                visible += 1
                i = end + 1
            else:
                result.append(text[i])
                visible += 1
                i += 1
            if visible >= max_chars:
                result.append("...")
                break
        return "".join(result)

    def _format_timestamp(self, ts_str: str) -> str:
        """Format ISO timestamp to DD.MM.YYYY HH:MM."""
        try:
            dt = datetime.fromisoformat(ts_str)
            return dt.strftime("%d.%m.%Y %H:%M")
        except (ValueError, TypeError):
            return "Unknown date"

    def _char_budget_per_message(self, msg_count: int) -> int:
        """Calculate per-message character budget to stay within Telegram limit."""
        overhead = 80  # header line
        per_msg_overhead = 50  # name + date line per message
        available = config.MESSAGE_CHAR_LIMIT - overhead - (per_msg_overhead * msg_count)
        return max(available // max(msg_count, 1), 100)

    def format_search_results(
        self,
        page_results: list[dict],
        total: int,
        page: int,
        highlight_terms: list[str],
        sort_order: str,
    ) -> tuple[str, InlineKeyboardMarkup | None]:
        """Format a page of search results with inline keyboard."""
        if not page_results:
            return "Nothing found. Try rephrasing your question.", None

        per_page = config.RESULTS_PER_PAGE
        start = page * per_page + 1
        end = start + len(page_results) - 1
        sort_label = "oldest first" if sort_order == "asc" else "newest first"

        lines = [f"Found {total} messages. Showing {start}-{end} ({sort_label}):\n"]
        budget = self._char_budget_per_message(len(page_results))

        for i, msg in enumerate(page_results):
            name = self.escape_html(msg.get("first_name") or msg.get("username") or "Unknown")
            date = self._format_timestamp(msg.get("timestamp", ""))
            text = msg.get("text") or ""
            text = self.escape_html(text)
            text = self.highlight(text, highlight_terms)
            text = self.truncate_html(text, budget)

            emoji = self.NUMBER_EMOJIS[i] if i < len(self.NUMBER_EMOJIS) else f"{i+1}."
            lines.append(f"{emoji} <b>{name}</b> \u00b7 {date}")
            lines.append(text)
            lines.append("")

        # Build keyboard
        buttons_nav = []
        total_pages = (total + per_page - 1) // per_page
        if page > 0:
            buttons_nav.append(InlineKeyboardButton("\u2b05\ufe0f Prev", callback_data=f"p:{page - 1}"))
        if page < total_pages - 1:
            buttons_nav.append(InlineKeyboardButton("Next \u27a1\ufe0f", callback_data=f"p:{page + 1}"))

        buttons_dialogue = []
        for i, msg in enumerate(page_results):
            buttons_dialogue.append(
                InlineKeyboardButton(f"\U0001f4ac {i+1}", callback_data=f"d:{msg['id']}")
            )

        keyboard_rows = []
        if buttons_nav:
            keyboard_rows.append(buttons_nav)
        if buttons_dialogue:
            keyboard_rows.append(buttons_dialogue)

        keyboard = InlineKeyboardMarkup(keyboard_rows) if keyboard_rows else None
        return "\n".join(lines).strip(), keyboard

    def format_dialogue_window(
        self,
        messages: list[dict],
        anchor_id: int,
        highlight_terms: list[str],
        has_earlier: bool,
        has_later: bool,
    ) -> tuple[str, InlineKeyboardMarkup]:
        """Format a dialogue window around an anchor message."""
        budget = self._char_budget_per_message(len(messages))

        lines = []
        for msg in messages:
            name = self.escape_html(msg.get("first_name") or msg.get("username") or "Unknown")
            date = self._format_timestamp(msg.get("timestamp", ""))
            text = msg.get("text") or ""
            text = self.truncate(text, budget)

            is_anchor = msg.get("id") == anchor_id
            if is_anchor:
                text = self.highlight(text, highlight_terms)
                lines.append(f"\U0001f449 <b>{name}</b> \u00b7 {date}")
            else:
                text = self.escape_html(text)
                lines.append(f"{name} \u00b7 {date}")
            lines.append(text)
            lines.append("")

        # Navigation buttons
        buttons_nav = []
        if has_earlier:
            first_ts = messages[0].get("timestamp_unix") or 0
            buttons_nav.append(InlineKeyboardButton("\u2b05\ufe0f Back", callback_data=f"db:{first_ts}"))
        if has_later:
            last_ts = messages[-1].get("timestamp_unix") or 0
            buttons_nav.append(InlineKeyboardButton("Forward \u27a1\ufe0f", callback_data=f"df:{last_ts}"))

        buttons_back = [InlineKeyboardButton("\U0001f519 Back to results", callback_data="br")]

        keyboard_rows = []
        if buttons_nav:
            keyboard_rows.append(buttons_nav)
        keyboard_rows.append(buttons_back)

        return "\n".join(lines).strip(), InlineKeyboardMarkup(keyboard_rows)
```

- [ ] **Step 4: Run tests**

```bash
cd /home/sion/telegram-chat-history-wizard && python -m pytest tests/test_formatter.py -v
```
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add agent/formatter.py tests/test_formatter.py
git commit -m "feat: add message formatter with highlighting and pagination keyboard"
```

---

### Task 11: Create agent/dialogue.py

**Files:**
- Create: `agent/dialogue.py`

- [ ] **Step 1: Create agent/dialogue.py**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add agent/dialogue.py
git commit -m "feat: add dialogue window navigation"
```

---

### Task 12: Rewrite bot/handlers.py and bot/__init__.py

**Files:**
- Modify: `bot/handlers.py`
- Modify: `bot/__init__.py`

- [ ] **Step 1: Replace bot/handlers.py**

```python
import asyncio
import logging
import re

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

import config
from db.database import Database
from agent.loop import AgentLoop
from agent.state import StateManager, SearchState, DialogueState
from agent.formatter import Formatter
from agent.dialogue import DialogueWindow

logger = logging.getLogger(__name__)


class BotHandlers:
    def __init__(self, db: Database, agent: AgentLoop):
        self.db = db
        self.agent = agent
        self.state = StateManager()
        self.formatter = Formatter()
        self.dialogue = DialogueWindow(db)

    # --- Commands ---

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Hi! I search through chat history.\n\n"
            "In a group — mention me with your question: @bot_name who said X?\n"
            "In DM — just type your question.\n\n"
            "Type /help for examples."
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "<b>How to search:</b>\n\n"
            "Ask a question — I'll find the messages.\n\n"
            "<b>Examples:</b>\n"
            "\u2022 Who first said 'разбирается в интеллекте'?\n"
            "\u2022 What did Леха say about crypto in 2023?\n"
            "\u2022 How many messages did Саша send?\n"
            "\u2022 What were we talking about last week?\n\n"
            "<b>Tips:</b>\n"
            "\u2022 Say 'точно' or 'дословно' for exact phrase match\n"
            "\u2022 Click \U0001f4ac buttons to see surrounding dialogue\n"
            "\u2022 Use \u2b05\ufe0f/\u27a1\ufe0f to navigate pages or scroll dialogue",
            parse_mode="HTML",
        )

    # --- Message Handling ---

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle search queries (group mentions and DMs)."""
        message = update.message
        if not message or not message.text:
            return

        # Extract query text
        query = message.text

        # Strip bot mention in groups
        if message.chat.type != "private" and context.bot.username:
            mention = f"@{context.bot.username}"
            if mention.lower() not in query.lower():
                return  # Not mentioned in group, ignore
            query = re.sub(re.escape(mention), "", query, flags=re.IGNORECASE).strip()

        if not query:
            return

        # Send "searching" indicator
        reply = await message.reply_text("\U0001f50d Searching...")

        # Run agent in thread (blocking I/O)
        try:
            result = await asyncio.to_thread(self.agent.process_query, query)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            await reply.edit_text("Search service is temporarily unavailable. Try again later.")
            return

        # Handle errors
        if result.get("error"):
            await reply.edit_text(result["error"])
            return

        # Handle empty results
        if not result["results"]:
            await reply.edit_text("Nothing found. Try rephrasing your question.")
            return

        # Store state
        search_state = SearchState(
            all_results=result["results"],
            original_query=query,
            sort_order=result.get("sort_order", "asc"),
            highlight_terms=result.get("highlight_terms", []),
        )
        self.state.set(message.chat_id, reply.message_id, search_state)

        # Format first page
        page_results = result["results"][:config.RESULTS_PER_PAGE]
        text, keyboard = self.formatter.format_search_results(
            page_results=page_results,
            total=len(result["results"]),
            page=0,
            highlight_terms=search_state.highlight_terms,
            sort_order=search_state.sort_order,
        )

        await reply.edit_text(text, parse_mode="HTML", reply_markup=keyboard)

    # --- Callback Handlers ---

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Route inline button callbacks."""
        query = update.callback_query
        await query.answer()
        data = query.data

        if data.startswith("p:"):
            await self._handle_page(query, data)
        elif data.startswith("d:"):
            await self._handle_dialogue_open(query, data)
        elif data.startswith("db:"):
            await self._handle_dialogue_back(query, data)
        elif data.startswith("df:"):
            await self._handle_dialogue_forward(query, data)
        elif data == "br":
            await self._handle_back_to_results(query)

    async def _handle_page(self, query, data: str):
        """Navigate search result pages."""
        page = int(data.split(":")[1])
        state = self.state.get(query.message.chat_id, query.message.message_id)

        if not isinstance(state, SearchState):
            await query.message.edit_text("This search has expired. Please search again.")
            return

        state.current_page = page
        per_page = config.RESULTS_PER_PAGE
        start = page * per_page
        page_results = state.all_results[start : start + per_page]

        text, keyboard = self.formatter.format_search_results(
            page_results=page_results,
            total=len(state.all_results),
            page=page,
            highlight_terms=state.highlight_terms,
            sort_order=state.sort_order,
        )
        await query.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)

    async def _handle_dialogue_open(self, query, data: str):
        """Open dialogue window around a message."""
        msg_id = int(data.split(":")[1])
        chat_id = query.message.chat_id
        bot_msg_id = query.message.message_id

        # Save search state before overwriting with dialogue state
        search_state = self.state.get(chat_id, bot_msg_id)
        saved_search = search_state if isinstance(search_state, SearchState) else None
        highlight_terms = search_state.highlight_terms if search_state else []

        # Open dialogue
        window, anchor_id, anchor_chat_id = await asyncio.to_thread(
            self.dialogue.open, msg_id
        )

        if not window:
            await query.message.edit_text("Message not found.")
            return

        # Check navigation availability
        has_earlier = await asyncio.to_thread(
            self.dialogue.has_earlier, anchor_chat_id, window[0].get("timestamp_unix", 0)
        )
        has_later = await asyncio.to_thread(
            self.dialogue.has_later, anchor_chat_id, window[-1].get("timestamp_unix", 0)
        )

        # Store dialogue state with embedded search state for "back to results"
        dialogue_state = DialogueState(
            anchor_message_id=anchor_id,
            anchor_chat_id=anchor_chat_id,
            current_window=window,
            highlight_terms=highlight_terms,
            saved_search_state=saved_search,
        )
        self.state.set(chat_id, bot_msg_id, dialogue_state)

        text, keyboard = self.formatter.format_dialogue_window(
            messages=window,
            anchor_id=anchor_id,
            highlight_terms=highlight_terms,
            has_earlier=has_earlier,
            has_later=has_later,
        )
        await query.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)

    async def _handle_dialogue_back(self, query, data: str):
        """Scroll dialogue backward."""
        first_ts = int(data.split(":")[1])
        state = self.state.get(query.message.chat_id, query.message.message_id)

        if not isinstance(state, DialogueState):
            await query.message.edit_text("This search has expired. Please search again.")
            return

        # Fetch earlier messages
        earlier = await asyncio.to_thread(
            self.dialogue.scroll_back, state.anchor_chat_id, first_ts
        )

        if not earlier:
            return  # No earlier messages

        # New window: earlier + first message of current window (overlap)
        window = earlier + [state.current_window[0]]
        state.current_window = window

        has_earlier = await asyncio.to_thread(
            self.dialogue.has_earlier, state.anchor_chat_id, window[0].get("timestamp_unix", 0)
        )
        has_later = True  # We came from a later window, so there are later messages

        text, keyboard = self.formatter.format_dialogue_window(
            messages=window,
            anchor_id=state.anchor_message_id,
            highlight_terms=state.highlight_terms,
            has_earlier=has_earlier,
            has_later=has_later,
        )
        await query.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)

    async def _handle_dialogue_forward(self, query, data: str):
        """Scroll dialogue forward."""
        last_ts = int(data.split(":")[1])
        state = self.state.get(query.message.chat_id, query.message.message_id)

        if not isinstance(state, DialogueState):
            await query.message.edit_text("This search has expired. Please search again.")
            return

        # Fetch later messages
        later = await asyncio.to_thread(
            self.dialogue.scroll_forward, state.anchor_chat_id, last_ts
        )

        if not later:
            return  # No later messages

        # New window: last message of current window (overlap) + later
        window = [state.current_window[-1]] + later
        state.current_window = window

        has_earlier = True  # We came from an earlier window
        has_later = await asyncio.to_thread(
            self.dialogue.has_later, state.anchor_chat_id, window[-1].get("timestamp_unix", 0)
        )

        text, keyboard = self.formatter.format_dialogue_window(
            messages=window,
            anchor_id=state.anchor_message_id,
            highlight_terms=state.highlight_terms,
            has_earlier=has_earlier,
            has_later=has_later,
        )
        await query.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)

    async def _handle_back_to_results(self, query):
        """Return to search results from dialogue view."""
        chat_id = query.message.chat_id
        bot_msg_id = query.message.message_id

        state = self.state.get(chat_id, bot_msg_id)

        # Recover search state embedded in dialogue state
        search_state = None
        if isinstance(state, DialogueState) and state.saved_search_state:
            search_state = state.saved_search_state

        if not isinstance(search_state, SearchState):
            await query.message.edit_text("This search has expired. Please search again.")
            return

        # Restore search state to this message
        self.state.set(chat_id, bot_msg_id, search_state)

        page = search_state.current_page
        per_page = config.RESULTS_PER_PAGE
        start = page * per_page
        page_results = search_state.all_results[start : start + per_page]

        text, keyboard = self.formatter.format_search_results(
            page_results=page_results,
            total=len(search_state.all_results),
            page=page,
            highlight_terms=search_state.highlight_terms,
            sort_order=search_state.sort_order,
        )
        await query.message.edit_text(text, parse_mode="HTML", reply_markup=keyboard)


def setup_handlers(app: Application, db: Database, agent: AgentLoop):
    """Register all handlers with the Telegram application."""
    handlers = BotHandlers(db, agent)

    app.add_handler(CommandHandler("start", handlers.cmd_start))
    app.add_handler(CommandHandler("help", handlers.cmd_help))
    app.add_handler(CallbackQueryHandler(handlers.handle_callback))

    # DM: all text messages are queries
    app.add_handler(MessageHandler(
        filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND,
        handlers.handle_message,
    ))

    # Group: messages mentioning the bot
    app.add_handler(MessageHandler(
        filters.TEXT & filters.ChatType.GROUPS & ~filters.COMMAND,
        handlers.handle_message,
    ))
```

- [ ] **Step 2: Replace bot/__init__.py**

```python
from .handlers import setup_handlers

__all__ = ["setup_handlers"]
```

- [ ] **Step 3: Commit**

```bash
git add bot/handlers.py bot/__init__.py
git commit -m "feat: rewrite bot handlers — thin routing with agent loop"
```

---

### Task 13: Rewrite main.py and verify

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Replace main.py**

```python
#!/usr/bin/env python3
"""
Telegram Chat History Wizard

An agent-powered bot for searching through chat history.

Usage:
    python main.py bot    - Run the Telegram bot
    python main.py help   - Show this help
"""
import sys
import logging

from telegram.ext import Application

import config
from db import Database
from agent import AgentLoop
from bot import setup_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_bot():
    """Run the Telegram bot."""
    if not config.TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env file")
        sys.exit(1)
    if not config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set in .env file")
        sys.exit(1)

    db = Database(config.SQLITE_DB_PATH)
    agent = AgentLoop(db)

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    setup_handlers(app, db, agent)

    print("=" * 50)
    print("Chat History Wizard — Agent Mode")
    print("=" * 50)
    print(f"Model: {config.CHAT_MODEL}")
    print(f"Max iterations: {config.AGENT_MAX_ITERATIONS}")
    print()
    print("Bot is running... Press Ctrl+C to stop.")

    app.run_polling(allowed_updates=["message", "callback_query"])


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "-h", "--help"):
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()
    if command == "bot":
        run_bot()
    else:
        print(f"Unknown command: {command}")
        print("Use 'python main.py bot' to start the bot.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests**

```bash
cd /home/sion/telegram-chat-history-wizard && python -m pytest tests/ -v
```
Expected: ALL PASS

- [ ] **Step 3: Verify imports work**

```bash
cd /home/sion/telegram-chat-history-wizard && python -c "from agent import AgentLoop, StateManager; from bot import setup_handlers; print('All imports OK')"
```
Expected: "All imports OK"

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: simplify main.py — agent mode only"
```

- [ ] **Step 5: Final commit — update agent/__init__.py with all exports**

```python
from .loop import AgentLoop
from .state import StateManager, SearchState, DialogueState
from .formatter import Formatter
from .dialogue import DialogueWindow

__all__ = [
    "AgentLoop",
    "StateManager",
    "SearchState",
    "DialogueState",
    "Formatter",
    "DialogueWindow",
]
```

```bash
git add agent/__init__.py
git commit -m "chore: update agent exports"
```

---

## Post-Implementation Checklist

After all tasks are complete:

- [ ] Run full test suite: `python -m pytest tests/ -v`
- [ ] Verify bot starts: `python main.py bot` (check for import errors, config issues)
- [ ] Test in Telegram DM: send a search query, verify results appear with buttons
- [ ] Test pagination: click Next/Prev buttons
- [ ] Test dialogue: click a 💬 button, verify surrounding messages appear
- [ ] Test dialogue navigation: Back/Forward buttons
- [ ] Test "Back to results" button
- [ ] Test in group: mention bot with a query
