# Agent-Based Bot Rework — Design Spec

## Overview

Complete rework of the Telegram Chat History Wizard bot from a complex vector search pipeline (HyDE + reranking + answer synthesis) to a simple **agent loop** that understands user questions, generates queries, and shows actual messages with pagination and dialogue browsing.

**Core principle:** Find the right messages and show them. No AI summaries. No over-engineering.

## Problem Statement

The current bot consistently fails to find relevant messages. Key failure mode: when searching for a phrase like "разбирается в интеллекте," the system returns jokes that repeat the phrase but never finds the original message. Users find the results unusable.

Root causes:
- Vector search ranks by semantic similarity, not chronological order
- Repeated/popular phrases score higher than originals
- Complex pipeline (HyDE, reranking, diversity filtering) adds noise rather than precision
- AI-synthesized answers obscure whether the actual messages were found

## Architecture

### Agent Loop

The bot uses a GPT-5 mini powered agent that receives the user's question and has access to two tools:

1. **`vector_search(query, n_results)`** — semantic search in ChromaDB. Best for fuzzy topics, phrases with unknown exact wording, morphology variations (Russian word endings).
2. **`run_sql(sql)`** — execute a SELECT query against the SQLite `messages` table. Best for exact phrases, date/time filters, counting, user-specific queries.

The agent decides which tool to use based on the question. It can chain tools (e.g., vector search first, then SQL to refine). Max **3 iterations** — if results aren't satisfactory after 3 tries, return whatever was found.

### Tool Selection Logic

| Question Type | Tool | Reasoning |
|---------------|------|-----------|
| "Кто сказал X" (fuzzy phrase) | vector_search | Morphology variations handled by embeddings |
| "Точная цитата: X" (exact) | run_sql (LIKE) | User explicitly wants literal match |
| "Сколько сообщений написал X в 2023" | run_sql | Aggregation + date filter |
| "Что говорили про криптовалюту" | vector_search | Semantic topic match |
| "Кто первый написал слово биткоин" | run_sql | Exact keyword + chronological sort |

### Sort Order

- **Default:** oldest first (`ORDER BY timestamp ASC`) — finds first occurrences
- LLM infers from the question if a different order is needed:
  - "Кто первый сказал X" → ASC
  - "Что последнее говорили про X" → DESC
  - "Кто сказал X" (no time hint) → ASC (default)

### Safety Constraints

- **Whitelist approach:** Only `SELECT` statements allowed. Parse the query and reject if the first keyword is not `SELECT`. Do not use a denylist.
- Query timeout: 5 seconds max via SQLAlchemy `execution_options`
- Result limit: wrap user query as `SELECT * FROM ({user_query}) LIMIT 50` to enforce cap
- Max 3 agent iterations per user query

## System Prompt

The LLM receives:

```
You are a chat history search agent. You have two tools:

1. vector_search(query, n_results) — semantic search in ChromaDB,
   returns messages similar in meaning. Best for fuzzy topics,
   phrases with unknown wording, morphology variations.

2. run_sql(sql) — execute SELECT query against SQLite messages table:
   Columns: id, message_id, chat_id, user_id, username, first_name,
            last_name, text, timestamp, timestamp_unix,
            reply_to_message_id, is_forwarded, forward_from, forward_date
   - ALWAYS sort by timestamp, NEVER by id (two chats were merged)
   - Use LIKE for text matching (case-insensitive by default in SQLite)
   - Best for: exact phrases, date filters, counting, user filters
   - All queries search across all chats (no chat_id filter needed)

Rules:
- Default sort: oldest first (ASC) unless user asks for recent/latest
- Max 3 iterations to find results
- Return: matching messages + highlight_terms + sort_order
- If user says "exact/точно/дословно" → use SQL LIKE only
- Otherwise prefer vector_search first for phrase queries
```

### Structured Output

The LLM returns structured output containing:
```json
{
  "tool_calls": [...],
  "highlight_terms": ["разбирается в интеллекте"],
  "sort_order": "asc",
  "explanation": "Looking for first occurrence of this phrase"
}
```

After tool execution, results are fed back. The LLM either returns final results or issues another tool call for refinement.

### Tool I/O Formats

**`vector_search` input:**
```json
{"query": "разбирается в интеллекте", "n_results": 20}
```

**`vector_search` output** (list of dicts):
```json
[
  {
    "id": 12345,
    "user_id": 645706876,
    "username": "leha",
    "first_name": "Леха",
    "text": "Ну я вообще-то разберусь в интелекте...",
    "timestamp": "2021-03-15T14:32:00",
    "similarity": 0.87
  }
]
```

**`run_sql` input:**
```json
{"sql": "SELECT * FROM messages WHERE text LIKE '%интеллект%' ORDER BY timestamp ASC"}
```

**`run_sql` output** (list of dicts, same shape as vector_search minus `similarity`):
```json
[
  {
    "id": 12345,
    "user_id": 645706876,
    "username": "leha",
    "first_name": "Леха",
    "text": "Ну я вообще-то разберусь в интелекте...",
    "timestamp": "2021-03-15T14:32:00"
  }
]
```

**Final LLM response** (when satisfied with results):
```json
{
  "done": true,
  "result_ids": [12345, 12400, 12455],
  "highlight_terms": ["разбирается в интеллекте", "разберусь в интелекте"],
  "sort_order": "asc",
  "explanation": "Found original phrase and subsequent references"
}
```

### Callback Data Format

Telegram inline button callback data (max 64 bytes):

| Button | Callback Data | Example |
|--------|---------------|---------|
| Prev page | `p:{page}` | `p:0` |
| Next page | `p:{page}` | `p:2` |
| Dialogue N | `d:{msg_id}` | `d:12345` |
| Dialogue Back | `db:{timestamp}` | `db:1615815120` |
| Dialogue Forward | `df:{timestamp}` | `df:1615815360` |
| Back to results | `br` | `br` |

## Result Display

### Search Results (3 per page)

Format per message:
```
{number_emoji} {first_name} · {DD.MM.YYYY HH:MM}
{message_text_with_bold_highlights}
```

- 3 messages shown per page
- Hard character cut per message — budget calculated dynamically to fit within Telegram's 4096 char limit (accounting for metadata lines and all 3 messages)
- Highlight terms wrapped in bold (`<b>...</b>` in Telegram HTML mode)
- Truncated messages end with `...`

Header line:
```
Found {N} messages. Showing {X}-{Y} ({sort_description}):
```

### Inline Keyboard — Search Results

```
Row 1: [⬅️ Prev] [Next ➡️]
Row 2: [💬 1] [💬 2] [💬 3]
```

- Prev/Next navigate pages of 3
- Prev hidden on page 1, Next hidden on last page
- 💬 buttons open dialogue window for that message

### Dialogue Window

When user clicks a dialogue button, shows a window of messages around the selected one:

**Initial view:** 2 messages before + selected message + 2 messages after = **5 messages**

- Selected message marked with 👉 prefix and bold name
- Surrounding messages shown in dimmer style (plain text)
- Highlight terms still bolded in the selected message

**Navigation:**
- **Back button:** 3 earlier messages + first message from current view = **4 messages** (1 overlap for continuity)
- **Forward button:** last message from current view + 3 newer messages = **4 messages** (1 overlap for continuity)

### Inline Keyboard — Dialogue Window

```
Row 1: [⬅️ Back] [Forward ➡️]
Row 2: [🔙 Back to results]
```

- Back hidden when at the beginning of chat history
- Forward hidden when at the end
- "Back to results" returns to the search results at the last viewed page

### Character Budget

Telegram message limit: 4096 characters.

Budget allocation per view:
- Header line: ~60 chars
- Per message metadata (name + date): ~40 chars
- Buttons markup: not counted (separate from message text)
- Remaining budget split equally among messages
- Search results: 3 messages → ~1300 chars each max
- Dialogue (5 messages): ~780 chars each max
- Dialogue navigation (4 messages): ~990 chars each max

## State Management

In-memory dictionary keyed by `(chat_id, bot_message_id)`.

### Search State
```python
{
    "type": "search",
    "all_results": [...],        # Up to 50 messages
    "current_page": 0,           # Which page of 3
    "sort_order": "asc",
    "highlight_terms": [...],
    "original_query": "...",
}
```

### Dialogue State
```python
{
    "type": "dialogue",
    "anchor_message_id": 123,    # The message user clicked on
    "current_window": [...],     # Currently displayed messages
    "window_start_ts": ...,      # Timestamp of first message in window
    "window_end_ts": ...,        # Timestamp of last message in window
    "search_state_key": (...),   # Reference back to search state
    "highlight_terms": [...],
}
```

**TTL:** 30 minutes. **Max concurrent:** 100 conversations. When the limit is reached, the oldest entry (by last access time) is evicted (LRU).

## Interaction Modes

- **Group chat:** Bot triggered by `@bot_name` mention. Mention is stripped before processing.
- **Direct message:** No mention needed. All text treated as a query.

### Commands

- `/start` — Welcome message explaining what the bot does and how to use it
- `/help` — Usage instructions with example queries

## Project Structure

### Files to Remove
```
search/                          # Entire directory
  embeddings.py, vector_store.py, search_agent.py, flip_detector.py,
  question_parser.py, answer_synthesizer.py, intent_detection.py,
  entity_aliases.py, language_utils.py, diversity.py, analytics.py, tools.py

bot/formatters.py                # Replaced by agent/formatter.py
bot/conversation_context.py      # Replaced by agent/state.py
bot/upload_wizard.py             # Not needed
indexer.py                       # Not needed
ingestion/                       # Entire directory
migrations/                      # Not needed
scripts/                         # Not needed
```

### Files to Keep (modified)
```
main.py                          # Simplified — only "bot" command
config.py                        # Stripped down — remove search-specific configs
db/models.py                     # Keep messages table, remove other models
db/database.py                   # Strip to basic message queries only
bot/__init__.py                  # Keep
bot/handlers.py                  # Complete rewrite — thin routing layer
```

### New Files
```
agent/
  __init__.py
  loop.py                       # Agent loop orchestrator (LLM ↔ tool execution)
  prompts.py                    # System prompt + schema definition
  formatter.py                  # Message formatting + highlighting
  dialogue.py                   # Dialogue window fetch + navigation
  state.py                      # Search/dialogue state management
```

### DB Tables to Remove
```
user_aliases
entity_aliases
relevance_cache
search_feedback
exports
```

### DB Tables to Keep
```
messages                         # Core data — id, message_id, chat_id, user_id,
                                 # username, first_name, last_name, text, timestamp,
                                 # timestamp_unix, reply_to_message_id, vector_id,
                                 # is_embedded, is_forwarded, forward_from, forward_date
```

### DB Migration

The `messages.export_id` column has a foreign key to the `exports` table. Since `exports` is being removed:
- Drop the `export_id` FK constraint and column from the `messages` model
- The `DISPLAY_NAME_OVERRIDES` config is removed — strip all references from `db/models.py` and `db/database.py` (use `first_name` directly for display)

### ChromaDB
Kept as-is. Used only by the `vector_search` tool. No changes to the existing embeddings or collection.

## Dependencies

### Keep
```
python-telegram-bot>=21.0        # Telegram bot framework
openai>=1.0                      # GPT-5 mini API
chromadb>=0.4                    # Vector search tool
sqlalchemy>=2.0                  # Database ORM
python-dotenv>=1.0               # Environment config
```

### Remove
```
ijson                            # Was for streaming JSON import
aiohttp                         # Was for async HTTP
aiofiles                        # Was for async file ops
psutil                          # Was for memory monitoring
```

## Configuration Changes

### config.py — Keep
```python
TELEGRAM_BOT_TOKEN               # From .env
OPENAI_API_KEY                   # From .env
SQLITE_DB_PATH                   # data/metadata.db
CHROMA_DB_PATH                   # data/chroma
```

### config.py — Add
```python
CHAT_MODEL = "gpt-5-mini"        # Agent LLM
AGENT_MAX_ITERATIONS = 3         # Max tool call rounds
AGENT_QUERY_TIMEOUT = 5          # SQL timeout in seconds
AGENT_MAX_RESULTS = 50           # Max results per search
RESULTS_PER_PAGE = 3             # Messages per page
DIALOGUE_INITIAL_WINDOW = 5      # 2 before + selected + 2 after
DIALOGUE_SCROLL_SIZE = 3         # Messages to load on back/forward
STATE_TTL_MINUTES = 30           # State expiry
STATE_MAX_CONCURRENT = 100       # Max active conversations
MESSAGE_CHAR_LIMIT = 4096        # Telegram message limit
```

### config.py — Remove
```python
EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, CHROMA_BATCH_SIZE,
MAX_MESSAGE_LENGTH, CHUNK_OVERLAP, EMBEDDING_WORKERS,
DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT, RELEVANCE_CACHE_TTL_HOURS,
DIVERSITY_LAMBDA, MAX_RESULTS_PER_USER, PENALIZE_FORWARDS_FACTOR,
EXCLUDE_FORWARDS_FOR_SPEAKER_QUERIES, DISPLAY_NAME_OVERRIDES,
ANALYTICS_TOP_LIMIT, CHAT_EXPORTS_DIR
```

## Error Handling

| Scenario | User-facing message |
|----------|---------------------|
| OpenAI API down/timeout | "Search service is temporarily unavailable. Try again later." |
| SQL query timeout (>5s) | "Query took too long. Try a more specific search." |
| 0 results after 3 iterations | "Nothing found. Try rephrasing your question." |
| ChromaDB unavailable | Agent falls back to SQL-only search |
| Non-search message in DM | Treated as a search query (everything is a query in DM) |
| Callback for expired state | "This search has expired. Please search again." |

## Dialogue Window Fetch Logic

All dialogue queries filter by `chat_id` of the selected message, so scrolling stays within the same chat and doesn't pull in messages from the other merged chat.

**Initial view** (clicking 💬 N):
```
anchor_ts = selected_message.timestamp
cid = selected_message.chat_id
before = SELECT * FROM messages WHERE chat_id = cid AND timestamp < anchor_ts ORDER BY timestamp DESC LIMIT 2
after  = SELECT * FROM messages WHERE chat_id = cid AND timestamp > anchor_ts ORDER BY timestamp ASC LIMIT 2
display = reversed(before) + [selected] + after   # 5 messages, chronological
```

**Back button:**
```
first_ts = current_window[0].timestamp
cid = current_window[0].chat_id
earlier = SELECT * FROM messages WHERE chat_id = cid AND timestamp < first_ts ORDER BY timestamp DESC LIMIT 3
display = reversed(earlier) + [current_window[0]]  # 4 messages, chronological
```

**Forward button:**
```
last_ts = current_window[-1].timestamp
cid = current_window[-1].chat_id
later = SELECT * FROM messages WHERE chat_id = cid AND timestamp > last_ts ORDER BY timestamp ASC LIMIT 3
display = [current_window[-1]] + later              # 4 messages, chronological
```

## Data Flow Summary

```
1. USER → message (group @mention or DM text)
2. HANDLER → strips mention, passes to AgentLoop
3. AGENT LOOP → sends to GPT-5 mini with system prompt + schema
   ├── LLM returns tool call (vector_search or run_sql)
   ├── Agent executes tool
   ├── LLM reviews results
   └── Repeats if needed (max 3 iterations)
4. FORMATTER → sorts, paginates (3 per page), truncates, highlights
5. BOT → sends formatted message + inline keyboard
6. USER → clicks Prev/Next → paginate search results
7. USER → clicks 💬 N → open dialogue window
8. USER → clicks Back/Forward → scroll dialogue
9. USER → clicks 🔙 Results → return to search page
```
