"""
Tool-based analytics system.
Allows the AI to call database/search tools to answer complex questions.
"""
import json
import logging
from typing import Any
from dataclasses import dataclass

import config
from db import Database
from search.vector_store import VectorStore
from search.embeddings import ChatService
from search.entity_aliases import get_all_forms, get_canonical


def _get_display_name(user_id: int, username: str) -> str:
    """Get display name for user, checking overrides first."""
    overrides = getattr(config, 'DISPLAY_NAME_OVERRIDES', {})
    if user_id and user_id in overrides:
        return overrides[user_id]
    return username or f"User#{user_id}"

logger = logging.getLogger(__name__)


# Tool definitions for the AI
TOOLS_SCHEMA = [
    {
        "name": "count_term_mentions",
        "description": "Count how many times each user mentioned a specific term/word. Aliases are auto-expanded (e.g., 'зелупа' searches for all forms: зе, зеля, Зеленський, etc.). Use for questions like 'who mentions X most often'.",
        "parameters": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "The term/word to search for. Use any form - aliases expand automatically (e.g., 'зелупа', 'порох', 'біток')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max users to return (default 10)"
                }
            },
            "required": ["term"]
        }
    },
    {
        "name": "get_top_speakers",
        "description": "Get users ranked by total message count. Use for questions like 'who talks/writes most'.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max users to return (default 10)"
                }
            }
        }
    },
    {
        "name": "search_messages",
        "description": "Semantic search for messages matching a query. Use to find what people said about a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (topic, phrase, or question)"
                },
                "user_filter": {
                    "type": "string",
                    "description": "Optional: filter by username, @username, user_id, or User#id format"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "compare_term_mentions",
        "description": "Compare how often different terms are mentioned. Aliases are auto-expanded for each term. Use for comparison questions like 'who mentions зелупу vs пороха'.",
        "parameters": {
            "type": "object",
            "properties": {
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of terms to compare. Use any form - aliases expand automatically (e.g., ['зелупа', 'порох'] or ['біток', 'ефір'])"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max users to return per term (default 10)"
                }
            },
            "required": ["terms"]
        }
    },
    {
        "name": "get_user_stats",
        "description": "Get detailed statistics for a specific user by username or user_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "Username, user_id, or User#id format to look up"
                }
            },
            "required": ["username"]
        }
    },
    {
        "name": "get_user_messages",
        "description": "Get recent messages from a specific user. Use when asked to show someone's messages without a specific topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_identifier": {
                    "type": "string",
                    "description": "Username, user_id, or User#id format"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max messages to return (default 10)"
                }
            },
            "required": ["user_identifier"]
        }
    }
]


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    data: Any
    error: str = None


class ToolExecutor:
    """Executes tools against the database and vector store."""

    def __init__(self, db: Database, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store

    def execute(self, tool_name: str, params: dict) -> ToolResult:
        """Execute a tool by name with given parameters."""
        try:
            if tool_name == "count_term_mentions":
                return self._count_term_mentions(params)
            elif tool_name == "get_top_speakers":
                return self._get_top_speakers(params)
            elif tool_name == "search_messages":
                return self._search_messages(params)
            elif tool_name == "compare_term_mentions":
                return self._compare_term_mentions(params)
            elif tool_name == "get_user_stats":
                return self._get_user_stats(params)
            elif tool_name == "get_user_messages":
                return self._get_user_messages(params)
            else:
                return ToolResult(tool_name, False, None, f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return ToolResult(tool_name, False, None, str(e))

    def _count_term_mentions(self, params: dict) -> ToolResult:
        """Count term mentions by user, expanding entity aliases."""
        term = params.get("term", "")
        limit = params.get("limit", 10)

        # Expand term to all known alias forms
        all_forms = get_all_forms(term)
        canonical = get_canonical(term)

        logger.info(f"Searching for term '{term}' expanded to forms: {all_forms}")

        # Search for all forms at once
        results = self.db.get_term_mention_counts_multi(all_forms, limit=limit)

        data = {
            "term": canonical,  # Use canonical form for display
            "searched_forms": all_forms,
            "results": [
                {"username": username, "count": count, "rank": i + 1}
                for i, (user_id, username, count) in enumerate(results)
            ],
            "total_users": len(results)
        }

        return ToolResult("count_term_mentions", True, data)

    def _get_top_speakers(self, params: dict) -> ToolResult:
        """Get top speakers by message count."""
        limit = params.get("limit", 10)

        results = self.db.get_message_count_by_user(limit=limit)

        data = {
            "results": [
                {"username": username, "message_count": count, "rank": i + 1}
                for i, (user_id, username, count) in enumerate(results)
            ]
        }

        return ToolResult("get_top_speakers", True, data)

    def _search_messages(self, params: dict) -> ToolResult:
        """Semantic search for messages."""
        query = params.get("query", "")
        limit = params.get("limit", 10)
        user_filter = params.get("user_filter")

        # Handle user_filter - could be username, @username, user_id, or User#id
        if user_filter:
            # Normalize user_filter
            user_identifier = user_filter.lstrip("@")
            if user_identifier.startswith("User#"):
                user_identifier = user_identifier[5:]  # Extract just the number

            results = self.vector_store.search_by_user(
                query=query,
                user_identifier=user_identifier,
                n_results=limit
            )
        else:
            results = self.vector_store.search(
                query=query,
                n_results=limit
            )

        # Apply display name overrides
        formatted_results = []
        for r in results:
            meta = r.get("metadata", {})
            user_id = meta.get("user_id")
            username = meta.get("display_name", "Unknown")

            # Apply override if available
            if user_id:
                display = _get_display_name(user_id, username)
            else:
                display = username

            formatted_results.append({
                "text": r.get("text", "")[:300],
                "username": display,
                "date": meta.get("formatted_date", ""),
                "similarity": round(r.get("similarity", 0), 3)
            })

        data = {
            "query": query,
            "user_filter": user_filter,
            "results": formatted_results,
            "total_found": len(results),
            "message": "Нічого не знайдено" if not results else None
        }

        return ToolResult("search_messages", True, data)

    def _compare_term_mentions(self, params: dict) -> ToolResult:
        """Compare mentions of multiple terms, expanding entity aliases."""
        terms = params.get("terms", [])
        limit = params.get("limit", 10)

        comparison = {}
        all_users = set()
        term_forms_map = {}  # Track which forms were searched for each term

        for term in terms:
            # Expand each term to all known alias forms
            all_forms = get_all_forms(term)
            canonical = get_canonical(term)
            term_forms_map[canonical] = all_forms

            logger.info(f"Comparing term '{term}' expanded to forms: {all_forms}")

            # Search for all forms at once
            results = self.db.get_term_mention_counts_multi(all_forms, limit=limit * 2)
            comparison[canonical] = {
                username: count
                for user_id, username, count in results
            }
            all_users.update(comparison[canonical].keys())

        # Use canonical forms for the comparison
        canonical_terms = list(comparison.keys())

        # Build comparison table
        user_comparison = []
        for username in all_users:
            row = {"username": username}
            for term in canonical_terms:
                row[term] = comparison[term].get(username, 0)
            row["total"] = sum(row.get(term, 0) for term in canonical_terms)
            user_comparison.append(row)

        # Sort by total mentions
        user_comparison.sort(key=lambda x: x["total"], reverse=True)

        data = {
            "terms": canonical_terms,  # Use canonical forms
            "original_terms": terms,
            "searched_forms": term_forms_map,
            "by_user": user_comparison[:limit],
            "totals": {
                term: sum(comparison[term].values())
                for term in canonical_terms
            }
        }

        return ToolResult("compare_term_mentions", True, data)

    def _get_user_stats(self, params: dict) -> ToolResult:
        """Get stats for a specific user."""
        username = params.get("username", "").lstrip("@")

        # Check if it's a user_id (e.g., "928442575" or "User#928442575")
        user_id_match = None
        if username.startswith("User#"):
            try:
                user_id_match = int(username[5:])
            except ValueError:
                pass
        elif username.isdigit():
            user_id_match = int(username)

        # Find user
        users = self.db.get_all_users()
        user_match = None

        if user_id_match:
            # Search by user_id
            for uid, uname in users:
                if uid == user_id_match:
                    user_match = (uid, uname)
                    break
        else:
            # Search by username
            for uid, uname in users:
                if uname and uname.lower() == username.lower():
                    user_match = (uid, uname)
                    break

        if not user_match:
            return ToolResult("get_user_stats", False, None, f"User '{username}' not found")

        user_id, actual_username = user_match
        stats = self.db.get_user_message_stats(user_id)

        # Use display name override if available
        display_name = _get_display_name(user_id, actual_username)

        data = {
            "user_id": user_id,
            "username": display_name,
            "message_count": stats.get("message_count", 0),
            "first_message": stats.get("first_message").isoformat() if stats.get("first_message") else None,
            "last_message": stats.get("last_message").isoformat() if stats.get("last_message") else None,
        }

        return ToolResult("get_user_stats", True, data)

    def _get_user_messages(self, params: dict) -> ToolResult:
        """Get recent messages from a specific user."""
        user_identifier = params.get("user_identifier", "").lstrip("@")
        limit = params.get("limit", 10)

        # Check if it's a user_id (e.g., "928442575" or "User#928442575")
        user_id_match = None
        if user_identifier.startswith("User#"):
            try:
                user_id_match = int(user_identifier[5:])
            except ValueError:
                pass
        elif user_identifier.startswith("User"):
            try:
                user_id_match = int(user_identifier[4:])
            except ValueError:
                pass
        elif user_identifier.isdigit():
            user_id_match = int(user_identifier)

        # Find user
        users = self.db.get_all_users()
        user_match = None

        if user_id_match:
            # Search by user_id
            for uid, uname in users:
                if uid == user_id_match:
                    user_match = (uid, uname)
                    break
        else:
            # Search by username
            for uid, uname in users:
                if uname and uname.lower() == user_identifier.lower():
                    user_match = (uid, uname)
                    break

        if not user_match:
            return ToolResult("get_user_messages", False, None, f"User '{user_identifier}' not found")

        user_id, actual_username = user_match

        # Get messages from database
        messages = self.db.get_messages_by_user(user_id, limit=limit)

        if not messages:
            display_name = _get_display_name(user_id, actual_username)
            return ToolResult("get_user_messages", True, {
                "user": display_name,
                "user_id": user_id,
                "messages": [],
                "total_found": 0,
                "message": f"Користувач {display_name} не має повідомлень в базі"
            })

        # Format messages
        display_name = _get_display_name(user_id, actual_username)
        formatted = []
        for msg in messages:
            formatted.append({
                "text": msg.text[:300] if msg.text else "",
                "date": msg.formatted_date,
                "message_id": msg.message_id
            })

        data = {
            "user": display_name,
            "user_id": user_id,
            "messages": formatted,
            "total_found": len(messages)
        }

        return ToolResult("get_user_messages", True, data)


class ToolAgent:
    """
    AI agent that uses tools to answer questions.
    """

    def __init__(self, db: Database, vector_store: VectorStore, chat_service: ChatService = None):
        self.executor = ToolExecutor(db, vector_store)
        self.chat_service = chat_service or ChatService()

    async def answer(self, question: str, max_iterations: int = 3) -> str:
        """
        Answer a question using available tools.

        The AI will:
        1. Analyze the question
        2. Decide which tools to call
        3. Execute tools and collect results
        4. Synthesize a final answer
        """
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": question
            }
        ]

        tool_results = []

        for iteration in range(max_iterations):
            # Ask AI what to do
            response = await self.chat_service.complete_with_tools_async(
                messages=messages,
                tools=TOOLS_SCHEMA
            )

            # Check if AI wants to call tools
            if response.get("tool_calls"):
                for tool_call in response["tool_calls"]:
                    tool_name = tool_call["name"]
                    params = tool_call.get("arguments", {})

                    logger.info(f"Executing tool: {tool_name}({params})")

                    result = self.executor.execute(tool_name, params)
                    tool_results.append(result)

                    # Add tool result to conversation
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.get("id", f"call_{tool_name}"),
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(params, ensure_ascii=False)
                            }
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", f"call_{tool_name}"),
                        "content": json.dumps(result.data if result.success else {"error": result.error}, ensure_ascii=False)
                    })
            else:
                # AI is done calling tools, return final answer
                return response.get("content", "Не вдалося отримати відповідь.")

        # Max iterations reached, ask for final answer
        messages.append({
            "role": "user",
            "content": "Based on the tool results above, provide your final answer in Ukrainian."
        })

        final_response = await self.chat_service.complete_with_tools_async(
            messages=messages,
            tools=[]  # No more tools
        )

        return final_response.get("content", "Не вдалося отримати відповідь.")

    def _get_system_prompt(self) -> str:
        return """Ти аналітик чату. Відповідай на питання використовуючи доступні інструменти.

Доступні інструменти:
- count_term_mentions: підрахувати скільки разів кожен користувач згадував термін
- get_top_speakers: отримати топ користувачів за кількістю повідомлень
- search_messages: семантичний пошук повідомлень (можна фільтрувати по user_id)
- compare_term_mentions: порівняти згадування різних термінів
- get_user_stats: статистика конкретного користувача (по username або user_id)
- get_user_messages: отримати останні повідомлення від користувача (для "покажи повідомлення від User#123")

ВАЖЛИВО про аліаси:
Система автоматично розширює сленг/прізвиська до всіх форм:
- "зелупа", "зе", "зеля" → шукає всі форми включно з "Зеленський"
- "порох", "петя", "барига" → шукає всі форми включно з "Порошенко"
- "біток", "btc" → шукає всі форми включно з "біткоін"
Використовуй терміни як їх написав користувач - система сама знайде всі варіанти.

ВАЖЛИВО про користувачів:
- Деякі користувачі мають тільки user_id без username (наприклад, User#928442575)
- Для пошуку по таких користувачах використовуй числовий user_id як user_filter
- Для search_messages: user_filter="928442575" або user_filter="User#928442575"
- Для get_user_stats: username="928442575" або username="User#928442575"

Правила:
1. Для порівняльних питань ("хто більше X чи Y") використовуй compare_term_mentions
2. Для питань "хто частіше згадує X" використовуй count_term_mentions
3. Для питань "хто найактивніший" використовуй get_top_speakers
4. Для "покажи повідомлення від User#123" використовуй get_user_messages з user_identifier="123"
5. Відповідай українською мовою
5. НЕ використовуй Markdown (**, ##, тощо). Використовуй тільки:
   - Емодзі для візуального виділення (📊, 👤, 🏆, 📈, ❌)
   - Прості списки з цифрами (1. 2. 3.)
   - Тире для пунктів
6. Форматуй відповідь зрозуміло з цифрами та іменами
7. В результатах вказуй канонічну форму (Зеленський, Порошенко) для ясності
8. Якщо результатів не знайдено (total_found=0), ОБОВ'ЯЗКОВО повідом про це:
   "❌ Нічого не знайдено за запитом X" або "❌ Користувач Y не має повідомлень на цю тему"

Приклад формату:
📊 Згадки "Зеленський" (зе, зеля, зелупа...):
1. 👤 Username — 100 разів
2. 👤 Username2 — 50 разів

Приклад коли немає результатів:
❌ Нічого не знайдено за запитом "тема X"

Після отримання результатів від інструментів, надай чітку відповідь користувачу."""
