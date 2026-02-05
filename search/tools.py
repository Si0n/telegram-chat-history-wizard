"""
Tool-based analytics system.
Allows the AI to call database/search tools to answer complex questions.
"""
import json
import logging
from typing import Any
from dataclasses import dataclass

from db import Database
from search.vector_store import VectorStore
from search.embeddings import ChatService

logger = logging.getLogger(__name__)


# Tool definitions for the AI
TOOLS_SCHEMA = [
    {
        "name": "count_term_mentions",
        "description": "Count how many times each user mentioned a specific term/word in their messages. Use for questions like 'who mentions X most often'.",
        "parameters": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "The term/word to search for (e.g., 'Зеленський', 'біткоін')"
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
                    "description": "Optional: filter by username"
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
        "description": "Compare how often different terms are mentioned by users. Use for comparison questions like 'who mentions X more vs Y'.",
        "parameters": {
            "type": "object",
            "properties": {
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of terms to compare (e.g., ['Зеленський', 'Порошенко'])"
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
        "description": "Get detailed statistics for a specific user.",
        "parameters": {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "Username to look up"
                }
            },
            "required": ["username"]
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
            else:
                return ToolResult(tool_name, False, None, f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return ToolResult(tool_name, False, None, str(e))

    def _count_term_mentions(self, params: dict) -> ToolResult:
        """Count term mentions by user."""
        term = params.get("term", "")
        limit = params.get("limit", 10)

        results = self.db.get_term_mention_counts(term, limit=limit)

        data = {
            "term": term,
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

        if user_filter:
            results = self.vector_store.search_by_user(
                query=query,
                user_identifier=user_filter,
                n_results=limit
            )
        else:
            results = self.vector_store.search(
                query=query,
                n_results=limit
            )

        data = {
            "query": query,
            "results": [
                {
                    "text": r.get("text", "")[:300],
                    "username": r.get("metadata", {}).get("display_name", "Unknown"),
                    "date": r.get("metadata", {}).get("formatted_date", ""),
                    "similarity": round(r.get("similarity", 0), 3)
                }
                for r in results
            ],
            "total_found": len(results)
        }

        return ToolResult("search_messages", True, data)

    def _compare_term_mentions(self, params: dict) -> ToolResult:
        """Compare mentions of multiple terms."""
        terms = params.get("terms", [])
        limit = params.get("limit", 10)

        comparison = {}
        all_users = set()

        for term in terms:
            results = self.db.get_term_mention_counts(term, limit=limit)
            comparison[term] = {
                username: count
                for user_id, username, count in results
            }
            all_users.update(comparison[term].keys())

        # Build comparison table
        user_comparison = []
        for username in all_users:
            row = {"username": username}
            for term in terms:
                row[term] = comparison[term].get(username, 0)
            row["total"] = sum(row.get(term, 0) for term in terms)
            user_comparison.append(row)

        # Sort by total mentions
        user_comparison.sort(key=lambda x: x["total"], reverse=True)

        data = {
            "terms": terms,
            "by_user": user_comparison[:limit],
            "totals": {
                term: sum(comparison[term].values())
                for term in terms
            }
        }

        return ToolResult("compare_term_mentions", True, data)

    def _get_user_stats(self, params: dict) -> ToolResult:
        """Get stats for a specific user."""
        username = params.get("username", "").lstrip("@")

        # Find user
        users = self.db.get_all_users()
        user_match = None
        for user_id, uname in users:
            if uname and uname.lower() == username.lower():
                user_match = (user_id, uname)
                break

        if not user_match:
            return ToolResult("get_user_stats", False, None, f"User '{username}' not found")

        user_id, actual_username = user_match
        stats = self.db.get_user_message_stats(user_id)

        data = {
            "username": actual_username,
            "message_count": stats.get("message_count", 0),
            "first_message": stats.get("first_message").isoformat() if stats.get("first_message") else None,
            "last_message": stats.get("last_message").isoformat() if stats.get("last_message") else None,
        }

        return ToolResult("get_user_stats", True, data)


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
- search_messages: семантичний пошук повідомлень
- compare_term_mentions: порівняти згадування різних термінів
- get_user_stats: статистика конкретного користувача

Правила:
1. Для порівняльних питань ("хто більше X чи Y") використовуй compare_term_mentions
2. Для питань "хто частіше згадує X" використовуй count_term_mentions
3. Для питань "хто найактивніший" використовуй get_top_speakers
4. Відповідай українською мовою
5. Форматуй відповідь зрозуміло з цифрами та іменами

Після отримання результатів від інструментів, надай чітку відповідь користувачу."""
