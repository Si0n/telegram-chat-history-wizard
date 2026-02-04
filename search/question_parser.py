"""
AI-powered question parser for natural language queries.
"""
import json
import re
from dataclasses import dataclass
from typing import Optional

from search.embeddings import ChatService


@dataclass
class ParsedQuestion:
    """Structured representation of a parsed question."""
    mentioned_users: list[tuple[int, str]]  # [(user_id, username), ...]
    search_query: str                        # What to search for
    question_type: str                       # quote_search, yes_no, summary, comparison
    original_question: str
    raw_response: dict                       # Full AI response for debugging
    date_from: Optional[str] = None         # ISO date string YYYY-MM-DD
    date_to: Optional[str] = None           # ISO date string YYYY-MM-DD
    sort_order: str = "relevance"           # relevance, oldest, newest


SYSTEM_PROMPT = """You are a semantic query parser for a Telegram chat history search bot.
Your task is to convert user questions in Ukrainian/Russian into effective SEMANTIC search queries.

The chat has these users with their nicknames:
{users_context}

CRITICAL: The search uses semantic/vector similarity, NOT keyword matching.
You must extract the MEANING and INTENT of the question, not just keywords.

EXAMPLES of good search_query extraction:

Question: "чи гусь казав що порошенко дебіл?"
BAD search_query: "Порошенко" (too narrow, just a name)
GOOD search_query: "Порошенко дурний поганий критика образа негативна думка" (captures the semantic intent - looking for insults/criticism)

Question: "що буш думає про біткоін?"
BAD search_query: "біткоін" (misses the opinion aspect)
GOOD search_query: "біткоін думка погляд інвестиція крипта хороший поганий" (captures opinions about bitcoin)

Question: "чи серж хвалив зеленського?"
BAD search_query: "Зеленський"
GOOD search_query: "Зеленський хороший молодець підтримка позитив похвала" (looking for praise)

Question: "коли гусь лаявся на когось?"
BAD search_query: "лаявся"
GOOD search_query: "лайка образа мат злість конфлікт сварка" (semantic field of swearing/conflict)

RULES for search_query:
1. Include the main subject/topic (person, thing being discussed)
2. Add SEMANTIC synonyms that capture the INTENT (criticism → поганий, дурний, критика, негатив)
3. Include related concepts that might appear in such messages
4. Write in the same language as the question (Ukrainian/Russian)
5. Use 5-10 words that form a semantic cluster around the intent

IMPORTANT for nickname matching:
- Match variations: "гусь", "гуся", "гусем" → the user with "гусь" alias
- Match "птиц*" patterns: "птица", "птичка" → Лех Качинський
- Case-insensitive matching

DATE CONSTRAINTS:
Extract date ranges if mentioned in the question:
- "до 2022" → date_to: "2022-01-01"
- "після 2023" → date_from: "2023-01-01"
- "в 2021 році" → date_from: "2021-01-01", date_to: "2021-12-31"
- "за останній рік" → date_from: one year ago from today
- "минулого літа" → approximate date range
- If no date mentioned → null for both

Today's date is 2026-02-04. Use this for relative dates like "останній рік", "минулого місяця".

SORTING:
Detect if user wants specific order:
- "старі спочатку", "від старих", "хронологічно", "oldest first" → sort_order: "oldest"
- "нові спочатку", "від нових", "останні", "newest first" → sort_order: "newest"
- Default (by relevance) → sort_order: "relevance"

Respond ONLY with valid JSON:
{{
    "mentioned_users": ["username1"],
    "search_query": "semantic search query with synonyms and related concepts",
    "question_type": "quote_search|yes_no|summary|comparison",
    "confidence": "high|medium|low",
    "date_from": "YYYY-MM-DD or null",
    "date_to": "YYYY-MM-DD or null",
    "sort_order": "relevance|oldest|newest"
}}

Question types:
- quote_search: find specific messages/quotes
- yes_no: answer a yes/no question about what someone said
- summary: summarize opinions/statements on a topic
- comparison: compare what different users said"""


class QuestionParser:
    """Parse natural language questions about chat history."""

    def __init__(self):
        self.chat_service = ChatService()

    def _build_users_context(self, aliases_dict: dict, users: list) -> str:
        """Build context string describing users and their aliases."""
        # Group aliases by user
        user_aliases = {}
        for alias, (user_id, username) in aliases_dict.items():
            if user_id not in user_aliases:
                user_aliases[user_id] = {"username": username, "aliases": []}
            user_aliases[user_id]["aliases"].append(alias)

        # Add users without aliases
        for user_id, username in users:
            if user_id not in user_aliases:
                user_aliases[user_id] = {"username": username, "aliases": []}

        # Format as readable text
        lines = []
        for user_id, info in user_aliases.items():
            aliases_str = ", ".join(info["aliases"]) if info["aliases"] else "no aliases"
            lines.append(f"- {info['username']} (ID: {user_id}): {aliases_str}")

        return "\n".join(lines) if lines else "No users configured"

    def _resolve_usernames(
        self,
        usernames: list[str],
        aliases_dict: dict,
        users: list
    ) -> list[tuple[int, str]]:
        """Resolve mentioned usernames/aliases to (user_id, username) pairs."""
        resolved = []
        username_lower_map = {u[1].lower(): u for u in users if u[1]}

        for name in usernames:
            name_lower = name.lower().lstrip("@")

            # Check aliases first
            if name_lower in aliases_dict:
                user_id, username = aliases_dict[name_lower]
                if (user_id, username) not in resolved:
                    resolved.append((user_id, username))
                continue

            # Check direct username match
            if name_lower in username_lower_map:
                if username_lower_map[name_lower] not in resolved:
                    resolved.append(username_lower_map[name_lower])
                continue

            # Fuzzy match on usernames (partial match)
            for username_lc, user_tuple in username_lower_map.items():
                if name_lower in username_lc or username_lc in name_lower:
                    if user_tuple not in resolved:
                        resolved.append(user_tuple)
                    break

        return resolved

    async def parse_async(
        self,
        question: str,
        aliases_dict: dict,
        users: list
    ) -> ParsedQuestion:
        """Parse a question using AI to extract structured info."""
        users_context = self._build_users_context(aliases_dict, users)
        system_prompt = SYSTEM_PROMPT.format(users_context=users_context)

        response_text = await self.chat_service.complete_async(
            prompt=f"Parse this question: {question}",
            system=system_prompt,
            max_tokens=500
        )

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                raw_response = json.loads(json_match.group())
            else:
                raw_response = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback: treat entire question as search query
            raw_response = {
                "mentioned_users": [],
                "search_query": question,
                "question_type": "quote_search",
                "confidence": "low"
            }

        # Resolve usernames to actual user tuples
        mentioned_users = self._resolve_usernames(
            raw_response.get("mentioned_users", []),
            aliases_dict,
            users
        )

        # Extract date constraints (normalize null/None)
        date_from = raw_response.get("date_from")
        date_to = raw_response.get("date_to")
        if date_from in ("null", "None", None, ""):
            date_from = None
        if date_to in ("null", "None", None, ""):
            date_to = None

        # Extract sort order
        sort_order = raw_response.get("sort_order", "relevance")
        if sort_order not in ("relevance", "oldest", "newest"):
            sort_order = "relevance"

        return ParsedQuestion(
            mentioned_users=mentioned_users,
            search_query=raw_response.get("search_query", question),
            question_type=raw_response.get("question_type", "quote_search"),
            original_question=question,
            raw_response=raw_response,
            date_from=date_from,
            date_to=date_to,
            sort_order=sort_order
        )

    def parse(
        self,
        question: str,
        aliases_dict: dict,
        users: list
    ) -> ParsedQuestion:
        """Sync version of parse."""
        import asyncio
        return asyncio.run(self.parse_async(question, aliases_dict, users))
