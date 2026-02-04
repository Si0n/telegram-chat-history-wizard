"""
AI-powered question parser for natural language queries.
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from search.embeddings import ChatService

logger = logging.getLogger(__name__)


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


SYSTEM_PROMPT = """You are a semantic query expander for a Telegram chat history search bot.
Your task is to convert user questions into RICH SEMANTIC search queries that will find relevant messages.

The chat has these users with their nicknames:
{users_context}

CRITICAL: The search uses EMBEDDING SIMILARITY. You must create a query that will be SEMANTICALLY CLOSE to the messages we want to find.

Think: "What words would actually appear in messages we're looking for?"

EXAMPLES:

Question: "чи гусь казав що порошенко дебіл?"
Intent: Find messages where someone criticizes/insults Poroshenko
Messages might contain: "Порошенко ідіот", "він дурень", "поганий президент", "некомпетентний"
search_query: "Порошенко поганий дурний ідіот некомпетентний президент критика"

Question: "що буш думає про біткоін?"
Intent: Find opinions about bitcoin
Messages might contain: "біткоін буде рости", "крипта скам", "треба купувати", "я вірю в btc"
search_query: "біткоін bitcoin крипта криптовалюта інвестиція купувати продавати рости падати думаю вважаю"

Question: "чи серж хвалив зеленського?"
Intent: Find positive statements about Zelensky
Messages might contain: "Зеленський молодець", "він робить правильно", "підтримую", "хороший президент"
search_query: "Зеленський молодець хороший правильно підтримую президент позитив"

Question: "хто говорив про війну?"
Intent: Find any war-related messages
Messages might contain: "війна", "Україна", "обстріл", "фронт", "ЗСУ", "окупанти", "перемога"
search_query: "війна Україна обстріл фронт ЗСУ армія бойові перемога окупанти"

Question: "що казали про гроші та фінанси?"
Intent: Find financial discussions
Messages might contain: "зарплата", "ціни", "курс долара", "інфляція", "інвестиції"
search_query: "гроші зарплата ціни курс долар євро інфляція інвестиції фінанси бюджет економіка"

RULES:
1. Include the MAIN TOPIC (person, thing, event)
2. Add SYNONYMS and RELATED WORDS in both Ukrainian and Russian if relevant
3. Include words that would ACTUALLY APPEAR in relevant messages
4. Think about HOW people discuss this topic in casual chat
5. Use 8-15 words to create a rich semantic field

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

        search_query = raw_response.get("search_query", question)

        logger.info(f"Parsed question: '{question}' -> search_query: '{search_query}'")

        return ParsedQuestion(
            mentioned_users=mentioned_users,
            search_query=search_query,
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
