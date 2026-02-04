"""
AI-powered question parser for natural language queries.
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from search.embeddings import ChatService
from search.entity_aliases import expand_query_with_aliases, get_canonical, ENTITY_ALIASES

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
    self_mentioned_as_subject: bool = False # "про мене" - searching for messages ABOUT the requester


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
6. EXPAND SLANG/NICKNAMES to their full forms (see below)

COMMON SLANG AND NICKNAMES - always include canonical form:
- "зе", "зеля", "зелупа" → include "Зеленський"
- "порох", "петя", "рошен" → include "Порошенко"
- "пуйло", "хуйло", "бункерний" → include "Путін"
- "біток", "btc" → include "біткоін bitcoin криптовалюта"
- "крипта" → include "криптовалюта bitcoin"
- "рашка", "мордор" → include "росія"
- "орки", "кацапи" → include "росіяни"

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


# Self-reference patterns: "I said" (self as speaker)
SELF_AS_SPEAKER_PATTERNS = [
    # Ukrainian: я казав, я писав, я говорив
    r'\bя\s+каза', r'\bя\s+писа', r'\bя\s+говори', r'\bя\s+думав', r'\bя\s+вважа',
    r'\bчи\s+я\b', r'\bколи\s+я\b', r'\bде\s+я\b', r'\bщо\s+я\b',
    # Russian: я говорил, я писал, я сказал
    r'\bя\s+говори', r'\bя\s+писа', r'\bя\s+сказа', r'\bя\s+дума', r'\bя\s+счита',
    r'\bговорил\s+ли\s+я\b', r'\bписал\s+ли\s+я\b', r'\bсказал\s+ли\s+я\b',
    r'\bкогда\s+я\b', r'\bгде\s+я\b', r'\bчто\s+я\b',
    # English
    r'\bi\s+said', r'\bi\s+wrote', r'\bdid\s+i\s+say', r'\bhave\s+i\b',
    r'\bwhen\s+i\b', r'\bwhere\s+i\b', r'\bwhat\s+i\b',
]

# Self-reference patterns: "about me" (self as subject/target)
SELF_AS_SUBJECT_PATTERNS = [
    # Ukrainian: про мене, обо мені, мене згадував
    r'\bпро\s+мене\b', r'\bобо\s+мен[еі]\b', r'\bмене\s+згадув', r'\bщодо\s+мене\b',
    r'\bна\s+мене\b', r'\bмені\s+казав', r'\bмені\s+говорив', r'\bмені\s+писав',
    # Russian: обо мне, про меня, меня упоминал
    r'\bобо\s+мне\b', r'\bпро\s+меня\b', r'\bменя\s+упомин', r'\bнасчет\s+меня\b',
    r'\bо\s+мне\b', r'\bна\s+меня\b', r'\bмне\s+говорил', r'\bмне\s+писал', r'\bмне\s+сказал',
    r'\bменя\s+называ', r'\bменя\s+обсужда',
    # English
    r'\babout\s+me\b', r'\bmentioned\s+me\b', r'\btalked\s+about\s+me\b',
    r'\bcalled\s+me\b', r'\btold\s+me\b',
]

# General self-reference (fallback)
SELF_REFERENCE_PATTERNS = [
    # Ukrainian
    r'\bя\b', r'\bмене\b', r'\bмені\b', r'\bмною\b', r'\bмій\b', r'\bмоя\b', r'\bмоє\b', r'\bмої\b',
    # Russian
    r'\bменя\b', r'\bмне\b', r'\bмной\b', r'\bмоё\b',
    # English
    r'\bi\b', r'\bme\b', r'\bmy\b', r'\bmyself\b',
]

SELF_AS_SPEAKER_REGEX = re.compile('|'.join(SELF_AS_SPEAKER_PATTERNS), re.IGNORECASE)
SELF_AS_SUBJECT_REGEX = re.compile('|'.join(SELF_AS_SUBJECT_PATTERNS), re.IGNORECASE)
SELF_REFERENCE_REGEX = re.compile('|'.join(SELF_REFERENCE_PATTERNS), re.IGNORECASE)


class QuestionParser:
    """Parse natural language questions about chat history."""

    def __init__(self):
        self.chat_service = ChatService()

    def _detect_self_reference(self, question: str) -> tuple[bool, bool, bool]:
        """
        Check if the question contains self-references.

        Returns:
            (has_self_ref, is_speaker, is_subject)
            - has_self_ref: Any self-reference found
            - is_speaker: "я казав" - self as the message author
            - is_subject: "про мене" - searching for messages ABOUT self
        """
        is_speaker = bool(SELF_AS_SPEAKER_REGEX.search(question))
        is_subject = bool(SELF_AS_SUBJECT_REGEX.search(question))
        has_any = is_speaker or is_subject or bool(SELF_REFERENCE_REGEX.search(question))

        return has_any, is_speaker, is_subject

    def _find_user_by_telegram_id(
        self,
        telegram_user_id: int,
        users: list
    ) -> Optional[tuple[int, str]]:
        """
        Find a user in the chat history by their Telegram user_id.

        Note: telegram_user_id is the ID of the person asking the question.
        We need to find if they have messages in our history.
        """
        for user_id, username in users:
            if user_id == telegram_user_id:
                return (user_id, username)
        return None

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
        users: list,
        requesting_user_id: int = None
    ) -> ParsedQuestion:
        """
        Parse a question using AI to extract structured info.

        Args:
            question: The question text
            aliases_dict: User aliases mapping
            users: List of (user_id, username) tuples from chat history
            requesting_user_id: Telegram user_id of the person asking (for self-reference resolution)
        """
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

        # Handle self-references (я, мене, I, me, etc.)
        self_mentioned_as_subject = False
        self_name_for_query = None

        if requesting_user_id:
            has_self_ref, is_speaker, is_subject = self._detect_self_reference(question)

            if has_self_ref:
                requesting_user = self._find_user_by_telegram_id(requesting_user_id, users)

                if requesting_user:
                    user_id, username = requesting_user

                    if is_subject:
                        # "про мене" - searching for messages ABOUT the requester
                        # Don't filter by user_id, but add their name to search query
                        self_mentioned_as_subject = True
                        self_name_for_query = username
                        logger.info(f"Self-as-subject detected: will search for messages about '{username}'")

                    elif is_speaker or not mentioned_users:
                        # "я казав" or generic self-reference with no other users
                        # Filter by the requester's user_id
                        if requesting_user not in mentioned_users:
                            mentioned_users.insert(0, requesting_user)
                            logger.info(f"Self-as-speaker detected: filtering by {username} (ID: {user_id})")
                else:
                    logger.warning(f"Self-reference detected but user {requesting_user_id} not found in chat history")

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

        # Expand entity aliases (зе -> Зеленський, порох -> Порошенко, etc.)
        search_query_expanded = expand_query_with_aliases(search_query)
        if search_query_expanded != search_query:
            logger.info(f"Expanded aliases: '{search_query}' -> '{search_query_expanded}'")
            search_query = search_query_expanded

        # Add self's name to search query if "про мене" pattern detected
        if self_name_for_query and self_name_for_query.lower() not in search_query.lower():
            search_query = f"{search_query} {self_name_for_query}"
            logger.info(f"Added self name to query: '{search_query}'")

        logger.info(f"Parsed question: '{question}' -> search_query: '{search_query}'")

        return ParsedQuestion(
            mentioned_users=mentioned_users,
            search_query=search_query,
            question_type=raw_response.get("question_type", "quote_search"),
            original_question=question,
            raw_response=raw_response,
            date_from=date_from,
            date_to=date_to,
            sort_order=sort_order,
            self_mentioned_as_subject=self_mentioned_as_subject
        )

    def parse(
        self,
        question: str,
        aliases_dict: dict,
        users: list,
        requesting_user_id: int = None
    ) -> ParsedQuestion:
        """Sync version of parse."""
        import asyncio
        return asyncio.run(self.parse_async(question, aliases_dict, users, requesting_user_id))

    async def generate_suggestions_async(
        self,
        question: str,
        results: list[dict],
        max_suggestions: int = 3
    ) -> list[str]:
        """
        Generate follow-up question suggestions based on search results.

        Args:
            question: Original question
            results: Search results with text and metadata
            max_suggestions: Maximum number of suggestions to generate

        Returns:
            List of suggested follow-up questions
        """
        if not results:
            return []

        # Extract topics/keywords from results
        topics = set()
        usernames = set()
        for result in results[:5]:
            meta = result.get("metadata", {})
            username = meta.get("display_name")
            if username:
                usernames.add(username)
            # Simple keyword extraction from text
            text = result.get("text", "")[:200]
            words = [w for w in text.split() if len(w) > 4 and w.isalpha()]
            topics.update(words[:3])

        topics_str = ", ".join(list(topics)[:10])
        users_str = ", ".join(list(usernames)[:5])

        prompt = f"""Based on this search question and found topics, suggest 2-3 related follow-up questions in the same language as the original question.

Original question: {question}
Related topics found: {topics_str}
Users mentioned: {users_str}

Generate SHORT follow-up questions (under 50 chars) that would help explore the topic further.
Questions should be natural and useful for chat history search.

Respond ONLY with JSON array of strings:
["question1", "question2", "question3"]"""

        system = """You generate search suggestions for a chat history bot.
Rules:
1. Match the language of the original question (Ukrainian, Russian, or English)
2. Keep suggestions short (under 50 characters)
3. Make them specific and actionable
4. Focus on "who said what" or "when did they discuss" patterns
5. Return ONLY valid JSON array"""

        try:
            response = await self.chat_service.complete_async(
                prompt=prompt,
                system=system,
                max_tokens=200
            )

            # Parse JSON response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                suggestions = json.loads(json_match.group())
                return suggestions[:max_suggestions]

        except Exception as e:
            logger.warning(f"Failed to generate suggestions: {e}")

        return []
