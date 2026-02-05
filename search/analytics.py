"""
Analytics engine for aggregation queries.
Handles both quantitative (database counts) and behavioral (AI analysis) questions.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import config
from db import Database
from search.vector_store import VectorStore
from search.embeddings import ChatService
from search.intent_detection import extract_analytics_type, extract_trait_from_question
from search.entity_aliases import get_all_forms, get_canonical

logger = logging.getLogger(__name__)


class AnalyticsType(Enum):
    """Type of analytics query."""
    QUANTITATIVE = "quantitative"  # Database aggregation
    BEHAVIORAL = "behavioral"       # AI analysis of content


@dataclass
class AnalyticsResult:
    """Result of an analytics query."""
    answer: str
    analytics_type: AnalyticsType
    stats: list[dict]
    total_analyzed: int = 0


class AnalyticsEngine:
    """
    Analytics engine for answering aggregation and behavioral questions.

    Handles:
    - "Who talks most?" -> count messages per user
    - "Who mentioned X most?" -> count term occurrences
    - "Who is more angry?" -> AI analysis of message tone
    """

    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        chat_service: ChatService = None
    ):
        self.db = db
        self.vector_store = vector_store
        self.chat_service = chat_service or ChatService()

    # === Quantitative Analytics ===

    async def get_top_speakers(
        self,
        limit: int = None,
        date_from: str = None,
        date_to: str = None
    ) -> list[dict]:
        """
        Who talks the most? Return user stats sorted by message count.

        Returns:
            List of dicts with user_id, display_name, message_count
        """
        limit = limit or config.ANALYTICS_TOP_LIMIT

        results = self.db.get_message_count_by_user(
            date_from=date_from,
            date_to=date_to,
            limit=limit
        )

        return [
            {
                "user_id": user_id,
                "display_name": display_name,
                "message_count": count,
                "rank": i + 1
            }
            for i, (user_id, display_name, count) in enumerate(results)
        ]

    async def get_mention_counts(
        self,
        term: str,
        limit: int = None
    ) -> list[dict]:
        """
        Who mentioned X the most? Automatically expands aliases.

        Args:
            term: The term to search for (can be any alias form)
            limit: Max results to return

        Returns:
            List of dicts with user_id, display_name, mention_count, canonical_term
        """
        limit = limit or config.ANALYTICS_TOP_LIMIT

        # Expand term to all alias forms
        all_forms = get_all_forms(term)
        canonical = get_canonical(term)

        logger.info(f"Analytics: searching term '{term}' expanded to: {all_forms}")

        # Search for all forms at once
        results = self.db.get_term_mention_counts_multi(all_forms, limit=limit)

        return [
            {
                "user_id": user_id,
                "display_name": display_name,
                "mention_count": count,
                "rank": i + 1,
                "canonical_term": canonical,
                "searched_forms": all_forms
            }
            for i, (user_id, display_name, count) in enumerate(results)
        ]

    async def get_user_stats(self, user_id: int) -> dict:
        """Get comprehensive stats for a specific user."""
        return self.db.get_user_message_stats(user_id)

    # === Behavioral Analytics ===

    def _get_trait_search_queries(self, trait: str) -> list[str]:
        """Get search queries for a behavioral trait."""
        trait_queries = {
            "angry": [
                "злий", "бісить", "дратує", "ненавиджу", "задовбали",
                "злой", "бесит", "достали", "ненавижу", "заебали"
            ],
            "strict": [
                "мусиш", "повинен", "обов'язково", "заборонено",
                "должен", "нельзя", "обязан", "запрещено"
            ],
            "psycho": [
                "божевільний", "шизо", "параноя", "психоз",
                "сумасшедший", "шиза", "параноик"
            ],
            "swears": [
                "блять", "сука", "хуй", "пиздец", "нахуй",
                "ебать", "пизда", "хуйня"
            ],
            "positive": [
                "супер", "чудово", "молодець", "класно", "прекрасно",
                "отлично", "круто", "замечательно", "ура"
            ],
            "negative": [
                "погано", "жахливо", "кошмар", "огидно",
                "ужасно", "плохо", "дерьмо", "отвратительно"
            ],
            "kind": [
                "дякую", "вдячний", "допомогти", "підтримую",
                "спасибо", "благодарю", "помочь", "поддерживаю"
            ],
            "aggressive": [
                "йди нахуй", "заткнись", "ідіот", "дебіл",
                "иди нахуй", "заткнись", "идиот", "дебил"
            ],
            "toxic": [
                "ти тупий", "ви всі", "ніхто не розуміє", "всі дурні",
                "ты тупой", "вы все", "никто не понимает", "все дураки"
            ],
        }
        return trait_queries.get(trait, [trait])

    async def _score_trait(self, trait: str, messages: list[str]) -> float:
        """
        Use AI to score how much messages exhibit a trait (0-10).

        Args:
            trait: The trait name
            messages: Sample messages to analyze

        Returns:
            Score from 0-10
        """
        if not messages:
            return 0.0

        # Take up to 10 messages for analysis
        sample = messages[:10]
        messages_text = "\n".join(f"- {m[:200]}" for m in sample)

        prompt = f"""Проаналізуй ці повідомлення та оціни, наскільки вони демонструють характеристику "{trait}" за шкалою 0-10.
0 = зовсім не демонструють
10 = дуже сильно демонструють

Повідомлення:
{messages_text}

Відповідь дай ЛИШЕ числом від 0 до 10, без пояснень."""

        try:
            response = await self.chat_service.complete_async(
                prompt=prompt,
                system="You are a text analyzer. Respond only with a single number 0-10.",
                max_tokens=10
            )
            # Parse the score
            score_str = response.strip().split()[0]
            score = float(score_str)
            return min(max(score, 0.0), 10.0)  # Clamp to 0-10
        except Exception as e:
            logger.warning(f"Trait scoring failed for {trait}: {e}")
            return 0.0

    async def analyze_behavioral_trait(
        self,
        trait: str,
        limit: int = 5
    ) -> list[dict]:
        """
        Find users who exhibit a behavioral trait most.

        1. Search for messages matching trait keywords
        2. Group by user
        3. Use AI to score trait intensity per user
        4. Return ranked list

        Args:
            trait: The trait to analyze (e.g., "angry", "strict")
            limit: Number of top users to return

        Returns:
            List of dicts with user_id, display_name, score, example_count
        """
        # Step 1: Search for trait-related messages
        trait_queries = self._get_trait_search_queries(trait)

        all_results = []
        seen_ids = set()

        for query in trait_queries:
            try:
                results = self.vector_store.search(
                    query=query,
                    n_results=50
                )
                for r in results:
                    vec_id = r.get("vector_id", "")
                    if vec_id not in seen_ids:
                        seen_ids.add(vec_id)
                        all_results.append(r)
            except Exception as e:
                logger.warning(f"Search failed for trait query '{query}': {e}")

        if not all_results:
            return []

        # Step 2: Group by user
        user_messages = defaultdict(list)
        user_display_names = {}

        for r in all_results:
            meta = r.get("metadata", {})
            user_id = meta.get("user_id")
            if user_id:
                user_messages[user_id].append(r.get("text", ""))
                if user_id not in user_display_names:
                    user_display_names[user_id] = meta.get("display_name", f"User#{user_id}")

        # Step 3: AI scoring for each user (limit to users with enough messages)
        user_scores = []
        for user_id, messages in user_messages.items():
            if len(messages) < 2:  # Need at least 2 messages for meaningful analysis
                continue

            score = await self._score_trait(trait, messages)
            user_scores.append({
                "user_id": user_id,
                "display_name": user_display_names.get(user_id, f"User#{user_id}"),
                "score": score,
                "example_count": len(messages)
            })

        # Step 4: Sort and return top users
        user_scores.sort(key=lambda x: x["score"], reverse=True)
        return user_scores[:limit]

    # === Main Entry Point ===

    async def answer_analytics_question(
        self,
        question: str,
        search_term: str = None
    ) -> AnalyticsResult:
        """
        Route analytics question to appropriate handler.

        Args:
            question: The user's question
            search_term: Optional extracted search term

        Returns:
            AnalyticsResult with answer text and stats
        """
        question_lower = question.lower()

        # Detect analytics type
        analytics_type_str = extract_analytics_type(question)
        analytics_type = (
            AnalyticsType.BEHAVIORAL if analytics_type_str == "behavioral"
            else AnalyticsType.QUANTITATIVE
        )

        if analytics_type == AnalyticsType.QUANTITATIVE:
            # Route to count-based analytics
            if any(kw in question_lower for kw in [
                'найактивніш', 'більше пише', 'самый активный',
                'больше пишет', 'most active', 'talks more', 'writes more'
            ]):
                stats = await self.get_top_speakers()
                answer = self._format_top_speakers(stats)
                return AnalyticsResult(
                    answer=answer,
                    analytics_type=analytics_type,
                    stats=stats,
                    total_analyzed=sum(s["message_count"] for s in stats)
                )

            elif any(kw in question_lower for kw in [
                'згадував', 'упоминал', 'mentioned', 'писав про', 'писал про'
            ]):
                # Extract term from search_term or question
                term = search_term or self._extract_search_term(question)
                if term:
                    stats = await self.get_mention_counts(term)
                    answer = self._format_mention_stats(term, stats)
                    return AnalyticsResult(
                        answer=answer,
                        analytics_type=analytics_type,
                        stats=stats,
                        total_analyzed=sum(s["mention_count"] for s in stats)
                    )
                else:
                    return AnalyticsResult(
                        answer="Не вдалося визначити, що саме шукати. Вкажіть термін для пошуку.",
                        analytics_type=analytics_type,
                        stats=[]
                    )

            else:
                # Default to top speakers
                stats = await self.get_top_speakers()
                answer = self._format_top_speakers(stats)
                return AnalyticsResult(
                    answer=answer,
                    analytics_type=analytics_type,
                    stats=stats,
                    total_analyzed=sum(s["message_count"] for s in stats)
                )

        else:
            # Behavioral analysis
            trait = extract_trait_from_question(question)
            if not trait:
                # Try to extract from the question directly
                trait = self._extract_trait_fallback(question)

            if trait:
                stats = await self.analyze_behavioral_trait(trait)
                answer = self._format_behavioral_stats(trait, stats)
                return AnalyticsResult(
                    answer=answer,
                    analytics_type=analytics_type,
                    stats=stats,
                    total_analyzed=sum(s["example_count"] for s in stats)
                )
            else:
                return AnalyticsResult(
                    answer="Не вдалося визначити характеристику для аналізу.",
                    analytics_type=analytics_type,
                    stats=[]
                )

    def _extract_search_term(self, question: str) -> Optional[str]:
        """Extract search term from mention-count questions."""
        import re

        # Patterns like "хто згадував X", "хто писав про X"
        patterns = [
            r'згадував\s+["\']?(.+?)["\']?(?:\s|$|,|\?)',
            r'упоминал\s+["\']?(.+?)["\']?(?:\s|$|,|\?)',
            r'mentioned\s+["\']?(.+?)["\']?(?:\s|$|,|\?)',
            r'писав\s+про\s+["\']?(.+?)["\']?(?:\s|$|,|\?)',
            r'писал\s+про\s+["\']?(.+?)["\']?(?:\s|$|,|\?)',
            r'wrote\s+about\s+["\']?(.+?)["\']?(?:\s|$|,|\?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                term = match.group(1).strip()
                if term and len(term) > 1:
                    return term

        return None

    def _extract_trait_fallback(self, question: str) -> Optional[str]:
        """Fallback trait extraction from question keywords."""
        question_lower = question.lower()

        # Direct keyword mapping
        fallback_traits = {
            'злий': 'angry',
            'злой': 'angry',
            'строгий': 'strict',
            'позитивн': 'positive',
            'негативн': 'negative',
            'лається': 'swears',
            'матюкається': 'swears',
            'ругается': 'swears',
            'добрий': 'kind',
            'добрый': 'kind',
            'агресивн': 'aggressive',
            'токсичн': 'toxic',
        }

        for keyword, trait in fallback_traits.items():
            if keyword in question_lower:
                return trait

        return None

    def _format_top_speakers(self, stats: list[dict]) -> str:
        """Format top speakers for display."""
        if not stats:
            return "Не знайдено повідомлень для аналізу."

        lines = ["🏆 Найактивніші учасники:", ""]
        medals = ["🥇", "🥈", "🥉"]

        for i, stat in enumerate(stats[:10]):
            medal = medals[i] if i < 3 else f"{i + 1}."
            lines.append(
                f"{medal} 👤 {stat['display_name']} — {stat['message_count']:,} повідомлень"
            )

        return "\n".join(lines)

    def _format_mention_stats(self, term: str, stats: list[dict]) -> str:
        """Format mention counts for display."""
        if not stats:
            return f"Ніхто не згадував «{term}»."

        lines = [f"📊 Хто згадував «{term}»:", ""]

        for i, stat in enumerate(stats[:10]):
            rank = i + 1
            count = stat['mention_count']
            suffix = self._pluralize_times(count)
            lines.append(
                f"{rank}. 👤 {stat['display_name']} — {count} {suffix}"
            )

        return "\n".join(lines)

    def _format_behavioral_stats(self, trait: str, stats: list[dict]) -> str:
        """Format behavioral analysis for display."""
        if not stats:
            return f"Не вдалося проаналізувати характеристику «{trait}»."

        # Translate trait name
        trait_names = {
            'angry': 'злий/агресивний',
            'strict': 'строгий',
            'positive': 'позитивний',
            'negative': 'негативний',
            'psycho': 'божевільний',
            'swears': 'лається',
            'kind': 'добрий',
            'aggressive': 'агресивний',
            'toxic': 'токсичний',
        }
        trait_display = trait_names.get(trait, trait)

        lines = [f"🔍 Хто найбільш «{trait_display}»:", ""]

        for i, stat in enumerate(stats[:5]):
            rank = i + 1
            score = stat['score']
            count = stat['example_count']
            lines.append(
                f"{rank}. 👤 {stat['display_name']} — оцінка {score:.1f}/10 "
                f"(знайдено {count} повідомлень)"
            )

        return "\n".join(lines)

    def _pluralize_times(self, count: int) -> str:
        """Pluralize 'раз/рази/разів' in Ukrainian."""
        if count % 10 == 1 and count % 100 != 11:
            return "раз"
        elif 2 <= count % 10 <= 4 and (count % 100 < 10 or count % 100 >= 20):
            return "рази"
        else:
            return "разів"
