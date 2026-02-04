"""
AI-powered answer synthesis from search results.
Uses ChatGPT to analyze found messages and formulate direct answers.
Includes relevance filtering to skip irrelevant results.
"""
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from search.embeddings import ChatService

logger = logging.getLogger(__name__)


@dataclass
class SynthesizedAnswer:
    """Structured answer from AI analysis."""
    answer: str                    # Direct answer to the question
    confidence: str                # high/medium/low
    supporting_quotes: list[dict]  # Most relevant quotes
    summary: str                   # Brief summary


RELEVANCE_SYSTEM_PROMPT = """You are a relevance judge. Score how relevant each message is to the user's question.

For each message, respond with a relevance score from 0-10:
- 0-2: Completely irrelevant, wrong topic
- 3-4: Tangentially related, same keywords but different context
- 5-6: Somewhat relevant, discusses related topic
- 7-8: Relevant, addresses the question's topic
- 9-10: Highly relevant, directly answers or discusses what was asked

Respond ONLY with JSON array of scores in the same order as messages:
[score1, score2, score3, ...]"""


ANSWER_SYSTEM_PROMPT = """Ти помічник для аналізу історії чату. Твоя задача - відповісти на питання користувача на основі знайдених повідомлень.

ВАЖЛИВО:
1. Відповідай ТІЛЬКИ на основі наданих повідомлень
2. Якщо в повідомленнях немає відповіді - чесно скажи про це
3. Цитуй конкретні повідомлення як докази
4. Відповідай тією ж мовою, що й питання (українська/російська)

Формат відповіді:
1. Спочатку дай ПРЯМУ відповідь на питання (Так/Ні/Частково/Немає даних)
2. Потім коротко поясни на основі чого зроблено висновок
3. Вкажи найбільш релевантні цитати

Будь стислим але інформативним. Не вигадуй інформацію, якої немає в повідомленнях."""


class AnswerSynthesizer:
    """Synthesize answers to questions based on search results."""

    def __init__(self):
        self.chat_service = ChatService()

    async def filter_by_relevance(
        self,
        question: str,
        results: list[dict],
        min_score: int = 5
    ) -> list[dict]:
        """Filter results by AI-judged relevance to the question."""
        if not results:
            return []

        # Format messages for relevance check
        messages_text = []
        for i, result in enumerate(results[:20], 1):  # Check max 20
            meta = result.get("metadata", {})
            username = meta.get("display_name", "Unknown")
            text = result.get("text", "")[:300]
            messages_text.append(f"[{i}] {username}: \"{text}\"")

        prompt = f"""Question: {question}

Messages to evaluate:
{chr(10).join(messages_text)}

Rate each message's relevance (0-10) to the question. Return JSON array of scores."""

        try:
            response = await self.chat_service.complete_async(
                prompt=prompt,
                system=RELEVANCE_SYSTEM_PROMPT,
                max_tokens=200
            )

            # Parse scores
            json_match = re.search(r'\[[\d,\s]+\]', response)
            if json_match:
                scores = json.loads(json_match.group())

                # Filter by minimum score
                filtered = []
                for i, result in enumerate(results[:len(scores)]):
                    if i < len(scores) and scores[i] >= min_score:
                        result["relevance_score"] = scores[i]
                        filtered.append(result)

                # Sort by relevance score
                filtered.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

                logger.info(f"Relevance filter: {len(results)} -> {len(filtered)} results (min_score={min_score})")
                return filtered

        except Exception as e:
            logger.warning(f"Relevance filtering failed: {e}")

        # Fallback: return original results
        return results

    def _format_messages_for_context(self, results: list[dict], max_messages: int = 10) -> str:
        """Format search results as context for the AI."""
        if not results:
            return "Повідомлень не знайдено."

        lines = []
        for i, result in enumerate(results[:max_messages], 1):
            meta = result.get("metadata", {})
            username = meta.get("display_name", "Unknown")
            date = meta.get("formatted_date", "Unknown")
            text = result.get("text", "")[:500]  # Truncate long messages

            lines.append(f"[{i}] {username} ({date}):")
            lines.append(f'"{text}"')
            lines.append("")

        return "\n".join(lines)

    async def synthesize_async(
        self,
        question: str,
        results: list[dict],
        mentioned_users: list[tuple[int, str]] = None,
        filter_relevance: bool = True
    ) -> SynthesizedAnswer:
        """Generate an AI answer based on search results."""
        if not results:
            return SynthesizedAnswer(
                answer="❌ Не знайдено релевантних повідомлень для відповіді на це питання.",
                confidence="low",
                supporting_quotes=[],
                summary="Пошук не дав результатів."
            )

        # Filter by relevance first
        if filter_relevance and len(results) > 3:
            results = await self.filter_by_relevance(question, results, min_score=5)

            if not results:
                return SynthesizedAnswer(
                    answer="❌ Знайдені повідомлення не відповідають на питання. Спробуйте переформулювати запит.",
                    confidence="low",
                    supporting_quotes=[],
                    summary="Результати не пройшли фільтр релевантності."
                )

        # Format context
        messages_context = self._format_messages_for_context(results)

        user_context = ""
        if mentioned_users:
            names = [u[1] for u in mentioned_users]
            user_context = f"\nПитання стосується користувача(ів): {', '.join(names)}\n"

        prompt = f"""Питання: {question}
{user_context}
Знайдені повідомлення з історії чату:

{messages_context}

На основі цих повідомлень, дай відповідь на питання. Почни з прямої відповіді (Так/Ні/Частково), потім поясни."""

        response = await self.chat_service.complete_async(
            prompt=prompt,
            system=ANSWER_SYSTEM_PROMPT,
            max_tokens=800
        )

        # Extract top 3 most relevant quotes for display (already sorted by relevance if filtered)
        supporting_quotes = results[:3]

        # Determine confidence based on relevance scores if available
        if supporting_quotes and "relevance_score" in supporting_quotes[0]:
            avg_score = sum(q.get("relevance_score", 5) for q in supporting_quotes) / len(supporting_quotes)
            confidence = "high" if avg_score >= 7 else "medium" if avg_score >= 5 else "low"
        else:
            confidence = "high" if len(results) >= 3 else "medium" if results else "low"

        return SynthesizedAnswer(
            answer=response,
            confidence=confidence,
            supporting_quotes=supporting_quotes,
            summary=response[:200] + "..." if len(response) > 200 else response
        )

    def synthesize(
        self,
        question: str,
        results: list[dict],
        mentioned_users: list[tuple[int, str]] = None,
        filter_relevance: bool = True
    ) -> SynthesizedAnswer:
        """Sync version."""
        import asyncio
        return asyncio.run(self.synthesize_async(question, results, mentioned_users, filter_relevance))
