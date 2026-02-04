"""
AI-powered answer synthesis from search results.
Uses ChatGPT to analyze found messages and formulate direct answers.
"""
from dataclasses import dataclass
from typing import Optional

from search.embeddings import ChatService


@dataclass
class SynthesizedAnswer:
    """Structured answer from AI analysis."""
    answer: str                    # Direct answer to the question
    confidence: str                # high/medium/low
    supporting_quotes: list[dict]  # Most relevant quotes
    summary: str                   # Brief summary


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
        mentioned_users: list[tuple[int, str]] = None
    ) -> SynthesizedAnswer:
        """Generate an AI answer based on search results."""
        if not results:
            return SynthesizedAnswer(
                answer="❌ Не знайдено релевантних повідомлень для відповіді на це питання.",
                confidence="low",
                supporting_quotes=[],
                summary="Пошук не дав результатів."
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

        # Extract top 3 most relevant quotes for display
        supporting_quotes = results[:3]

        return SynthesizedAnswer(
            answer=response,
            confidence="high" if len(results) >= 3 else "medium" if results else "low",
            supporting_quotes=supporting_quotes,
            summary=response[:200] + "..." if len(response) > 200 else response
        )

    def synthesize(
        self,
        question: str,
        results: list[dict],
        mentioned_users: list[tuple[int, str]] = None
    ) -> SynthesizedAnswer:
        """Sync version."""
        import asyncio
        return asyncio.run(self.synthesize_async(question, results, mentioned_users))
