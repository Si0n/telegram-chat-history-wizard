"""
Detect position changes / contradictions in user's messages.
"""
from dataclasses import dataclass
from typing import Optional

from .embeddings import ChatService
from .vector_store import VectorStore


@dataclass
class FlipResult:
    """Result of flip detection analysis."""
    user: str
    topic: str
    has_flip: bool
    confidence: str  # "high", "medium", "low"
    summary: str
    messages: list[dict]  # Relevant messages with dates
    analysis: Optional[str] = None


class FlipDetector:
    """Detect when users changed their position on a topic."""

    SYSTEM_PROMPT = """You are an analyst examining chat messages to detect if a person changed their position or opinion on a topic over time.

Your task:
1. Analyze the messages chronologically
2. Identify the person's stance/opinion in each message
3. Determine if their position changed, and if so, how
4. Be objective - only report actual contradictions, not minor clarifications

Respond in this format:
FLIP_DETECTED: [YES/NO/UNCLEAR]
CONFIDENCE: [HIGH/MEDIUM/LOW]
SUMMARY: [1-2 sentence summary of what changed, or "No significant position change detected"]

If FLIP_DETECTED is YES, also include:
BEFORE: [Their original position with date]
AFTER: [Their new position with date]
"""

    def __init__(
        self,
        vector_store: VectorStore,
        chat_service: ChatService = None
    ):
        self.vector_store = vector_store
        self.chat_service = chat_service or ChatService()

    def _format_messages_for_analysis(self, messages: list[dict]) -> str:
        """Format messages for LLM analysis."""
        lines = []
        for msg in sorted(messages, key=lambda m: m["metadata"].get("timestamp", "")):
            date = msg["metadata"].get("formatted_date", "Unknown date")
            text = msg["text"][:500]  # Truncate long messages
            lines.append(f"[{date}]: {text}")
        return "\n\n".join(lines)

    async def detect_flip_async(
        self,
        user: str,
        topic: str,
        n_messages: int = 15
    ) -> FlipResult:
        """
        Analyze user's messages about topic for position changes.
        """
        # Get relevant messages
        messages = self.vector_store.get_user_messages_about(
            user_identifier=user,
            topic=topic,
            n_results=n_messages
        )

        if not messages:
            return FlipResult(
                user=user,
                topic=topic,
                has_flip=False,
                confidence="low",
                summary=f"No messages found from {user} about '{topic}'",
                messages=[]
            )

        if len(messages) < 2:
            return FlipResult(
                user=user,
                topic=topic,
                has_flip=False,
                confidence="low",
                summary=f"Only 1 message found - need more data to detect changes",
                messages=messages
            )

        # Format for analysis
        formatted = self._format_messages_for_analysis(messages)
        prompt = f"""Analyze these messages from {user} about "{topic}":

{formatted}

Did this person change their position or opinion on this topic over time?"""

        # Get LLM analysis
        analysis = await self.chat_service.complete_async(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            max_tokens=500
        )

        # Parse response
        has_flip = "FLIP_DETECTED: YES" in analysis.upper()
        confidence = "medium"
        if "CONFIDENCE: HIGH" in analysis.upper():
            confidence = "high"
        elif "CONFIDENCE: LOW" in analysis.upper():
            confidence = "low"

        # Extract summary
        summary = ""
        for line in analysis.split("\n"):
            if line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
                break

        if not summary:
            summary = "Analysis complete - see details below"

        return FlipResult(
            user=user,
            topic=topic,
            has_flip=has_flip,
            confidence=confidence,
            summary=summary,
            messages=messages,
            analysis=analysis
        )

    def detect_flip(self, user: str, topic: str, n_messages: int = 15) -> FlipResult:
        """Sync wrapper for flip detection."""
        import asyncio
        return asyncio.run(self.detect_flip_async(user, topic, n_messages))
