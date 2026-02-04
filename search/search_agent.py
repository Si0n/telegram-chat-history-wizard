"""
Multi-step search agent with query reformulation and result caching.
Iteratively searches until finding relevant results or exhausting options.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

from search.embeddings import ChatService
from search.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Enhanced search result with relevance tracking."""
    vector_id: str
    message_id: int
    text: str
    metadata: dict
    similarity: float  # Vector similarity
    relevance_score: Optional[int] = None  # AI-judged relevance (0-10)
    relevance_query: Optional[str] = None  # Query it was judged against


@dataclass
class SearchSession:
    """Track a multi-step search session."""
    original_question: str
    queries_tried: list[str] = field(default_factory=list)
    all_candidates: dict[str, SearchResult] = field(default_factory=dict)  # vector_id -> result
    relevant_results: list[SearchResult] = field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 3


# Cache for relevance scores (message_id -> {query_hash: score})
_relevance_cache: dict[int, dict[str, int]] = {}
_cache_expiry: datetime = datetime.now()
CACHE_TTL_HOURS = 24


def _get_query_hash(query: str) -> str:
    """Simple hash for query caching."""
    # Normalize and hash
    normalized = " ".join(sorted(query.lower().split()))
    return str(hash(normalized) % 10000000)


def _get_cached_relevance(message_id: int, query: str) -> Optional[int]:
    """Get cached relevance score if available."""
    global _cache_expiry
    if datetime.now() > _cache_expiry:
        _relevance_cache.clear()
        _cache_expiry = datetime.now() + timedelta(hours=CACHE_TTL_HOURS)
        return None

    query_hash = _get_query_hash(query)
    return _relevance_cache.get(message_id, {}).get(query_hash)


def _cache_relevance(message_id: int, query: str, score: int):
    """Cache relevance score."""
    query_hash = _get_query_hash(query)
    if message_id not in _relevance_cache:
        _relevance_cache[message_id] = {}
    _relevance_cache[message_id][query_hash] = score


REFORMULATE_PROMPT = """You are helping improve a search query that didn't find good results.

Original question: {question}
Previous search query: {previous_query}
Problem: Found {found_count} results but only {relevant_count} were relevant.

Generate a NEW search query that might find better results. Think about:
1. Different synonyms or phrasings
2. Related concepts that might be discussed
3. How people actually talk about this topic in casual chat

Respond with ONLY the new search query (8-15 words), no explanation."""


RELEVANCE_PROMPT = """Score how relevant each message is to answering this question.

Question: {question}

Messages:
{messages}

For each message, give a score 0-10:
- 0-2: Completely irrelevant
- 3-4: Same keywords but wrong context
- 5-6: Related topic but doesn't answer
- 7-8: Relevant, discusses the topic
- 9-10: Directly answers or highly relevant

Respond with JSON array of scores: [score1, score2, ...]"""


class SearchAgent:
    """Multi-step search agent with query reformulation."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.chat_service = ChatService()

    async def search(
        self,
        question: str,
        initial_query: str,
        user_id: int = None,
        date_from: str = None,
        date_to: str = None,
        min_relevant: int = 3,
        min_relevance_score: int = 5,
        max_iterations: int = 3
    ) -> tuple[list[dict], SearchSession]:
        """
        Multi-step search with automatic query reformulation.

        Returns:
            (relevant_results, session) - results and search session for debugging
        """
        session = SearchSession(
            original_question=question,
            max_iterations=max_iterations
        )

        current_query = initial_query

        while session.iterations < max_iterations:
            session.iterations += 1
            session.queries_tried.append(current_query)

            logger.info(f"Search iteration {session.iterations}: '{current_query}'")

            # Search
            raw_results = self.vector_store.search(
                query=current_query,
                n_results=20,  # Get more candidates
                user_id=user_id,
                date_from=date_from,
                date_to=date_to
            )

            if not raw_results:
                logger.info("No results found, trying to reformulate...")
                current_query = await self._reformulate_query(
                    question, current_query, 0, 0
                )
                if not current_query or current_query in session.queries_tried:
                    break
                continue

            # Convert to SearchResult and check cache
            new_candidates = []
            for r in raw_results:
                vec_id = r.get("vector_id", "")
                msg_id = r.get("metadata", {}).get("message_id", 0)

                # Skip if already evaluated for this question
                if vec_id in session.all_candidates:
                    continue

                # Check cache for relevance
                cached_score = _get_cached_relevance(msg_id, question)

                result = SearchResult(
                    vector_id=vec_id,
                    message_id=msg_id,
                    text=r.get("text", ""),
                    metadata=r.get("metadata", {}),
                    similarity=r.get("similarity", 0),
                    relevance_score=cached_score,
                    relevance_query=question if cached_score else None
                )

                session.all_candidates[vec_id] = result

                if cached_score is None:
                    new_candidates.append(result)
                elif cached_score >= min_relevance_score:
                    session.relevant_results.append(result)

            # Score new candidates
            if new_candidates:
                await self._score_relevance(question, new_candidates)

                # Update cache and collect relevant
                for result in new_candidates:
                    if result.relevance_score is not None:
                        _cache_relevance(result.message_id, question, result.relevance_score)
                        if result.relevance_score >= min_relevance_score:
                            session.relevant_results.append(result)

            # Check if we have enough relevant results
            if len(session.relevant_results) >= min_relevant:
                logger.info(f"Found {len(session.relevant_results)} relevant results")
                break

            # Not enough - reformulate and try again
            logger.info(f"Only {len(session.relevant_results)} relevant, reformulating...")

            current_query = await self._reformulate_query(
                question,
                current_query,
                len(raw_results),
                len(session.relevant_results)
            )

            if not current_query or current_query in session.queries_tried:
                logger.info("No new query to try, stopping")
                break

        # Sort by relevance score
        session.relevant_results.sort(
            key=lambda x: (x.relevance_score or 0, x.similarity),
            reverse=True
        )

        # Convert back to dict format for compatibility
        results = []
        for r in session.relevant_results:
            results.append({
                "vector_id": r.vector_id,
                "text": r.text,
                "metadata": r.metadata,
                "similarity": r.similarity,
                "relevance_score": r.relevance_score
            })

        return results, session

    async def _score_relevance(
        self,
        question: str,
        candidates: list[SearchResult]
    ):
        """Score relevance of candidates using AI."""
        if not candidates:
            return

        # Format messages
        messages_text = []
        for i, r in enumerate(candidates[:15], 1):  # Max 15 at a time
            username = r.metadata.get("display_name", "Unknown")
            text = r.text[:300] if r.text else ""
            messages_text.append(f"[{i}] {username}: \"{text}\"")

        prompt = RELEVANCE_PROMPT.format(
            question=question,
            messages="\n".join(messages_text)
        )

        try:
            response = await self.chat_service.complete_async(
                prompt=prompt,
                system="You are a relevance judge. Respond only with JSON array of integer scores.",
                max_tokens=150
            )

            # Parse scores
            json_match = re.search(r'\[[\d,\s]+\]', response)
            if json_match:
                scores = json.loads(json_match.group())

                for i, r in enumerate(candidates[:len(scores)]):
                    if i < len(scores):
                        r.relevance_score = scores[i]
                        r.relevance_query = question

        except Exception as e:
            logger.warning(f"Relevance scoring failed: {e}")
            # Fallback: use similarity as proxy
            for r in candidates:
                r.relevance_score = int(r.similarity * 10)

    async def _reformulate_query(
        self,
        question: str,
        previous_query: str,
        found_count: int,
        relevant_count: int
    ) -> Optional[str]:
        """Generate a new search query."""
        prompt = REFORMULATE_PROMPT.format(
            question=question,
            previous_query=previous_query,
            found_count=found_count,
            relevant_count=relevant_count
        )

        try:
            response = await self.chat_service.complete_async(
                prompt=prompt,
                system="Generate only the search query, nothing else.",
                max_tokens=100
            )

            # Clean up response
            new_query = response.strip().strip('"\'')

            if new_query and len(new_query) > 5 and new_query != previous_query:
                logger.info(f"Reformulated query: '{previous_query}' -> '{new_query}'")
                return new_query

        except Exception as e:
            logger.warning(f"Query reformulation failed: {e}")

        return None
