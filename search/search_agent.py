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

import config
from search.embeddings import ChatService
from search.vector_store import VectorStore
from search.diversity import apply_diversity_pipeline

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


def _get_query_hash(query: str) -> str:
    """Simple hash for query caching."""
    # Normalize and hash
    normalized = " ".join(sorted(query.lower().split()))
    return str(hash(normalized) % 10000000)


HYDE_PROMPT = """You are simulating chat messages that would answer a question about chat history.

Question: {question}

Generate 2-3 SHORT example chat messages (in Ukrainian/Russian) that would directly answer or discuss this question.
These should sound like real casual chat messages, not formal text.

IMPORTANT - Use common slang and real names together:
- "зе/зеля" = Зеленський
- "порох/петя" = Порошенко
- "пуйло/путлер" = Путін
- "біток/btc" = біткоін
- "крипта" = криптовалюта

Examples of good output:
Question: "чи гусь казав що біткоін скам?"
Output: "біткоін це повний скам, не вкладайте гроші"
"я вже давно кажу що крипта - це піраміда"
"btc може впасти до нуля"

Question: "що думали про зе?"
Output: "Зеленський робить все правильно"
"зєля знову щось обіцяє"
"клоун з 95 кварталу"

Now generate messages for the question above. Output ONLY the messages, one per line:"""


RERANK_PROMPT = """You are ranking search results by relevance to a question.

Question: {question}

Search Results:
{results}

Rank these results from MOST to LEAST relevant. Consider:
1. Does it directly answer the question?
2. Is it from the right person (if specified)?
3. Is the topic/context correct?

Respond with JSON array of result numbers in order of relevance (most relevant first):
Example: [3, 1, 5, 2, 4]

Your ranking:"""


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
    """Multi-step search agent with query reformulation, HyDE, and reranking."""

    def __init__(self, vector_store: VectorStore, db=None):
        self.vector_store = vector_store
        self.chat_service = ChatService()
        self.db = db  # Database for persistent cache
        self._cache_ttl = getattr(config, 'RELEVANCE_CACHE_TTL_HOURS', 24)

    def _get_cached_relevance(self, message_id: int, query: str) -> Optional[int]:
        """Get cached relevance score from database if available."""
        if self.db is None:
            return None
        query_hash = _get_query_hash(query)
        return self.db.get_cached_relevance(message_id, query_hash, self._cache_ttl)

    def _get_cached_relevance_batch(self, message_ids: list[int], query: str) -> dict[int, int]:
        """Get cached relevance scores for multiple messages."""
        if self.db is None:
            return {}
        query_hash = _get_query_hash(query)
        return self.db.get_cached_relevance_batch(message_ids, query_hash, self._cache_ttl)

    def _cache_relevance(self, message_id: int, query: str, score: int):
        """Cache relevance score to database."""
        if self.db is None:
            return
        query_hash = _get_query_hash(query)
        self.db.cache_relevance(message_id, query_hash, score)

    def _cache_relevance_batch(self, entries: list[tuple[int, str, int]]):
        """Cache multiple relevance scores to database."""
        if self.db is None:
            return
        self.db.cache_relevance_batch(entries)

    async def generate_hyde_documents(self, question: str) -> list[str]:
        """
        Generate Hypothetical Document Embeddings (HyDE).
        Creates synthetic chat messages that would answer the question.
        """
        prompt = HYDE_PROMPT.format(question=question)

        try:
            response = await self.chat_service.complete_async(
                prompt=prompt,
                system="Generate realistic chat messages in Ukrainian or Russian. Be concise.",
                max_tokens=200
            )

            # Split into individual messages
            lines = [line.strip().strip('"\'') for line in response.strip().split('\n')]
            hyde_docs = [line for line in lines if line and len(line) > 10]

            logger.info(f"Generated {len(hyde_docs)} HyDE documents for: {question[:50]}...")
            return hyde_docs[:3]  # Max 3

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return []

    async def rerank_results(
        self,
        question: str,
        results: list[SearchResult],
        top_k: int = 10
    ) -> list[SearchResult]:
        """
        Rerank results using LLM for better relevance ordering.
        """
        if not results or len(results) <= 1:
            return results

        # Take top candidates for reranking (limit API cost)
        candidates = results[:min(15, len(results))]

        # Format results for reranking
        results_text = []
        for i, r in enumerate(candidates, 1):
            username = r.metadata.get("display_name", "Unknown")
            date = r.metadata.get("formatted_date", "")
            text = r.text[:200] if r.text else ""
            results_text.append(f"[{i}] {username} ({date}): \"{text}\"")

        prompt = RERANK_PROMPT.format(
            question=question,
            results="\n".join(results_text)
        )

        try:
            response = await self.chat_service.complete_async(
                prompt=prompt,
                system="You are a search result ranker. Respond only with JSON array of numbers.",
                max_tokens=100
            )

            # Parse ranking
            json_match = re.search(r'\[[\d,\s]+\]', response)
            if json_match:
                ranking = json.loads(json_match.group())

                # Reorder results based on ranking
                reranked = []
                seen = set()
                for idx in ranking:
                    if 1 <= idx <= len(candidates) and idx not in seen:
                        reranked.append(candidates[idx - 1])
                        seen.add(idx)

                # Add any remaining that weren't in ranking
                for i, r in enumerate(candidates, 1):
                    if i not in seen:
                        reranked.append(r)

                logger.info(f"Reranked {len(candidates)} results, new order: {ranking[:5]}...")
                return reranked[:top_k]

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")

        return results[:top_k]

    async def search(
        self,
        question: str,
        initial_query: str,
        user_id: int = None,
        date_from: str = None,
        date_to: str = None,
        min_relevant: int = 3,
        min_relevance_score: int = 5,
        max_iterations: int = 3,
        use_hyde: bool = True,
        use_reranking: bool = True,
        use_diversity: bool = True,
        diversity_config: dict = None
    ) -> tuple[list[dict], SearchSession]:
        """
        Multi-step search with automatic query reformulation, HyDE, and reranking.

        Args:
            question: Original user question
            initial_query: Parsed search query
            user_id: Filter by user
            date_from/date_to: Date filters
            min_relevant: Minimum relevant results needed
            min_relevance_score: Minimum score to consider relevant (0-10)
            max_iterations: Max reformulation attempts
            use_hyde: Use Hypothetical Document Embeddings
            use_reranking: Use LLM reranking for final results
            use_diversity: Apply diversity filtering (user-based + MMR)
            diversity_config: Optional dict with max_per_user, lambda_param

        Returns:
            (relevant_results, session) - results and search session for debugging
        """
        # Default diversity config
        if diversity_config is None:
            diversity_config = {
                "max_per_user": getattr(config, 'MAX_RESULTS_PER_USER', 2),
                "lambda_param": getattr(config, 'DIVERSITY_LAMBDA', 0.7),
            }
        session = SearchSession(
            original_question=question,
            max_iterations=max_iterations
        )

        # === HyDE: Generate hypothetical documents ===
        hyde_queries = []
        if use_hyde:
            hyde_docs = await self.generate_hyde_documents(question)
            if hyde_docs:
                hyde_queries = hyde_docs
                logger.info(f"Using {len(hyde_queries)} HyDE queries in addition to original")

        # Combine initial query with HyDE docs for search
        all_queries = [initial_query] + hyde_queries
        current_query = initial_query

        while session.iterations < max_iterations:
            session.iterations += 1
            session.queries_tried.append(current_query)

            logger.info(f"Search iteration {session.iterations}: '{current_query}' (date_from={date_from}, date_to={date_to})")

            # On first iteration, also search with HyDE queries
            queries_to_search = [current_query]
            if session.iterations == 1 and hyde_queries:
                queries_to_search.extend(hyde_queries)
                logger.info(f"Searching with {len(queries_to_search)} queries (1 original + {len(hyde_queries)} HyDE)")

            # Search with all queries and merge results
            raw_results = []
            seen_ids = set()
            for query in queries_to_search:
                results = self.vector_store.search(
                    query=query,
                    n_results=15,  # Get candidates per query
                    user_id=user_id,
                    date_from=date_from,
                    date_to=date_to
                )
                for r in results:
                    vec_id = r.get("vector_id", "")
                    if vec_id not in seen_ids:
                        seen_ids.add(vec_id)
                        raw_results.append(r)

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

            # Batch fetch cached relevance scores
            msg_ids_to_check = [
                r.get("metadata", {}).get("message_id", 0)
                for r in raw_results
                if r.get("vector_id", "") not in session.all_candidates
            ]
            cached_scores = self._get_cached_relevance_batch(msg_ids_to_check, question)

            for r in raw_results:
                vec_id = r.get("vector_id", "")
                msg_id = r.get("metadata", {}).get("message_id", 0)

                # Skip if already evaluated for this question
                if vec_id in session.all_candidates:
                    continue

                # Check cache for relevance
                cached_score = cached_scores.get(msg_id)

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

                # Batch cache relevance scores
                cache_entries = []
                query_hash = _get_query_hash(question)
                for result in new_candidates:
                    if result.relevance_score is not None:
                        cache_entries.append((result.message_id, query_hash, result.relevance_score))
                        if result.relevance_score >= min_relevance_score:
                            session.relevant_results.append(result)

                if cache_entries:
                    self._cache_relevance_batch(cache_entries)

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

        # === Reranking: Use LLM to reorder top results ===
        if use_reranking and len(session.relevant_results) > 3:
            logger.info(f"Reranking {len(session.relevant_results)} results...")
            session.relevant_results = await self.rerank_results(
                question=question,
                results=session.relevant_results,
                top_k=min(15, len(session.relevant_results))
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

        # === Diversity: Apply user diversity and MMR ===
        if use_diversity and len(results) > 1:
            logger.info(f"Applying diversity to {len(results)} results...")
            results = apply_diversity_pipeline(
                results=results,
                embeddings=None,  # Use relevance-based selection
                max_per_user=diversity_config.get("max_per_user", 2),
                lambda_param=diversity_config.get("lambda_param", 0.7),
                top_k=min(15, len(results)),
                use_user_diversity=True,
                use_mmr=True
            )
            logger.info(f"After diversity: {len(results)} results")

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
