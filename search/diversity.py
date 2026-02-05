"""
Result diversity algorithms for search results.
Includes user-based diversity and Maximal Marginal Relevance (MMR).
"""
import logging
from collections import defaultdict
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


def apply_user_diversity(
    results: list[dict],
    max_per_user: int = 2
) -> list[dict]:
    """
    Limit results per user to ensure diverse perspectives.

    Args:
        results: List of search results with metadata containing user_id
        max_per_user: Maximum results to keep per user

    Returns:
        Filtered list with max_per_user results per unique user
    """
    if not results or max_per_user <= 0:
        return results

    user_counts = defaultdict(int)
    filtered = []

    for result in results:
        user_id = result.get("metadata", {}).get("user_id")
        if user_id is None:
            # Keep results without user_id
            filtered.append(result)
            continue

        if user_counts[user_id] < max_per_user:
            filtered.append(result)
            user_counts[user_id] += 1

    logger.debug(
        f"User diversity: {len(results)} -> {len(filtered)} results "
        f"(max {max_per_user} per user)"
    )
    return filtered


def apply_mmr(
    results: list[dict],
    embeddings: Optional[list[list[float]]] = None,
    lambda_param: float = 0.7,
    top_k: int = 10
) -> list[dict]:
    """
    Maximal Marginal Relevance: balance relevance + diversity.

    MMR score = lambda * relevance - (1 - lambda) * max_similarity_to_selected

    Args:
        results: List of search results with 'similarity' or 'relevance_score'
        embeddings: Optional pre-computed embeddings for results (same order)
        lambda_param: Balance between relevance (1.0) and diversity (0.0)
        top_k: Number of results to return

    Returns:
        Reordered list of top_k results balancing relevance and diversity
    """
    if not results or len(results) <= 1:
        return results[:top_k]

    n = len(results)
    top_k = min(top_k, n)

    # Get relevance scores (normalize to 0-1)
    relevance_scores = []
    for r in results:
        score = r.get("relevance_score")
        if score is None:
            score = r.get("similarity", 0.5) * 10
        relevance_scores.append(float(score) / 10.0)  # Normalize to 0-1

    # If no embeddings provided, use text similarity as proxy
    if embeddings is None:
        # Fallback: use similarity scores directly as diversity proxy
        # Select greedily based on relevance only
        selected_indices = []
        remaining = set(range(n))

        while len(selected_indices) < top_k and remaining:
            best_idx = max(remaining, key=lambda i: relevance_scores[i])
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return [results[i] for i in selected_indices]

    # Convert embeddings to numpy for efficient computation
    embeddings_np = np.array(embeddings)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    embeddings_normalized = embeddings_np / norms

    selected_indices = []
    remaining = set(range(n))

    while len(selected_indices) < top_k and remaining:
        best_idx = None
        best_mmr = float('-inf')

        for idx in remaining:
            # Relevance component
            relevance = relevance_scores[idx]

            # Diversity component: max similarity to already selected
            if selected_indices:
                selected_embeddings = embeddings_normalized[selected_indices]
                current_embedding = embeddings_normalized[idx]
                similarities = np.dot(selected_embeddings, current_embedding)
                max_sim = float(np.max(similarities))
            else:
                max_sim = 0.0

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    logger.debug(
        f"MMR: selected {len(selected_indices)} from {n} results "
        f"(lambda={lambda_param})"
    )

    return [results[i] for i in selected_indices]


def apply_diversity_pipeline(
    results: list[dict],
    embeddings: Optional[list[list[float]]] = None,
    max_per_user: int = 2,
    lambda_param: float = 0.7,
    top_k: int = 10,
    use_user_diversity: bool = True,
    use_mmr: bool = True
) -> list[dict]:
    """
    Full diversity pipeline: user diversity + MMR.

    Args:
        results: Search results to diversify
        embeddings: Optional embeddings for MMR
        max_per_user: Max results per user (0 to disable)
        lambda_param: MMR lambda parameter
        top_k: Final number of results
        use_user_diversity: Whether to apply user diversity
        use_mmr: Whether to apply MMR

    Returns:
        Diversified list of results
    """
    if not results:
        return results

    current = results

    # Step 1: User diversity (limit per user)
    if use_user_diversity and max_per_user > 0:
        current = apply_user_diversity(current, max_per_user)

    # Step 2: MMR for content diversity
    if use_mmr:
        # Get embeddings subset if we filtered
        filtered_embeddings = None
        if embeddings and len(embeddings) == len(results):
            # Map remaining results back to embeddings
            result_indices = {id(r): i for i, r in enumerate(results)}
            filtered_embeddings = [
                embeddings[result_indices[id(r)]]
                for r in current
                if id(r) in result_indices
            ]

        current = apply_mmr(
            current,
            embeddings=filtered_embeddings,
            lambda_param=lambda_param,
            top_k=top_k
        )

    return current[:top_k]
