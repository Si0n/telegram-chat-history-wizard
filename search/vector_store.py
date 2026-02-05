"""
ChromaDB vector store for semantic search.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings

import config
from .embeddings import EmbeddingService


class VectorStore:
    """ChromaDB wrapper for message embeddings."""

    COLLECTION_NAME = "messages"

    def __init__(
        self,
        persist_path: Path = None,
        embedding_service: EmbeddingService = None
    ):
        self.persist_path = persist_path or config.CHROMA_DB_PATH
        self.persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                # Memory optimizations for large collections
                "hnsw:M": 8,  # Reduced from default 16 - less memory
                "hnsw:construction_ef": 64,  # Reduced from default 100
                "hnsw:search_ef": 32,  # Reduced for faster searches
            }
        )

        self.embedding_service = embedding_service or EmbeddingService()

    def add_messages(
        self,
        db_ids: list[int],
        texts: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]] = None
    ) -> list[str]:
        """
        Add messages to vector store.
        Returns list of vector IDs.
        """
        if not texts:
            return []

        # Generate embeddings if not provided
        if embeddings is None:
            embeddings = self.embedding_service.embed_batch(texts)

        # Create unique IDs (include chunk number for chunked messages)
        vector_ids = []
        for i, db_id in enumerate(db_ids):
            chunk = metadatas[i].get("chunk") if metadatas else None
            if chunk:
                vector_ids.append(f"msg_{db_id}_c{chunk}")
            else:
                vector_ids.append(f"msg_{db_id}")

        # Add to collection
        self.collection.add(
            ids=vector_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        return vector_ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        user_id: int = None,
        username: str = None,
        date_from: str = None,
        date_to: str = None,
        min_similarity: float = 0.0
    ) -> list[dict]:
        """
        Semantic search for messages.
        Returns list of matches with metadata.
        """
        # Fetch more results than needed for better coverage, then filter
        fetch_count = max(n_results * 3, 30)

        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Build where filter
        where_filter = None
        conditions = []

        if user_id:
            conditions.append({"user_id": user_id})
        if username:
            conditions.append({"username": username.lstrip("@")})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        # Search (fetch more for better coverage)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_count,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        matches = []
        if results and results["ids"] and results["ids"][0]:
            for i, vec_id in enumerate(results["ids"][0]):
                similarity = 1 - results["distances"][0][i]
                # Filter by minimum similarity
                if similarity < min_similarity:
                    continue
                matches.append({
                    "vector_id": vec_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": similarity
                })

        # Apply date filtering (post-search since ChromaDB doesn't handle date ranges well)
        if date_from or date_to:
            matches = self._filter_by_date(matches, date_from, date_to)

        # Return top n_results after all filtering
        return matches[:n_results]

    def _filter_by_date(
        self,
        matches: list[dict],
        date_from: str = None,
        date_to: str = None
    ) -> list[dict]:
        """Filter matches by date range."""
        if not date_from and not date_to:
            return matches

        import logging
        logger = logging.getLogger(__name__)

        # Parse date bounds
        from_ts = None
        to_ts = None

        if date_from:
            try:
                from_dt = datetime.strptime(date_from, "%Y-%m-%d")
                from_ts = int(from_dt.timestamp())
            except ValueError:
                logger.warning(f"Failed to parse date_from: {date_from}")

        if date_to:
            try:
                to_dt = datetime.strptime(date_to, "%Y-%m-%d")
                # Include the entire end date
                to_ts = int(to_dt.timestamp()) + 86400  # +1 day
            except ValueError:
                logger.warning(f"Failed to parse date_to: {date_to}")

        logger.info(f"Date filtering: from_ts={from_ts}, to_ts={to_ts}, matches={len(matches)}")

        filtered = []
        excluded_count = 0
        excluded_reasons = {"no_ts": 0, "before_from": 0, "after_to": 0}
        debug_samples = []  # Log first few for debugging

        for match in matches:
            meta = match.get("metadata", {})
            # Try timestamp_unix first, then parse formatted_date
            msg_ts = meta.get("timestamp_unix")
            ts_source = "timestamp_unix" if msg_ts is not None else None

            # Ensure msg_ts is an integer (ChromaDB might return string)
            if msg_ts is not None:
                try:
                    msg_ts = int(msg_ts)
                except (ValueError, TypeError):
                    msg_ts = None
                    ts_source = None

            if msg_ts is None:
                # Try to parse formatted_date (DD.MM.YYYY HH:MM)
                formatted = meta.get("formatted_date", "")
                if formatted:
                    try:
                        dt = datetime.strptime(formatted.split()[0], "%d.%m.%Y")
                        msg_ts = int(dt.timestamp())
                        ts_source = "formatted_date"
                    except ValueError:
                        pass

            # Log first 3 matches for debugging
            if len(debug_samples) < 3:
                debug_samples.append({
                    "formatted_date": meta.get("formatted_date"),
                    "timestamp_unix": meta.get("timestamp_unix"),
                    "msg_ts": msg_ts,
                    "ts_source": ts_source,
                    "to_ts": to_ts,
                    "should_exclude": msg_ts is None or (to_ts and msg_ts > to_ts)
                })

            if msg_ts is None:
                # Can't determine date, EXCLUDE by default for date-filtered queries
                excluded_count += 1
                excluded_reasons["no_ts"] += 1
                continue

            # Apply filter
            if from_ts and msg_ts < from_ts:
                excluded_count += 1
                excluded_reasons["before_from"] += 1
                continue
            if to_ts and msg_ts > to_ts:
                excluded_count += 1
                excluded_reasons["after_to"] += 1
                continue

            filtered.append(match)

        logger.info(f"Date filter debug samples: {debug_samples}")
        logger.info(f"Date filter: {len(matches)} -> {len(filtered)} (excluded {excluded_count}: {excluded_reasons})")
        return filtered

    def search_by_user(
        self,
        query: str,
        user_identifier: str,
        n_results: int = 10,
        date_from: str = None,
        date_to: str = None
    ) -> list[dict]:
        """
        Search messages from specific user about a topic.
        user_identifier can be username (@name) or user_id.
        """
        # Determine if it's username or user_id
        if user_identifier.startswith("@"):
            return self.search(
                query=query,
                n_results=n_results,
                username=user_identifier,
                date_from=date_from,
                date_to=date_to
            )
        elif user_identifier.isdigit():
            return self.search(
                query=query,
                n_results=n_results,
                user_id=int(user_identifier),
                date_from=date_from,
                date_to=date_to
            )
        else:
            # Try as username without @
            return self.search(
                query=query,
                n_results=n_results,
                username=user_identifier,
                date_from=date_from,
                date_to=date_to
            )

    def get_user_messages_about(
        self,
        user_identifier: str,
        topic: str,
        n_results: int = 20
    ) -> list[dict]:
        """Get all user's messages related to a topic (for flip detection)."""
        return self.search_by_user(
            query=topic,
            user_identifier=user_identifier,
            n_results=n_results
        )

    def count(self) -> int:
        """Get total count of indexed messages."""
        return self.collection.count()

    def delete_all(self):
        """Clear all data (for reindexing)."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                # Memory optimizations for large collections
                "hnsw:M": 8,  # Reduced from default 16 - less memory
                "hnsw:construction_ef": 64,  # Reduced from default 100
                "hnsw:search_ef": 32,  # Reduced for faster searches
            }
        )
