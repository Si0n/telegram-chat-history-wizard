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
            metadata={"hnsw:space": "cosine"}
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

        # Create unique IDs
        vector_ids = [f"msg_{db_id}" for db_id in db_ids]

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

        # Parse date bounds
        from_ts = None
        to_ts = None

        if date_from:
            try:
                from_dt = datetime.strptime(date_from, "%Y-%m-%d")
                from_ts = int(from_dt.timestamp())
            except ValueError:
                pass

        if date_to:
            try:
                to_dt = datetime.strptime(date_to, "%Y-%m-%d")
                # Include the entire end date
                to_ts = int(to_dt.timestamp()) + 86400  # +1 day
            except ValueError:
                pass

        filtered = []
        for match in matches:
            meta = match.get("metadata", {})
            # Try timestamp_unix first, then parse formatted_date
            msg_ts = meta.get("timestamp_unix")

            if msg_ts is None:
                # Try to parse formatted_date (DD.MM.YYYY HH:MM)
                formatted = meta.get("formatted_date", "")
                if formatted:
                    try:
                        dt = datetime.strptime(formatted.split()[0], "%d.%m.%Y")
                        msg_ts = int(dt.timestamp())
                    except ValueError:
                        pass

            if msg_ts is None:
                # Can't determine date, include by default
                filtered.append(match)
                continue

            # Apply filter
            if from_ts and msg_ts < from_ts:
                continue
            if to_ts and msg_ts > to_ts:
                continue

            filtered.append(match)

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
            metadata={"hnsw:space": "cosine"}
        )
