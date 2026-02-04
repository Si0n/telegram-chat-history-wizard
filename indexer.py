"""
Index chat exports into the database and vector store.
"""
import logging
from pathlib import Path
from typing import Optional

import config
from db import Database, Export
from search import EmbeddingService, VectorStore
from ingestion.parser import TelegramExportParser, find_exports

logger = logging.getLogger(__name__)

# Allowed chat IDs (supergroup + old basic group before migration)
ALLOWED_CHAT_IDS = {
    1101664871,  # Current supergroup
    232481601,   # Old basic group (pre-migration, 2019 messages)
}


class Indexer:
    """Index Telegram exports into searchable database."""

    def __init__(
        self,
        db: Database = None,
        vector_store: VectorStore = None,
        embedding_service: EmbeddingService = None
    ):
        self.db = db or Database(config.SQLITE_DB_PATH)
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore(
            embedding_service=self.embedding_service
        )

    def index_export(
        self,
        export_path: Path,
        progress_callback=None
    ) -> dict:
        """
        Index a single export directory.
        Returns stats about the indexing.
        """
        parser = TelegramExportParser(export_path)

        # Get metadata
        metadata = parser.get_metadata()
        logger.info(f"Indexing export: {metadata.chat_name} (ID: {metadata.chat_id})")

        # Validate chat_id against allowed list
        if metadata.chat_id not in ALLOWED_CHAT_IDS:
            raise ValueError(
                f"Unknown chat ID {metadata.chat_id}. Allowed: {ALLOWED_CHAT_IDS}"
            )

        # Create export record
        export = self.db.create_export(
            filename=export_path.name,
            chat_id=metadata.chat_id,
            chat_name=metadata.chat_name
        )

        # Process messages
        stats = {
            "total_parsed": 0,
            "new_messages": 0,
            "skipped_duplicates": 0,
            "export_id": export.id
        }

        batch = []
        batch_size = 500

        for msg in parser.stream_messages():
            stats["total_parsed"] += 1

            # Check for duplicate
            if self.db.message_exists(msg.message_id):
                stats["skipped_duplicates"] += 1
                continue

            batch.append(msg.to_dict())

            # Insert batch
            if len(batch) >= batch_size:
                inserted = self.db.bulk_insert_messages(batch, export.id)
                stats["new_messages"] += inserted
                batch = []

                if progress_callback:
                    progress_callback(stats)

                logger.info(
                    f"Progress: {stats['total_parsed']} parsed, "
                    f"{stats['new_messages']} new, "
                    f"{stats['skipped_duplicates']} duplicates"
                )

        # Insert remaining
        if batch:
            inserted = self.db.bulk_insert_messages(batch, export.id)
            stats["new_messages"] += inserted

        # Update export stats
        self.db.update_export_stats(export.id)

        logger.info(f"Export indexed: {stats}")
        return stats

    def create_embeddings(
        self,
        batch_size: int = None,
        progress_callback=None
    ) -> dict:
        """
        Create embeddings for messages that don't have them.
        """
        batch_size = batch_size or config.EMBEDDING_BATCH_SIZE

        stats = {
            "total_embedded": 0,
            "batches_processed": 0
        }

        while True:
            # Get messages without embeddings
            messages = self.db.get_messages_without_embeddings(limit=batch_size)

            if not messages:
                break

            # Prepare batch
            texts = []
            db_ids = []
            metadatas = []

            for msg in messages:
                if not msg.text or not msg.text.strip():
                    continue

                texts.append(msg.text)
                db_ids.append(msg.id)
                metadatas.append({
                    "message_id": msg.message_id,
                    "user_id": msg.user_id or 0,
                    "username": msg.username or "",
                    "display_name": msg.display_name,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else "",
                    "formatted_date": msg.formatted_date
                })

            if not texts:
                break

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} messages...")
            embeddings = self.embedding_service.embed_batch(texts)

            # Store in vector DB
            vector_ids = self.vector_store.add_messages(
                db_ids=db_ids,
                texts=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )

            # Mark as embedded
            self.db.mark_messages_embedded(
                message_ids=[m.id for m in messages if m.text],
                vector_ids=vector_ids
            )

            stats["total_embedded"] += len(texts)
            stats["batches_processed"] += 1

            if progress_callback:
                progress_callback(stats)

            logger.info(f"Embedded {stats['total_embedded']} messages so far...")

        logger.info(f"Embedding complete: {stats}")
        return stats

    def index_all_exports(self, progress_callback=None) -> dict:
        """Index all exports in the chat_exports directory."""
        exports = find_exports()

        if not exports:
            logger.warning("No exports found in chat_exports/")
            return {"exports_processed": 0}

        total_stats = {
            "exports_processed": 0,
            "total_new_messages": 0,
            "total_duplicates": 0
        }

        for export_path in exports:
            logger.info(f"Processing: {export_path}")
            try:
                stats = self.index_export(export_path, progress_callback)
                total_stats["exports_processed"] += 1
                total_stats["total_new_messages"] += stats["new_messages"]
                total_stats["total_duplicates"] += stats["skipped_duplicates"]
            except Exception as e:
                logger.error(f"Failed to index {export_path}: {e}")

        return total_stats

    def full_reindex(self, progress_callback=None) -> dict:
        """Full reindex: parse all exports and create all embeddings."""
        logger.info("Starting full reindex...")

        # Index all exports
        index_stats = self.index_all_exports(progress_callback)

        # Create embeddings
        embed_stats = self.create_embeddings(progress_callback=progress_callback)

        return {
            **index_stats,
            **embed_stats
        }


def run_indexer():
    """CLI entry point for indexer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    indexer = Indexer()

    def progress(stats):
        print(f"  Progress: {stats}")

    print("Starting indexing...")
    stats = indexer.full_reindex(progress_callback=progress)

    print("\n" + "=" * 50)
    print("Indexing complete!")
    print(f"  Exports processed: {stats.get('exports_processed', 0)}")
    print(f"  New messages: {stats.get('total_new_messages', 0)}")
    print(f"  Duplicates skipped: {stats.get('total_duplicates', 0)}")
    print(f"  Messages embedded: {stats.get('total_embedded', 0)}")


if __name__ == "__main__":
    run_indexer()
