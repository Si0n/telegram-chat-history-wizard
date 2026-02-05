"""
Index chat exports into the database and vector store.
"""
import gc
import logging
import os
from pathlib import Path
from typing import Optional

import config


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0
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

        chat_id = metadata.chat_id

        for msg in parser.stream_messages():
            stats["total_parsed"] += 1

            # Check for duplicate (same message_id in same chat)
            if self.db.message_exists(msg.message_id, chat_id):
                stats["skipped_duplicates"] += 1
                continue

            msg_data = msg.to_dict()
            msg_data["chat_id"] = chat_id
            batch.append(msg_data)

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

    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
        """Split long text into overlapping chunks."""
        chunk_size = chunk_size or config.MAX_MESSAGE_LENGTH
        overlap = overlap or getattr(config, 'CHUNK_OVERLAP', 200)

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence/word boundary
            if end < len(text):
                # Look for sentence end
                for sep in ['. ', '! ', '? ', '\n', ' ']:
                    last_sep = text.rfind(sep, start + chunk_size // 2, end)
                    if last_sep > start:
                        end = last_sep + 1
                        break

            chunks.append(text[start:end].strip())
            start = end - overlap

        return [c for c in chunks if c]  # Filter empty chunks

    def create_embeddings(
        self,
        batch_size: int = None,
        progress_callback=None
    ) -> dict:
        """
        Create embeddings for messages that don't have them.
        Long messages are split into chunks.
        Uses parallel API calls and large ChromaDB batch inserts for performance.

        Optimized for 8GB RAM with:
        - Larger message fetch batches (batch_size * 4)
        - Parallel embedding API calls (EMBEDDING_WORKERS)
        - Large ChromaDB batch inserts (CHROMA_BATCH_SIZE)
        """
        import time

        batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        chroma_batch_size = getattr(config, 'CHROMA_BATCH_SIZE', 2500)
        embedding_workers = getattr(config, 'EMBEDDING_WORKERS', 3)

        # Fetch larger batches from DB (we'll process in parallel)
        fetch_batch_size = batch_size * embedding_workers

        # Get total counts for progress display
        db_stats = self.db.get_stats()
        total_messages = db_stats.get("total_messages", 0)
        already_embedded = db_stats.get("embedded_messages", 0)
        remaining = total_messages - already_embedded

        logger.info(f"=== Embedding Progress (Optimized for 8GB RAM) ===")
        logger.info(f"Total messages: {total_messages:,}")
        logger.info(f"Already embedded: {already_embedded:,}")
        logger.info(f"Remaining: {remaining:,}")
        logger.info(f"Batch size: {batch_size} | Workers: {embedding_workers} | ChromaDB batch: {chroma_batch_size}")
        logger.info(f"==================================================")

        stats = {
            "total_embedded": 0,
            "total_chunks": 0,
            "batches_processed": 0,
            "total_messages": total_messages,
            "already_embedded": already_embedded,
            "remaining_start": remaining
        }

        start_time = time.time()

        # Accumulators for ChromaDB batch insert
        acc_texts = []
        acc_db_ids = []
        acc_metadatas = []
        acc_embeddings = []
        acc_message_ids = []

        # Track fetched message IDs to avoid re-fetching before ChromaDB insert
        fetched_ids = set()

        while True:
            # Get messages without embeddings (larger batch for parallel processing)
            messages = self.db.get_messages_without_embeddings(
                limit=fetch_batch_size,
                exclude_ids=fetched_ids
            )

            if not messages:
                break

            # Track these messages
            for m in messages:
                fetched_ids.add(m.id)

            # Prepare batch with chunking
            texts = []
            db_ids = []
            metadatas = []
            msg_ids_with_text = []

            for msg in messages:
                if not msg.text or not msg.text.strip():
                    continue

                msg_ids_with_text.append(msg.id)

                # Split long messages into chunks
                chunks = self._chunk_text(msg.text)

                for i, chunk in enumerate(chunks):
                    texts.append(chunk)
                    db_ids.append(msg.id)

                    # Calculate timestamp_unix for date filtering
                    ts_unix = None
                    if msg.timestamp_unix:
                        ts_unix = msg.timestamp_unix
                    elif msg.timestamp:
                        ts_unix = int(msg.timestamp.timestamp())

                    meta = {
                        "message_id": msg.message_id,
                        "user_id": msg.user_id or 0,
                        "username": msg.username or "",
                        "display_name": msg.display_name,
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else "",
                        "timestamp_unix": ts_unix,  # Include for date filtering
                        "formatted_date": msg.formatted_date
                    }
                    if len(chunks) > 1:
                        meta["chunk"] = i + 1
                        meta["total_chunks"] = len(chunks)
                    metadatas.append(meta)

            if not texts:
                # All messages in batch were empty, mark them as embedded anyway
                self.db.mark_messages_embedded(
                    message_ids=[m.id for m in messages],
                    vector_ids=[]
                )
                del messages
                gc.collect()
                continue

            # Generate embeddings with parallel API calls
            logger.info(f"Generating embeddings for {len(texts)} chunks ({embedding_workers} parallel workers)...")
            embeddings = self.embedding_service.embed_parallel(
                texts,
                batch_size=batch_size,
                max_workers=embedding_workers
            )

            # Accumulate for ChromaDB batch insert
            acc_texts.extend(texts)
            acc_db_ids.extend(db_ids)
            acc_metadatas.extend(metadatas)
            acc_embeddings.extend(embeddings)
            acc_message_ids.extend(msg_ids_with_text)

            stats["total_embedded"] += len(messages)
            stats["total_chunks"] += len(texts)
            stats["batches_processed"] += 1

            # Add to ChromaDB when accumulated enough
            if len(acc_texts) >= chroma_batch_size:
                logger.info(f"Adding {len(acc_texts)} vectors to ChromaDB...")
                vector_ids = self.vector_store.add_messages(
                    db_ids=acc_db_ids,
                    texts=acc_texts,
                    metadatas=acc_metadatas,
                    embeddings=acc_embeddings
                )
                self.db.mark_messages_embedded(
                    message_ids=acc_message_ids,
                    vector_ids=vector_ids
                )
                # Clear accumulators and tracking set
                acc_texts, acc_db_ids, acc_metadatas, acc_embeddings, acc_message_ids = [], [], [], [], []
                fetched_ids.clear()
                gc.collect()

            if progress_callback:
                progress_callback(stats)

            mem_mb = get_memory_mb()
            done = already_embedded + stats['total_embedded']
            remaining_now = total_messages - done
            pct = (done / total_messages * 100) if total_messages > 0 else 0

            # Calculate ETA
            elapsed = time.time() - start_time
            if stats['total_embedded'] > 0 and elapsed > 0:
                rate = stats['total_embedded'] / elapsed  # msgs per second
                eta_seconds = remaining_now / rate if rate > 0 else 0
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                eta_str = f"ETA: {eta_min}m {eta_sec}s" if eta_min < 60 else f"ETA: {eta_min//60}h {eta_min%60}m"
            else:
                eta_str = "ETA: calculating..."
                rate = 0

            logger.info(f"Progress: {done:,}/{total_messages:,} ({pct:.1f}%) | Remaining: {remaining_now:,} | {rate:.1f} msg/s | {eta_str} | Mem: {mem_mb:.0f}MB")

            del texts, embeddings, metadatas, db_ids, messages
            gc.collect()

        # Flush remaining accumulated embeddings
        if acc_texts:
            logger.info(f"Adding final {len(acc_texts)} vectors to ChromaDB...")
            vector_ids = self.vector_store.add_messages(
                db_ids=acc_db_ids,
                texts=acc_texts,
                metadatas=acc_metadatas,
                embeddings=acc_embeddings
            )
            self.db.mark_messages_embedded(
                message_ids=acc_message_ids,
                vector_ids=vector_ids
            )

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
