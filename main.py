#!/usr/bin/env python3
"""
Telegram Chat History Wizard

A bot for searching through Telegram chat history exports
with semantic search and flip detection capabilities.

Usage:
    python main.py index    - Index all exports (first run)
    python main.py bot      - Run the Telegram bot
    python main.py stats    - Show database statistics
    python main.py reindex  - Force full reindex
"""
import sys
import gc
import logging
from telegram.ext import Application

import config
from db import Database
from search import VectorStore, EmbeddingService
from search.entity_aliases import init_entity_aliases
from bot import setup_handlers
from indexer import Indexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_bot():
    """Run the Telegram bot."""
    if not config.TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env file")
        print("1. Copy .env.example to .env")
        print("2. Add your bot token from @BotFather")
        sys.exit(1)

    if not config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set in .env file")
        sys.exit(1)

    # Aggressive garbage collection to free memory
    gc.collect()

    # Initialize components
    db = Database(config.SQLITE_DB_PATH)
    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service=embedding_service)

    # Free memory after vector store initialization
    gc.collect()

    # Initialize entity aliases with database
    init_entity_aliases(db)

    # Check if database has messages
    stats = db.get_stats()
    if stats["total_messages"] == 0:
        print("WARNING: No messages indexed yet!")
        print("Run 'python main.py index' first to index your chat exports.")
        print()

    # Create application
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Setup handlers
    setup_handlers(app, db, vector_store)

    # Run
    print("=" * 50)
    print("Telegram Chat History Wizard Bot")
    print("=" * 50)
    print(f"Messages in database: {stats['total_messages']:,}")
    print(f"Messages with embeddings: {vector_store.count():,}")
    print()
    print("Bot is running... Press Ctrl+C to stop.")
    print()

    app.run_polling(allowed_updates=["message", "callback_query"])


def run_index():
    """Index all chat exports."""
    if not config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set in .env file")
        print("Required for generating embeddings.")
        sys.exit(1)

    print("=" * 50)
    print("Indexing Chat Exports")
    print("=" * 50)

    indexer = Indexer()

    def progress(stats):
        msg_count = stats.get("new_messages", stats.get("total_embedded", 0))
        print(f"  ... {msg_count} messages processed")

    stats = indexer.full_reindex(progress_callback=progress)

    print()
    print("=" * 50)
    print("Indexing Complete!")
    print("=" * 50)
    print(f"Exports processed: {stats.get('exports_processed', 0)}")
    print(f"New messages indexed: {stats.get('total_new_messages', 0)}")
    print(f"Duplicates skipped: {stats.get('total_duplicates', 0)}")
    print(f"Embeddings created: {stats.get('total_embedded', 0)}")
    print()
    print("You can now run 'python main.py bot' to start the bot.")


def show_stats():
    """Show database statistics."""
    db = Database(config.SQLITE_DB_PATH)
    vector_store = VectorStore()

    stats = db.get_stats()
    stats["embedded_messages"] = vector_store.count()

    print("=" * 50)
    print("Database Statistics")
    print("=" * 50)
    print(f"Total messages: {stats['total_messages']:,}")
    print(f"Embedded messages: {stats['embedded_messages']:,}")
    print(f"Unique users: {stats['unique_users']:,}")
    print(f"Exports processed: {stats['exports_count']}")

    if stats.get("date_start") and stats.get("date_end"):
        print(f"Date range: {stats['date_start'].strftime('%d.%m.%Y')} - {stats['date_end'].strftime('%d.%m.%Y')}")

    # List exports
    exports = db.list_exports()
    if exports:
        print()
        print("Processed exports:")
        for exp in exports:
            print(f"  - {exp.filename}: {exp.message_count:,} messages")


def run_reindex():
    """Force full reindex (clear and rebuild)."""
    print("WARNING: This will clear and rebuild the entire index.")
    confirm = input("Are you sure? (yes/no): ")

    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    # Clear vector store
    vector_store = VectorStore()
    vector_store.delete_all()
    print("Vector store cleared.")

    # Rerun indexing
    run_index()


def print_help():
    """Print usage help."""
    print(__doc__)
    print("Commands:")
    print("  index    - Index all exports + create embeddings")
    print("  import   - Import exports only (no embeddings)")
    print("  embed    - Create embeddings for imported messages")
    print("  bot      - Run the Telegram bot")
    print("  stats    - Show database statistics")
    print("  reindex  - Force full reindex (clears existing data)")
    print()
    print("Setup:")
    print("  1. Copy .env.example to .env")
    print("  2. Add your TELEGRAM_BOT_TOKEN and OPENAI_API_KEY")
    print("  3. Put your Telegram export in chat_exports/")
    print("  4. Run 'python main.py index'")
    print("  5. Run 'python main.py bot'")


def run_import_only():
    """Import messages without creating embeddings."""
    import time
    print("=" * 50)
    print("Importing Chat Exports (no embeddings)")
    print("=" * 50)

    indexer = Indexer()
    start_time = time.time()
    last_print = [0]

    def progress(stats):
        now = time.time()
        if now - last_print[0] >= 2:  # Print every 2 seconds
            elapsed = int(now - start_time)
            new_msgs = stats.get('new_messages', 0)
            total = stats.get('total_parsed', 0)
            rate = new_msgs / max(elapsed, 1)
            print(f"\r  Progress: {new_msgs:,} new / {total:,} parsed ({rate:.0f}/s)", end="", flush=True)
            last_print[0] = now

    stats = indexer.index_all_exports(progress_callback=progress)
    elapsed = int(time.time() - start_time)

    print(f"\n\n{'=' * 50}")
    print(f"Import complete in {elapsed}s!")
    print(f"  Exports processed: {stats.get('exports_processed', 0)}")
    print(f"  New messages: {stats.get('total_new_messages', 0):,}")
    print(f"  Duplicates skipped: {stats.get('total_duplicates', 0):,}")
    print("\nRun 'python main.py embed' to create embeddings.")


def run_embed_only():
    """Create embeddings for messages that don't have them."""
    import time
    print("=" * 50)
    print("Creating Embeddings")
    print("=" * 50)

    indexer = Indexer()
    start_time = time.time()
    last_print = [0]

    def progress(stats):
        now = time.time()
        if now - last_print[0] >= 2:  # Print every 2 seconds
            elapsed = int(now - start_time)
            embedded = stats.get('total_embedded', 0)
            chunks = stats.get('total_chunks', 0)
            rate = embedded / max(elapsed, 1)
            print(f"\r  Progress: {embedded:,} messages ({chunks:,} chunks) - {rate:.0f} msg/s", end="", flush=True)
            last_print[0] = now

    stats = indexer.create_embeddings(progress_callback=progress)
    elapsed = int(time.time() - start_time)

    print(f"\n\n{'=' * 50}")
    print(f"Embedding complete in {elapsed}s!")
    print(f"  Messages embedded: {stats.get('total_embedded', 0):,}")
    print(f"  Total chunks: {stats.get('total_chunks', 0):,}")
    print(f"  Batches processed: {stats.get('batches_processed', 0)}")


def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "index":
        run_index()
    elif command == "import":
        run_import_only()
    elif command == "embed":
        run_embed_only()
    elif command == "bot":
        run_bot()
    elif command == "stats":
        show_stats()
    elif command == "reindex":
        run_reindex()
    elif command in ("help", "-h", "--help"):
        print_help()
    else:
        print(f"Unknown command: {command}")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
