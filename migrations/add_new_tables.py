"""
Migration script to add new tables and columns for search improvements.

This script:
1. Adds new columns to messages table (is_forwarded, forward_from, forward_date)
2. Creates new tables (relevance_cache, search_feedback)

Usage:
    python -m migrations.add_new_tables
"""
import logging
from sqlalchemy import text

import config
from db import Database
from db.models import Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def add_message_columns(db: Database):
    """Add new columns to messages table if they don't exist."""
    with db.get_session() as session:
        # Check existing columns
        result = session.execute(text("PRAGMA table_info(messages)"))
        existing_columns = {row[1] for row in result.fetchall()}

        # Add missing columns
        if "is_forwarded" not in existing_columns:
            logger.info("Adding column: messages.is_forwarded")
            session.execute(text("ALTER TABLE messages ADD COLUMN is_forwarded BOOLEAN DEFAULT 0"))

        if "forward_from" not in existing_columns:
            logger.info("Adding column: messages.forward_from")
            session.execute(text("ALTER TABLE messages ADD COLUMN forward_from VARCHAR(255)"))

        if "forward_date" not in existing_columns:
            logger.info("Adding column: messages.forward_date")
            session.execute(text("ALTER TABLE messages ADD COLUMN forward_date DATETIME"))

        session.commit()


def create_new_tables(db: Database):
    """Create new tables (relevance_cache, search_feedback) if they don't exist."""
    # This will create any missing tables defined in models
    Base.metadata.create_all(db.engine)
    logger.info("New tables created (if missing): relevance_cache, search_feedback")


def run_migration():
    """Run the full schema migration."""
    logger.info("Starting schema migration...")

    db = Database(config.SQLITE_DB_PATH)

    # Step 1: Add new columns to messages table
    logger.info("Step 1: Adding new columns to messages table...")
    add_message_columns(db)

    # Step 2: Create new tables
    logger.info("Step 2: Creating new tables...")
    create_new_tables(db)

    logger.info("=" * 50)
    logger.info("Schema migration complete!")
    logger.info("You can now run: python -m migrations.backfill_forwards")


if __name__ == "__main__":
    run_migration()
