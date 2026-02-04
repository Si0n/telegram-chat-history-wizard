#!/usr/bin/env python3
"""
Migrate database to add chat_id column to messages table.
Run this once before reindexing.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path("data/chat_history.db")


def migrate():
    if not DB_PATH.exists():
        print("Database doesn't exist yet. No migration needed.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if chat_id column exists
    cursor.execute("PRAGMA table_info(messages)")
    columns = {row[1] for row in cursor.fetchall()}

    if "chat_id" in columns:
        print("chat_id column already exists. No migration needed.")
        conn.close()
        return

    print("Adding chat_id column to messages table...")

    # Add chat_id column
    cursor.execute("ALTER TABLE messages ADD COLUMN chat_id BIGINT")

    # Get chat_id from related export for existing messages
    print("Populating chat_id from exports...")
    cursor.execute("""
        UPDATE messages
        SET chat_id = (
            SELECT exports.chat_id
            FROM exports
            WHERE exports.id = messages.export_id
        )
    """)

    # Set default for any remaining nulls (shouldn't happen)
    cursor.execute("UPDATE messages SET chat_id = 1101664871 WHERE chat_id IS NULL")

    # Drop old unique constraint and create new one
    # SQLite doesn't support dropping constraints, so we need to recreate the table
    # For now, just add an index
    print("Creating index on chat_id...")
    try:
        cursor.execute("CREATE INDEX idx_messages_chat_id ON messages(chat_id)")
    except sqlite3.OperationalError:
        print("Index already exists")

    # Create composite unique index (instead of constraint)
    print("Creating composite unique index...")
    try:
        cursor.execute("CREATE UNIQUE INDEX uq_message_chat ON messages(message_id, chat_id)")
    except sqlite3.OperationalError as e:
        if "already exists" in str(e):
            print("Unique index already exists")
        else:
            print(f"Note: {e}")
            print("This is OK if you're reindexing from scratch")

    conn.commit()
    conn.close()
    print("Migration complete!")


if __name__ == "__main__":
    migrate()
