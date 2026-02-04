#!/usr/bin/env python3
"""
Migrate database to add chat_id column to messages table.
Run this once before reindexing.
"""
import sqlite3
from pathlib import Path

try:
    import config
    DB_PATH = config.SQLITE_DB_PATH
except ImportError:
    DB_PATH = Path("data/chat_history.db")

print(f"Database path: {DB_PATH}")


def migrate():
    if not DB_PATH.exists():
        print("Database doesn't exist yet. No migration needed.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if chat_id column exists
    cursor.execute("PRAGMA table_info(messages)")
    columns = {row[1] for row in cursor.fetchall()}

    if "chat_id" not in columns:
        print("Adding chat_id column to messages table...")
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

        # Set default for any remaining nulls
        cursor.execute("UPDATE messages SET chat_id = 1101664871 WHERE chat_id IS NULL")
        conn.commit()

    # Check for existing unique constraint on message_id alone
    cursor.execute("PRAGMA index_list(messages)")
    indexes = cursor.fetchall()
    print(f"Current indexes: {indexes}")

    # SQLite can't drop constraints easily, so we recreate the table
    print("\nRecreating messages table without old unique constraint...")

    # Create new table with correct schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages_new (
            id INTEGER PRIMARY KEY,
            message_id BIGINT NOT NULL,
            chat_id BIGINT NOT NULL,
            export_id INTEGER NOT NULL REFERENCES exports(id),
            user_id BIGINT,
            username VARCHAR(255),
            first_name VARCHAR(255),
            last_name VARCHAR(255),
            text TEXT,
            timestamp DATETIME NOT NULL,
            timestamp_unix BIGINT,
            reply_to_message_id BIGINT,
            vector_id VARCHAR(255),
            is_embedded BOOLEAN DEFAULT 0,
            UNIQUE(message_id, chat_id)
        )
    """)

    # Copy data
    print("Copying data to new table...")
    cursor.execute("""
        INSERT OR IGNORE INTO messages_new
        SELECT id, message_id, chat_id, export_id, user_id, username,
               first_name, last_name, text, timestamp, timestamp_unix,
               reply_to_message_id, vector_id, is_embedded
        FROM messages
    """)

    # Drop old table and rename
    print("Replacing old table...")
    cursor.execute("DROP TABLE messages")
    cursor.execute("ALTER TABLE messages_new RENAME TO messages")

    # Recreate indexes
    print("Creating indexes...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_message_id ON messages(message_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_reply_to ON messages(reply_to_message_id)")

    conn.commit()
    conn.close()
    print("\nMigration complete!")


if __name__ == "__main__":
    migrate()
