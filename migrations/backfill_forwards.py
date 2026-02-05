"""
Migration script to backfill forwarded message metadata for existing messages.

This script:
1. Reads the original JSON exports
2. Updates only is_forwarded, forward_from, forward_date columns
3. Does NOT touch embeddings (fast operation)

Usage:
    python -m migrations.backfill_forwards
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import ijson

import config
from db import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_forward_info(msg: dict) -> tuple[bool, str | None, datetime | None]:
    """
    Extract forward metadata from a message dict.

    Returns:
        (is_forwarded, forward_from, forward_date)
    """
    is_forwarded = False
    forward_from = None
    forward_date = None

    # Check for forwarded_from (Telegram export format)
    forwarded_from = msg.get("forwarded_from")
    if forwarded_from:
        is_forwarded = True
        forward_from = forwarded_from

    # Alternative format: forward_from
    if not is_forwarded:
        fwd_from = msg.get("forward_from")
        if fwd_from:
            is_forwarded = True
            forward_from = fwd_from

    # Check for forward_date
    fwd_date_str = msg.get("forward_date")
    if fwd_date_str:
        is_forwarded = True
        try:
            forward_date = datetime.fromisoformat(fwd_date_str)
        except (ValueError, TypeError):
            pass

    return is_forwarded, forward_from, forward_date


def find_json_exports() -> list[Path]:
    """Find all result.json files in chat_exports directory."""
    exports = []
    for path in config.CHAT_EXPORTS_DIR.iterdir():
        if path.is_dir():
            result_json = path / "result.json"
            if result_json.exists():
                exports.append(result_json)
    return sorted(exports)


def stream_messages_with_forwards(json_path: Path):
    """
    Stream messages from export file, yielding only those with forward info.

    Yields:
        (message_id, is_forwarded, forward_from, forward_date)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        messages = ijson.items(f, "messages.item")

        for msg in messages:
            # Skip non-messages
            if msg.get("type") != "message":
                continue

            message_id = msg.get("id")
            if not message_id:
                continue

            is_forwarded, forward_from, forward_date = extract_forward_info(msg)

            # Only yield if it's a forward
            if is_forwarded:
                yield message_id, is_forwarded, forward_from, forward_date


def run_migration(batch_size: int = 1000, dry_run: bool = False):
    """
    Run the migration to backfill forward metadata.

    Args:
        batch_size: Number of updates per batch
        dry_run: If True, don't actually update, just count
    """
    db = Database(config.SQLITE_DB_PATH)
    exports = find_json_exports()

    if not exports:
        logger.warning("No exports found in chat_exports/")
        return

    logger.info(f"Found {len(exports)} export(s) to process")

    total_forwards = 0
    total_updated = 0

    for json_path in exports:
        logger.info(f"Processing: {json_path.parent.name}")

        batch = []
        forwards_in_export = 0

        for message_id, is_forwarded, forward_from, forward_date in stream_messages_with_forwards(json_path):
            forwards_in_export += 1
            batch.append((message_id, is_forwarded, forward_from, forward_date))

            if len(batch) >= batch_size:
                if not dry_run:
                    updated = update_batch(db, batch)
                    total_updated += updated
                batch = []

                if forwards_in_export % 5000 == 0:
                    logger.info(f"  Processed {forwards_in_export} forwards...")

        # Process remaining batch
        if batch and not dry_run:
            updated = update_batch(db, batch)
            total_updated += updated

        total_forwards += forwards_in_export
        logger.info(f"  Found {forwards_in_export} forwarded messages in this export")

    logger.info("=" * 50)
    logger.info(f"Migration complete!")
    logger.info(f"Total forwards found: {total_forwards}")
    if not dry_run:
        logger.info(f"Total records updated: {total_updated}")
    else:
        logger.info("(Dry run - no changes made)")


def update_batch(db: Database, batch: list[tuple]) -> int:
    """
    Update a batch of messages with forward metadata.

    Returns count of successfully updated records.
    """
    updated = 0

    with db.get_session() as session:
        from db.models import Message

        for message_id, is_forwarded, forward_from, forward_date in batch:
            result = session.query(Message).filter(
                Message.message_id == message_id
            ).update({
                "is_forwarded": is_forwarded,
                "forward_from": forward_from,
                "forward_date": forward_date
            })
            updated += result

        session.commit()

    return updated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backfill forwarded message metadata")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count forwards without updating database"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of updates per batch (default: 1000)"
    )

    args = parser.parse_args()

    run_migration(batch_size=args.batch_size, dry_run=args.dry_run)
