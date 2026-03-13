#!/usr/bin/env python3
"""
Telegram Chat History Wizard

An agent-powered bot for searching through chat history.

Usage:
    python main.py bot    - Run the Telegram bot
    python main.py help   - Show this help
"""
import sys
import logging

from telegram.ext import Application

import config
from db import Database
from agent import AgentLoop
from bot import setup_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_bot():
    """Run the Telegram bot."""
    if not config.TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env file")
        sys.exit(1)
    if not config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set in .env file")
        sys.exit(1)

    db = Database(config.SQLITE_DB_PATH)
    agent = AgentLoop(db)

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    setup_handlers(app, db, agent)

    print("=" * 50)
    print("Chat History Wizard — Agent Mode")
    print("=" * 50)
    print(f"Model: {config.CHAT_MODEL}")
    print(f"Max iterations: {config.AGENT_MAX_ITERATIONS}")
    print()
    print("Bot is running... Press Ctrl+C to stop.")

    app.run_polling(allowed_updates=["message", "callback_query"])


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("help", "-h", "--help"):
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()
    if command == "bot":
        run_bot()
    else:
        print(f"Unknown command: {command}")
        print("Use 'python main.py bot' to start the bot.")
        sys.exit(1)


if __name__ == "__main__":
    main()
