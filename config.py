import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHAT_EXPORTS_DIR = BASE_DIR / "chat_exports"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CHAT_EXPORTS_DIR.mkdir(exist_ok=True)

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Database
SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", DATA_DIR / "metadata.db"))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", DATA_DIR / "chroma"))

# Processing
EMBEDDING_BATCH_SIZE = 30  # Messages per batch for embedding (low for memory)
MAX_MESSAGE_LENGTH = 8000   # Truncate very long messages

# Search
DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 20
