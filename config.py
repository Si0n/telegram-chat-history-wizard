import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Credentials
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Database
SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", DATA_DIR / "metadata.db"))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", DATA_DIR / "chroma"))

# Models
CHAT_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-large"  # Must match existing index

# Agent
AGENT_MAX_ITERATIONS = 3
AGENT_QUERY_TIMEOUT = 5        # seconds
AGENT_MAX_RESULTS = 50

# Display
RESULTS_PER_PAGE = 3
DIALOGUE_INITIAL_WINDOW = 5    # 2 before + selected + 2 after
DIALOGUE_SCROLL_SIZE = 3
MESSAGE_CHAR_LIMIT = 4096

# State
STATE_TTL_MINUTES = 30
STATE_MAX_CONCURRENT = 100
