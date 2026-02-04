# Telegram Chat History Wizard

A Telegram bot for semantic search through chat history exports with AI-powered question answering.

## Features

- **Semantic Search** - Find messages by meaning, not just keywords
- **AI Question Answering** - Ask questions in natural language by tagging the bot
- **Nickname Resolution** - Use nicknames/aliases to refer to users
- **Flip Detection** - Detect if someone changed their opinion on a topic
- **Follow-up Questions** - Reply to bot's answers for clarification
- **Context View** - See messages around a specific quote

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add:
- `TELEGRAM_BOT_TOKEN` - Get from [@BotFather](https://t.me/BotFather)
- `OPENAI_API_KEY` - Get from [OpenAI](https://platform.openai.com/api-keys)

### 3. Add Chat Export

1. In Telegram Desktop: Settings → Advanced → Export Telegram Data
2. Select the chat, choose JSON format
3. Extract the export to `chat_exports/` folder

### 4. Index Messages

```bash
python main.py index
```

### 5. Start Bot

```bash
python main.py bot
```

## Commands

### Search Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/search <query>` | Semantic search | `/search криптовалюти інвестиції` |
| `/quote <user> <topic>` | Find user's messages on topic | `/quote Женек біткоін` |
| `/flip <user> <topic>` | Check if user changed position | `/flip Женек крипта` |
| `/context <id>` | Show context around message | `/context 12345` |
| `/stats` | Database statistics | `/stats` |

### Alias Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/aliases` | List all nicknames | `/aliases` |
| `/alias <user> <nick>` | Add nickname | `/alias Женек буш` |
| `/alias_remove <nick>` | Remove nickname | `/alias_remove буш` |
| `/seed_aliases` | Load predefined nicknames | `/seed_aliases` |

### AI Questions

Tag the bot with a question using nicknames:

```
@dobby_the_free_trader_bot чи гусь казав що CCC погана людина?
```

The bot will:
1. Understand "гусь" refers to a specific user
2. Search chat history for relevant messages
3. Return formatted quotes with dates

**Follow-up**: Reply to bot's response to refine your search:
- "покажи ще" - show more results
- "а що він казав про біткоін?" - search for another topic

## CLI Commands

```bash
python main.py index     # Index chat exports
python main.py bot       # Run the bot
python main.py stats     # Show statistics
python main.py reindex   # Force full reindex
```

## Project Structure

```
telegram-chat-history-wizard/
├── main.py              # CLI entry point
├── config.py            # Configuration
├── indexer.py           # Message indexing
├── bot/
│   ├── handlers.py      # Telegram command handlers
│   ├── formatters.py    # Message formatting
│   └── conversation_context.py  # Follow-up tracking
├── db/
│   ├── models.py        # SQLAlchemy models
│   └── database.py      # Database operations
├── search/
│   ├── embeddings.py    # OpenAI embeddings
│   ├── vector_store.py  # ChromaDB wrapper
│   ├── flip_detector.py # Opinion change detection
│   └── question_parser.py # AI question understanding
├── ingestion/
│   ├── parser.py        # Telegram export parser
│   └── uploader.py      # ZIP upload handler
├── data/                # Runtime data (auto-created)
│   ├── metadata.db      # SQLite database
│   └── chroma/          # Vector embeddings
└── chat_exports/        # Place exports here
```

## First Run

After starting the bot for the first time:

1. Send `/seed_aliases` to load predefined nicknames
2. Send `/aliases` to verify they're loaded
3. Try: `@dobby_the_free_trader_bot що гусь казав про крипту?`

## Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full production deployment instructions using Supervisor.
