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

## Usage

### Ask Questions

Tag the bot with any question:

```
@bot_name чи гусь казав що порошенко поганий?
@bot_name що думав буш про крипту до 2022?
@bot_name коли серж міняв думку про біткоін?
```

The bot:
- Understands nicknames (гусь, буш, серж)
- Supports date filters (до 2022, після 2023)
- Provides AI-synthesized answers with quotes

**Follow-up**: Reply to bot's response to refine your search.

### Commands

| Command | Description |
|---------|-------------|
| `/stats` | Database statistics |
| `/aliases` | List nicknames |
| `/alias <user> <nick>` | Add nickname |
| `/seed_aliases` | Load predefined nicknames |
| `/context <id>` | Show message context |

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
3. Try: `@xxx що гусь казав про крипту?`

## Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full production deployment instructions using Supervisor.
