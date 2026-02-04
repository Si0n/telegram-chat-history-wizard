# Deployment Guide

Deploy the Telegram Chat History Wizard bot to a production server using Supervisor.

## Prerequisites

- Ubuntu/Debian server (20.04+ recommended)
- Python 3.10+
- Root or sudo access

## 1. Server Setup

### Install system dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv supervisor git
```

### Create bot user (optional but recommended)

```bash
sudo useradd -m -s /bin/bash botuser
sudo su - botuser
```

## 2. Deploy Application

### Clone or upload the project

```bash
cd /home/botuser
git clone <your-repo-url> telegram-chat-history-wizard
# OR upload via scp/rsync
```

### Create virtual environment

```bash
cd telegram-chat-history-wizard
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.example .env
nano .env
```

Add your credentials:
```
TELEGRAM_BOT_TOKEN=your_bot_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

### Set permissions

```bash
chmod 600 .env
```

## 3. Initial Setup

### Add chat exports

Upload your Telegram chat export to `chat_exports/` directory:

```bash
mkdir -p chat_exports
# Upload your ChatExport folder with result.json
```

### Index messages

```bash
source .venv/bin/activate
python main.py index
```

### Verify it works

```bash
python main.py stats
python main.py bot  # Test manually, Ctrl+C to stop
```

## 4. Supervisor Configuration

### Create supervisor config

```bash
sudo nano /etc/supervisor/conf.d/telegram-bot.conf
```

Add the following configuration:

```ini
[program:telegram-chat-bot]
command=/home/botuser/telegram-chat-history-wizard/.venv/bin/python main.py bot
directory=/home/botuser/telegram-chat-history-wizard
user=botuser
autostart=true
autorestart=true
startsecs=10
startretries=3
stderr_logfile=/var/log/telegram-bot/error.log
stdout_logfile=/var/log/telegram-bot/output.log
stderr_logfile_maxbytes=10MB
stdout_logfile_maxbytes=10MB
stderr_logfile_backups=5
stdout_logfile_backups=5
environment=PATH="/home/botuser/telegram-chat-history-wizard/.venv/bin"
```

### Create log directory

```bash
sudo mkdir -p /var/log/telegram-bot
sudo chown botuser:botuser /var/log/telegram-bot
```

### Load and start

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start telegram-chat-bot
```

## 5. Managing the Bot

### Check status

```bash
sudo supervisorctl status telegram-chat-bot
```

### View logs

```bash
# Real-time logs
sudo tail -f /var/log/telegram-bot/output.log
sudo tail -f /var/log/telegram-bot/error.log

# Or via supervisor
sudo supervisorctl tail -f telegram-chat-bot stderr
```

### Restart bot

```bash
sudo supervisorctl restart telegram-chat-bot
```

### Stop bot

```bash
sudo supervisorctl stop telegram-chat-bot
```

### Reload after code changes

```bash
cd /home/botuser/telegram-chat-history-wizard
git pull  # or upload new files
sudo supervisorctl restart telegram-chat-bot
```

## 6. First Run Checklist

After deployment, send these commands to your bot in Telegram:

1. `/seed_aliases` - Load predefined user nicknames
2. `/aliases` - Verify nicknames are loaded
3. `/stats` - Check database statistics
4. Test a question: `@dobby_the_free_trader_bot тест`

## 7. Updating Chat History

When you have new chat exports:

```bash
# Stop bot
sudo supervisorctl stop telegram-chat-bot

# Add new export to chat_exports/
# Then reindex
cd /home/botuser/telegram-chat-history-wizard
source .venv/bin/activate
python main.py index

# Restart bot
sudo supervisorctl start telegram-chat-bot
```

Or use the `/upload` command in Telegram to upload new exports directly.

## 8. Troubleshooting

### Bot not starting

```bash
# Check supervisor logs
sudo supervisorctl tail telegram-chat-bot stderr

# Check if another instance is running
ps aux | grep "python main.py bot"

# Verify environment
cd /home/botuser/telegram-chat-history-wizard
source .venv/bin/activate
python main.py bot  # Run manually to see errors
```

### Permission errors

```bash
# Fix ownership
sudo chown -R botuser:botuser /home/botuser/telegram-chat-history-wizard

# Fix .env permissions
chmod 600 .env
```

### Database locked

```bash
# Stop bot first
sudo supervisorctl stop telegram-chat-bot

# Then run indexing
python main.py index
```

### Out of memory (large exports)

The parser uses streaming, but embedding generation needs RAM. For very large exports:

```bash
# Increase swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 9. Security Recommendations

1. **Firewall**: Only allow SSH access
   ```bash
   sudo ufw allow ssh
   sudo ufw enable
   ```

2. **Keep .env secure**: Never commit to git
   ```bash
   chmod 600 .env
   ```

3. **Regular updates**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   pip install --upgrade -r requirements.txt
   ```

4. **Backup data**:
   ```bash
   # Backup database and vector store
   tar -czvf backup-$(date +%Y%m%d).tar.gz data/
   ```

## Directory Structure (Production)

```
/home/botuser/telegram-chat-history-wizard/
├── .env                 # Credentials (chmod 600)
├── .venv/               # Virtual environment
├── main.py
├── config.py
├── requirements.txt
├── data/
│   ├── metadata.db      # SQLite database
│   └── chroma/          # Vector embeddings
├── chat_exports/
│   └── ChatExport_*/    # Telegram exports
└── ...

/var/log/telegram-bot/
├── output.log           # Stdout logs
└── error.log            # Stderr logs

/etc/supervisor/conf.d/
└── telegram-bot.conf    # Supervisor config
```
