#!/bin/bash
set -e

SUPERVISOR_PROGRAM="chat-history-wizard"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Deploying Chat History Wizard ==="

cd "$DIR"

echo "Pulling latest changes..."
git pull

echo "Installing dependencies..."
.venv/bin/pip install -r requirements.txt --quiet

echo "Running tests..."
.venv/bin/python -m pytest tests/ -q
echo ""

echo "Restarting bot..."
sudo service supervisor restart

echo ""
echo "=== Deploy complete ==="
