#!/usr/bin/env bash
set -e

echo ">>> Setting up project with uv..."
uv sync

if [ ! -f ".env" ]; then
    echo ">>> Creating .env from example (please edit with your real values)"
    cp .env.example .env
fi

echo ">>> Exporting requirements.txt..."
uv export --format requirements-txt > requirements.txt

echo ">>> Setup complete! Run './run.sh' or 'make run'."
