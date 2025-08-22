#!/usr/bin/env bash
set -e

# Load environment variables safely
if [ -f ".env" ]; then
    set -o allexport
    source .env
    set +o allexport
fi

# Run as a module (so "app" is recognized as a package)
uv run python -m app.main
