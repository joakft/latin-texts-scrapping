#!/usr/bin/env bash
set -e

docker build -t app_template .
docker run --env-file .env app_template
