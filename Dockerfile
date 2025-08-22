FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen

COPY app ./app
COPY .env.example .env.example

CMD ["uv", "run", "python", "-m", "app.main"]
