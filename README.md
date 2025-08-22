Sure! Here's the **entire `README.md`** for the **generated project** (i.e. inside `app_template/`), in one single code block — ready to copy-paste as `README.md`:

````markdown
# {{ project_name }}

This project was scaffolded using the [joakft/templates](https://github.com/joakft/templates) system.  
It is a reproducible, container-ready, developer-friendly Python app skeleton using:

- [`uv`](https://github.com/astral-sh/uv) for dependency & environment management
- `.env` files for safe secret handling
- `Makefile` for common workflows
- `Dockerfile` for containerization
- `pytest` for testing
- Standard `app/` and `tests/` structure

---

## 🚀 Quickstart

### 1. Install dependencies and prepare environment

```bash
chmod +x setup.sh run.sh docker-run.sh
./setup.sh
```

This will:
- Install Python dependencies with `uv`
- Copy `.env.example` → `.env` (if `.env` doesn't exist)
- Export `requirements.txt` for compatibility

### 2. Run the app

```bash
./run.sh
# OR
make run
```

You should see:

```
Hello, World!
```

---

## 🧪 Running Tests

This template uses `pytest`. Run tests with:

```bash
make test
```

You can also run:

```bash
uv run pytest tests/
```

---

## 🐳 Running with Docker

```bash
make docker
```

This builds the Docker image and runs the app inside it, using your `.env` file for config.

---

## 🧰 Makefile Shortcuts

| Command        | Description                          |
|----------------|--------------------------------------|
| `make setup`   | Sync deps, create `.env`, export reqs |
| `make run`     | Run app locally (`uv run`)            |
| `make test`    | Run tests with pytest                 |
| `make docker`  | Build and run Docker container        |
| `make export`  | Re-export `requirements.txt` from lock |
| `make clean`   | Remove caches, `.venv`, lock          |

---

## 📂 Project Structure

```
.
├── app/               # Application code
│   ├── __init__.py
│   ├── main.py        # Entry point
│   └── core.py        # Example logic
│
├── tests/             # Unit tests
│   └── test_core.py
│
├── .env.example       # Template for secrets
├── .gitignore         # Ignores env, cache, etc.
├── pyproject.toml     # uv-managed dependencies
├── uv.lock            # Lockfile for reproducibility
├── requirements.txt   # Exported from lock for compatibility
├── Makefile           # Dev commands
├── Dockerfile         # Container build
├── setup.sh           # Setup script
├── run.sh             # Run locally
└── docker-run.sh      # Run with Docker
```

---

## 🔐 Secrets

- App secrets and environment variables go in `.env` (excluded from Git)
- Always keep `.env.example` updated as reference

---

## 🧱 Built With

- [uv](https://github.com/astral-sh/uv) – fast, reliable dependency manager
- [python-dotenv](https://pypi.org/project/python-dotenv/) – for loading `.env`
- [pytest](https://docs.pytest.org/) – for testing
- Docker – for containerization
- Bash + Make – for developer ergonomics

---

## 📄 License

MIT License – free to use, fork, and adapt.

---

Generated using [joakft/templates](https://github.com/joakft/templates)
````
