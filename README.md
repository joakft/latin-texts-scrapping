````
# latin-texts-scrapping

A small toolkit to scrape Latin texts (TheLatinLibrary), index them with embeddings (HuggingFace → FAISS) and explore/search via a Gradio dashboard.

This repo was scaffolded from a template but is customized for:
- a scraper that saves `.txt` files into `data/`
- an indexing & analytics Gradio app at `app/indexing_app.py`
- a simple runner using `uv`

---

## TL;DR — Quick run

1. Install & prepare (using the repo scripts):
```bash
chmod +x setup.sh run.sh docker-run.sh
./setup.sh
````

Scrape (saves `.txt` files into `./data/`):

```
uv run python app/scraper_service.py
```

Start the Gradio indexing / stats UI:

```
uv run python app/indexing_app.py
```

Open the URL shown in the terminal (usually `http://127.0.0.1:7860`) and use the dashboard:

**Run Batch Ingestion**: embed new files from `data/` into FAISS

**Refresh Stats**: show indexed doc counts & lexical metrics

**Show Embedding Plot**: t-SNE / PCA visualization

**Semantic Search**: query the corpus (Latin or English)

**Reset Index / Debug Info**: utilities

---

## Project structure (what matters)

```
.
├── app/
│   ├── indexing_app.py       # Gradio UI: indexing, stats, plots, search
│   ├── scraper_service.py    # Scraper that saves .txt files to ../data/
│   └── ...                   # other app code
├── data/                     # Scraped .txt files (generated)
├── faiss.index               # FAISS vector index (generated after ingestion)
├── docs.json                 # Metadata for indexed chunks (generated)
├── run.sh                    # convenience runner (calls uv)
├── setup.sh                  # dependency bootstrap (uv + env)
├── pyproject.toml            # project dependencies (uv-managed)
├── requirements.txt          # exported requirements (for compatibility)
├── Makefile
├── Dockerfile
└── README.md                 # this file
```

---

## What this does / Features

**Scraper** (`app/scraper_service.py`)

Crawl TheLatinLibrary (or other sites you configure), clean the HTML and write `.txt` files to `data/`.

Respects polite delays and skips forbidden pages.

**Indexing & Persistence** (`app/indexing_app.py`)

Splits files into chunks (configurable chunk size / overlap).

Embeds chunks with a HuggingFace embedding model (default: `paraphrase-multilingual-MiniLM-L12-v2` for speed).

Stores vectors in FAISS and metadata in `docs.json`.

Supports incremental ingestion: already-indexed files are skipped; only new files are embedded.

File-level short-file filter (default `min_words = 500`) — prevents indexing index-pages / TOCs.

**Gradio dashboard**

Run ingestion with a slider to set `min_words`.

Live status logs (per-file): skipped / ok / progress.

Stats: #chunks, total words, average lexical diversity.

Embedding visualizer (t-SNE or PCA) with coloring by text length.

Semantic search box (returns chunk snippets + source file).

Reset index + debug info utilities.

---

## Usage details

### Scraping

Run the scraper (it will write `.txt` files under `./data/`):

```
uv run python app/scraper_service.py
```

Make sure `data/` is populated before indexing. If the scraper writes files to a different path, update `DATA_DIR` in `app/indexing_app.py`.

### Indexing / Ingestion (Gradio)

Start the dashboard:

```
uv run python app/indexing_app.py
```

Actions in the UI:

**Run Batch Ingestion** — reads `data/*.txt`, filters short files (file-level), splits into chunks, embeds and stores vectors.

The UI logs every file (`[SKIP]`, `[OK]`) so you can monitor progress.

Files already present in `docs.json` are skipped (idempotent ingestion).

**Min words** slider — skip files under that many words (default 500).

**Reset Index** — deletes `faiss.index` and `docs.json` and resets in-memory index.

**Show Debug Info** — prints paths and sizes (where to look for `faiss.index`, `docs.json`, and how many `.txt` the app sees).

### Semantic search

Type a query (Latin or English). The system returns top-k chunk snippets with:

file path

word count

distance score

Pro tip: search works best when files are chunked (the app does that) and when you pick a multilingual embedding model that matches your needs.

---

## Persistence & safe restart

Index vectors: `faiss.index` (FAISS binary in repo root)

Metadata: `docs.json` (JSON list of chunk-level metadata: `path`, `total_words`, `lexical_diversity`, etc.)

On startup `app/indexing_app.py` will try to load those files (if present). You do **not** need to re-run ingestion every time — ingestion is incremental and `docs.json` + `faiss.index` are re-used.

---

## Common commands

Reset index (terminal):

```
rm faiss.index docs.json
```

Or via Gradio UI: **Reset Index** button.

Run ingestion non-interactively (terminal):

```
uv run python -c "from app.indexing_app import index_service; index_service.batch_ingest(Path('data'), min_words=500)"
```

(Only run that if you understand your `DATA_DIR` layout; recommended to use the UI for logs.)

---

## Recommended dependency list (put in `pyproject.toml` or `requirements.txt`)

beautifulsoup4

requests

gradio>=4.0

numpy

scikit-learn

matplotlib

faiss-cpu

llama-index-core

llama-index-embeddings-huggingface

llama-index-vector-stores-faiss

sentence-transformers (if you use those models)

python-dotenv

If you prefer cloud embeddings: also add `openai` and configure `OPENAI_API_KEY` in `.env`.

---

## Best practices & troubleshooting

**If ingestion is slow**:

switch to a lighter embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) for CPU, or run on a machine with GPU and set `device="cuda"` in the embedding initialization.

increase `embed_batch_size` to 16–64 to process chunks in batches.

**If search returns no hits for English queries**: use a cross-lingual embedding model (e.g. `intfloat/multilingual-e5-*`) or translate the query to Latin before embedding.

**Files not seen by ingestion**: check `Show Debug Info` in the UI to confirm `DATA_DIR` and the number of `*.txt` files. Paths in logs show absolute locations.

**Partial progress visibility**: by default the app persists at the end of a batch. For safer long runs you can checkpoint every N files (the app can be configured to persist more frequently).

---

## .gitignore suggestions

Add the following to `.gitignore`:

```
# Scraped data and indexes
/data/
/faiss.index
/docs.json

# env & python venvs
.env
.venv/
__pycache__/
*.pyc
```

This keeps heavy generated files out of git.

---

## Roadmap (ideas)

Add metadata filtering UI (search by author, lexical\_diversity, avg\_sentence\_length).

Add RAG (chat-with-corpus) using retrieved chunks + LLM.

Improve scraper (follow nested links, better HTML cleaning; optional politeness / rate-limit config).

Add unit & integration tests for ingestion pipeline.

---

## License

MIT License — feel free to use, adapt, and contribute.

---