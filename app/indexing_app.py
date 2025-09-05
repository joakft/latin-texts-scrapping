from pathlib import Path
import json
import re
from collections import Counter

import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SimpleNodeParser
import faiss


# =========================
# PATHS (robust for uv run)
# =========================
# repo_root/
#   ├─ data/           (scraped .txt)
#   ├─ faiss.index
#   ├─ docs.json
#   └─ app/
#       └─ indexing_app.py
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR   = REPO_ROOT / "data"
INDEX_PATH = REPO_ROOT / "faiss.index"
META_PATH  = REPO_ROOT / "docs.json"


# =========================
# EMBEDDINGS & CHUNKING
# =========================
# Fast multilingual CPU model (good enough + quick)
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384

# Smaller chunks = faster per-chunk and better retrieval locality
parser = SimpleNodeParser.from_defaults(chunk_size=256, chunk_overlap=50)

# =========================
# METRICS
# =========================
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
SENT_RE = re.compile(r"[.!?;:]")

def latin_metrics(text: str):
    words = [w.lower() for w in WORD_RE.findall(text)]
    total = len(words)
    unique = len(set(words))
    lexical_div = unique / total if total else 0
    sentences = [s for s in SENT_RE.split(text) if s.strip()]
    num_sent = len(sentences)
    avg_sent_len = total / num_sent if num_sent else 0
    sent_lengths = [len(WORD_RE.findall(s)) for s in sentences]
    counter = Counter(words)
    hapax = sum(1 for w, c in counter.items() if c == 1)
    return {
        "total_words": total,
        "unique_words": unique,
        "lexical_diversity": lexical_div,
        "avg_sentence_length": avg_sent_len,
        "longest_sentence_length": max(sent_lengths) if sent_lengths else 0,
        "hapax_ratio": hapax / total if total else 0,
        "top_terms": counter.most_common(10),
    }


# =========================
# INDEX SERVICE
# =========================
class IndexService:
    def __init__(self, dim: int, emb_model: str,
                 index_path: Path = INDEX_PATH,
                 meta_path: Path = META_PATH):
        self.dim = dim
        self.emb_model = emb_model
        self.embedding_model = HuggingFaceEmbedding(
            model_name=emb_model,
            device="cpu",          # set "cuda" if you have GPU
            embed_batch_size=32    # try 16/32/64 on CPU
        )
        self.index_path = index_path
        self.meta_path = meta_path

        print(f"[INIT] repo_root     → {REPO_ROOT}")
        print(f"[INIT] data_dir      → {DATA_DIR} (exists={DATA_DIR.exists()})")
        print(f"[INIT] index_path    → {self.index_path}")
        print(f"[INIT] metadata_path → {self.meta_path}")

        # Load or create FAISS
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            print(f"[INIT] Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            self.index = faiss.IndexFlatL2(dim)
            print("[INIT] Created new FAISS index")

        self.vector_store = FaissVectorStore(self.index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.vector_index = VectorStoreIndex(
            [], storage_context=self.storage_context, embed_model=self.embedding_model
        )

        # Load metadata (chunk-level)
        self.docs = []
        if self.meta_path.exists() and self.meta_path.stat().st_size > 0:
            try:
                with open(self.meta_path, encoding="utf-8") as f:
                    metas = json.load(f)
                self.docs = [Document(text="", metadata=m) for m in metas]
                print(f"[INIT] Loaded {len(self.docs)} metadata entries")
            except json.JSONDecodeError:
                print("[WARN] docs.json corrupted; ignoring.")
        else:
            print("[INIT] No metadata found; starting fresh")

    # -------------
    # Ingestion
    # -------------
    def batch_ingest(self, folder: Path, min_words: int = 500):
        txts = list(folder.rglob("*.txt"))
        print(f"[INFO] Found {len(txts)} text files in {folder}")

        new_nodes = []
        skipped = 0
        already = 0
        ingested_files = 0

        for i, fp in enumerate(txts, start=1):
            path_str = str(fp)
            print(f"[PROGRESS] File {i}/{len(txts)} → {path_str}")

            # Skip if already ingested (by file path)
            if any(d.metadata.get("path") == path_str for d in self.docs):
                print(f"[SKIP] {path_str} → already ingested")
                already += 1
                continue

            # Read text
            try:
                text = fp.read_text(encoding="utf-8")
            except Exception as e:
                print(f"[ERROR] Could not read {path_str}: {e}")
                continue

            # FILE-level filter (not per chunk!)
            metrics = latin_metrics(text)
            total_words = metrics["total_words"]
            if total_words < min_words:
                print(f"[SKIP] {path_str} → only {total_words} words (< {min_words})")
                skipped += 1
                continue

            # Make a doc with file-level metrics
            doc = Document(text=text, metadata={"path": path_str, **metrics})

            # Split into chunks (nodes inherit file-level metadata)
            nodes = parser.get_nodes_from_documents([doc])
            for node in nodes:
                node.metadata.update(doc.metadata)

            # Insert all nodes (correct API for TextNodes)
            try:
                self.vector_index.insert_nodes(nodes)
                new_nodes.extend(nodes)
                ingested_files += 1
                print(f"[OK] {path_str} → {total_words} words, {len(nodes)} chunks embedded")
            except Exception as e:
                print(f"[ERROR] Could not insert {path_str}: {e}")


        if new_nodes:
            self.docs.extend(new_nodes)
            print(f"[INFO] Ingested {len(new_nodes)} chunks from {ingested_files} new files.")
        else:
            print(f"[INFO] No new docs ingested. {skipped} files skipped (<{min_words}), {already} already present.")

        self.persist()
        return f"Ingestion finished. Added {len(new_nodes)} chunks, skipped {skipped}, already {already}."

    # -------------
    # Persistence
    # -------------
    def persist(self):
        print(f"[DEBUG] Saving index with {self.index.ntotal} vectors and {len(self.docs)} docs...")
        faiss.write_index(self.index, str(self.index_path))

        safe_meta = []
        for d in self.docs:
            m = {}
            for k, v in d.metadata.items():
                try:
                    json.dumps(v)  # test serializable
                    m[k] = v
                except TypeError:
                    m[k] = str(v)  # fallback to string
            safe_meta.append(m)

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(safe_meta, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Persisted index → {self.index_path}, metadata → {self.meta_path}")


    # -------------
    # Utilities
    # -------------
    def reset_index(self):
        # Delete files & reset memory
        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()
        self.index = faiss.IndexFlatL2(self.dim)
        self.vector_store = FaissVectorStore(self.index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.vector_index = VectorStoreIndex([], storage_context=self.storage_context, embed_model=self.embedding_model)
        self.docs = []
        print("[INFO] Index & metadata reset.")

    def stats(self):
        total_words = sum(d.metadata.get("total_words", 0) for d in self.docs)
        avg_words = np.mean([d.metadata.get("total_words", 0) for d in self.docs]) if self.docs else 0
        lexical = np.mean([d.metadata.get("lexical_diversity", 0) for d in self.docs]) if self.docs else 0
        return {
            "docs": len(self.docs),
            "total_words": total_words,
            "avg_words": round(avg_words, 2),
            "avg_lexical_div": round(lexical, 3),
        }

    def embeddings_matrix(self):
        if self.index.ntotal == 0:
            return None
        return self.index.reconstruct_n(0, self.index.ntotal)

    def query(self, q: str, top_k: int = 5):
        q_emb = self.embedding_model.get_text_embedding(q)
        q_emb = np.array([q_emb]).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            if idx < len(self.docs):
                doc = self.docs[idx]
                results.append({
                    "path": doc.metadata.get("path", ""),
                    "title": Path(doc.metadata.get("path", "")).stem,
                    "words": doc.metadata.get("total_words", 0),
                    "snippet": (doc.text[:300] + "...") if doc.text else "",
                    "distance": float(dist),
                })
        return results


# =========================
# GRADIO APP
# =========================
index_service = IndexService(DIM, EMB_MODEL, INDEX_PATH, META_PATH)

def run_batch_ingest(min_words=500):
    return index_service.batch_ingest(DATA_DIR, min_words=min_words)

def show_stats():
    s = index_service.stats()
    return s["docs"], s["total_words"], s["avg_words"], s["avg_lexical_div"]

def plot_embeddings(method="tsne"):
    mat = index_service.embeddings_matrix()
    if mat is None:
        return None

    # align with docs length
    n = min(len(index_service.docs), mat.shape[0])
    mat = mat[:n]
    docs = index_service.docs[:n]

    if method == "pca":
        from sklearn.decomposition import PCA
        reduced = PCA(n_components=2).fit_transform(mat)
    else:
        reduced = TSNE(n_components=2, perplexity=30, init="random", random_state=42).fit_transform(mat)

    vals = [d.metadata.get("total_words", 0) for d in docs]
    fig, ax = plt.subplots()
    sc = ax.scatter(reduced[:, 0], reduced[:, 1], c=vals, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Total words")
    ax.set_title(f"Semantic Embedding Scatterplot ({method.upper()})")
    return fig

def semantic_search(query, top_k):
    results = index_service.query(query, top_k=top_k)
    if not results:
        return "No results."
    lines = []
    for r in results:
        file_link = f"file://{r['path']}"
        title = r.get("title") or Path(r["path"]).stem
        lines.append(
            f"**{title}** ({r['words']} words, dist={r['distance']:.3f})  \n"
            f"{r['snippet']}  \n"
            f"[open file]({file_link})"
        )
    return "\n\n".join(lines)

def reset_index():
    index_service.reset_index()
    return "Index & metadata reset."

def debug_info():
    txt_files = list(DATA_DIR.rglob("*.txt"))
    return (
        str(REPO_ROOT),
        str(DATA_DIR),
        str(INDEX_PATH), INDEX_PATH.exists(), INDEX_PATH.stat().st_size if INDEX_PATH.exists() else 0,
        str(META_PATH),  META_PATH.exists(),  META_PATH.stat().st_size  if META_PATH.exists()  else 0,
        len(txt_files)
    )

with gr.Blocks(title="Indexing & Embedding Stats") as demo:
    gr.Markdown("## Batch Indexing of Scraped Latin Texts")

    with gr.Row():
        ingest_btn = gr.Button("Run Batch Ingestion")
        min_words = gr.Slider(0, 2000, value=500, step=50, label="Min words per FILE")
    ingest_out = gr.Textbox(label="Ingestion Log", lines=4)
    ingest_btn.click(run_batch_ingest, inputs=[min_words], outputs=[ingest_out])

    with gr.Row():
        stats_btn = gr.Button("Refresh Stats")
        docs_out = gr.Number(label="#Docs/Chunks")
        total_words_out = gr.Number(label="Total Words (sum over files)")
        avg_words_out = gr.Number(label="Avg Words/Doc (file-level)")
        lex_div_out = gr.Number(label="Avg Lexical Diversity (file-level)")
    stats_btn.click(show_stats, outputs=[docs_out, total_words_out, avg_words_out, lex_div_out])

    with gr.Row():
        method_dd = gr.Dropdown(choices=["tsne", "pca"], value="tsne", label="Dim reduction")
        plot_btn = gr.Button("Show Embedding Plot")
    plot_out = gr.Plot()
    plot_btn.click(plot_embeddings, inputs=[method_dd], outputs=[plot_out])

    gr.Markdown("### 🔍 Semantic Search")
    with gr.Row():
        query_box = gr.Textbox(label="Enter query (Latin or English)", lines=1)
        k_slider = gr.Slider(1, 10, value=5, step=1, label="Top K")
    search_btn = gr.Button("Search")
    results_out = gr.Markdown()
    search_btn.click(semantic_search, inputs=[query_box, k_slider], outputs=[results_out])

    gr.Markdown("### 🧰 Utilities")
    with gr.Row():
        reset_btn = gr.Button("Reset Index")
        debug_btn = gr.Button("Show Debug Info")
    reset_out = gr.Textbox(label="Reset Log")
    debug_out = gr.Textbox(label="Debug", lines=6)
    reset_btn.click(reset_index, outputs=[reset_out])
    debug_btn.click(debug_info, outputs=[debug_out])

if __name__ == "__main__":
    demo.launch()
