from __future__ import annotations

import gradio as gr
from pathlib import Path

from .scraper_service import ScraperService

# ---------- Defaults ----------
DEFAULT_URLS = "http://thelatinlibrary.com/"
OUT_DIR = Path("./data")
MAX_DEPTH = 3
MAX_PAGES = 300
DELAY = 2.0

service = ScraperService()

def _format_top_words(top_pairs):
    return "\n".join(f"{w}: {c}" for w, c in top_pairs)

def _format_top_files_md(items):
    """
    items: list of dicts {path, url, title, total_words, coverage}
    Render as markdown with clickable local-file links.
    """
    if not items:
        return "_No files yet meeting the threshold._"
    lines = []
    for d in items:
        pct = f"{100.0 * d['coverage']:.1f}%"
        # clickable local path (file://) and web URL
        local = d["path"]
        web = d["url"]
        title = d["title"] or Path(local).name
        lines.append(
            f"- **{title}** — {pct} of words in top-500, total={d['total_words']:,}  "
            f"[open file]({f'file://{local}'}) · [source]({web})"
        )
    return "\n".join(lines)

def start_crawl(urls_text: str, max_depth: int, max_pages: int, delay: float):
    # stop any previous run
    service.stop()
    svc = ScraperService()

    urls = [u.strip() for u in urls_text.replace("\n", ",").split(",") if u.strip()]

    for snap in svc.run(
        start_urls=urls,
        out_dir=OUT_DIR,
        max_depth=max_depth,
        max_pages=max_pages,
        delay=delay,
        include_host_folder=(len(urls) > 1),
        heartbeat_every=10,
        min_words_for_file=1000,    # your threshold
        top_files_k=10,             # show top 10
        top_vocab_n=500,            # rank by coverage of global top-500 words
    ):
        # prepare UI payloads
        top_words_txt = _format_top_words(snap.get("top_words", []))
        top_files_md = _format_top_files_md(snap.get("top_files", []))

        yield (
            f"{snap.get('site','')}",
            int(snap.get("visited", 0)),
            int(snap.get("saved", 0)),
            int(snap.get("words_total", 0)),
            int(snap.get("vocab_size", 0)),
            top_words_txt,
            top_files_md,
            snap.get("last_url", ""),
            snap.get("last_saved_path", ""),
            snap.get("note", ""),
        )

def stop_crawl():
    service.stop()
    return "Stopping…"

with gr.Blocks(title="Latin Library Scraper") as demo:
    gr.Markdown("## Latin Library Scraper — progress + live word stats")

    with gr.Row():
        urls = gr.Textbox(
            value=DEFAULT_URLS,
            label="Start URLs (comma or newline separated)",
            lines=2,
            placeholder="http://thelatinlibrary.com/\nhttps://othersite/",
        )
    with gr.Row():
        depth = gr.Slider(1, 6, value=MAX_DEPTH, step=1, label="Max Depth (BFS)")
        pages = gr.Slider(0, 2000, value=MAX_PAGES, step=50, label="Max Pages (0 = unlimited)")
        delay = gr.Slider(0.5, 5.0, value=DELAY, step=0.5, label="Delay between requests (sec)")

    with gr.Row():
        start_btn = gr.Button("Start Scraping", variant="primary")
        stop_btn = gr.Button("Stop")

    with gr.Row():
        site_out = gr.Textbox(label="Current Site", interactive=False)
        visited_out = gr.Number(label="Pages Visited", interactive=False)
        saved_out = gr.Number(label="Pages Saved", interactive=False)

    with gr.Row():
        words_total_out = gr.Number(label="Total Words", interactive=False)
        vocab_size_out = gr.Number(label="Distinct Words", interactive=False)

    with gr.Row():
        top_words_out = gr.Textbox(label="Top Words (live)", lines=12, interactive=False)
        top_files_out = gr.Markdown()  # <— the new widget with links

    last_url_out = gr.Textbox(label="Last URL", interactive=False)
    last_saved_out = gr.Textbox(label="Last Saved File", interactive=False)
    note_out = gr.Textbox(label="Note", interactive=False)

    start_btn.click(
        start_crawl,
        inputs=[urls, depth, pages, delay],
        outputs=[
            site_out,
            visited_out,
            saved_out,
            words_total_out,
            vocab_size_out,
            top_words_out,
            top_files_out,
            last_url_out,
            last_saved_out,
            note_out,
        ],
        api_name="start",
    )

    stop_btn.click(stop_crawl, outputs=[note_out])

if __name__ == "__main__":
    demo.launch()
