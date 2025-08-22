from __future__ import annotations

import re
import time
from collections import Counter, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

# ---------- Simple, self-contained crawler core with streaming updates ----------

PARSER = "html.parser"
REQUEST_TIMEOUT = 25
RETRIES = 3

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) "
        "Gecko/20100101 Firefox/128.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

EXCLUDE_HREF_PATTERNS = [
    r"\.css$", r"\.js$", r"\.jpg$", r"\.jpeg$", r"\.png$", r"\.gif$", r"\.pdf$",
    r"\.zip$", r"\.mp3$", r"\.mp4$", r"\.svg$", r"^mailto:",
]
EXCLUDE_RE = re.compile("|".join(EXCLUDE_HREF_PATTERNS), re.IGNORECASE)
HTML_EXT_RE = re.compile(r"\.(s?html?)$", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+", re.UNICODE)

def safe_filename(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^-\w.]+", "", s)
    return s[:120] or "untitled"

def _normalize_base(url: str) -> str:
    p = urlparse(url.strip())
    scheme = p.scheme or "http"
    netloc = p.netloc or p.path
    path = p.path if p.netloc else "/"
    if not path.endswith("/"):
        path += "/"
    return urlunparse((scheme, netloc, path, "", "", ""))

def _alt_bases(base: str) -> Iterable[str]:
    u = urlparse(base)
    host = u.netloc
    host_wo_www = host[4:] if host.startswith("www.") else host
    candidates = [
        ("http", host_wo_www),
        ("http", "www." + host_wo_www),
        ("https", host_wo_www),
        ("https", "www." + host_wo_www),
    ]
    seen = set()
    for scheme, netloc in candidates:
        alt = urlunparse((scheme, netloc, "/", "", "", ""))
        if alt not in seen:
            seen.add(alt)
            yield alt

def _strip_fragment(u: str) -> str:
    p = urlparse(u)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, p.query, ""))

def is_same_host(a: str, b: str) -> bool:
    return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()

def is_html_like(url: str) -> bool:
    path = urlparse(url).path
    return (
        path.endswith("/")
        or bool(HTML_EXT_RE.search(path))
        or not ("." in path.split("/")[-1])  # extension-less endpoint
    )

@contextmanager
def make_session():
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    try:
        yield s
    finally:
        s.close()

def get_soup(url: str, session: requests.Session, *, retries: int = RETRIES, delay: float = 2.0) -> Optional[BeautifulSoup]:
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code in (401, 403, 404, 429):
                return None
            resp.raise_for_status()
            return BeautifulSoup(resp.text, PARSER)
        except Exception:
            if attempt == retries:
                return None
            time.sleep(delay * attempt)
    return None

def clean_text(soup: BeautifulSoup) -> str:
    for selector in ["div.text", "div#content", "div.content", "body"]:
        node = soup.select_one(selector)
        if node:
            break
    else:
        node = soup
    for sel in ["nav", "header", "footer", ".nav", ".header", ".footer", "script", "style"]:
        for tag in node.select(sel):
            tag.decompose()
    text = node.get_text(separator=" ", strip=True)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def infer_titles(soup: BeautifulSoup, url: str) -> Tuple[str, str]:
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else urlparse(url).path.rsplit("/", 1)[-1]
    title = re.sub(r"\s*–\s*The Latin Library\s*$", "", title)
    if " - " in title:
        author, work = title.split(" - ", 1)
    else:
        author, work = ("Unknown", title)
    return (author.strip() or "Unknown", work.strip() or "Untitled")

def tokenize_count(text: str) -> Counter:
    words = [w.lower() for w in WORD_RE.findall(text)]
    return Counter(words)

class ScraperService:
    """
    Streamed BFS crawler that yields progress dicts for a UI to consume.
    Progress dict includes:
      - 'top_files': list of {path, url, total_words, coverage, title}
    """
    def __init__(self):
        self._stop = False

    def stop(self):
        self._stop = True

    def run(
        self,
        start_urls: List[str],
        out_dir: Path,
        max_depth: int = 3,
        max_pages: int = 300,
        delay: float = 2.0,
        include_host_folder: bool = True,
        heartbeat_every: int = 10,
        min_words_for_file: int = 1000,
        top_files_k: int = 10,
        top_vocab_n: int = 500,
    ) -> Generator[Dict, None, None]:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Global stats
        global_vocab = Counter()
        words_total = 0

        # Per-file stats: path -> dict(counter, total, url, title)
        file_stats: Dict[str, Dict] = {}

        for start_url in start_urls:
            if self._stop:
                break

            base = _normalize_base(start_url)
            host = urlparse(base).netloc
            site_out_root = out_dir / (safe_filename(host) if include_host_folder and len(start_urls) > 1 else "")
            site_out_root.mkdir(parents=True, exist_ok=True)

            with make_session() as session:
                working_base = None
                for candidate in _alt_bases(base):
                    session.headers["Referer"] = candidate
                    soup = get_soup(candidate, session, delay=delay)
                    if soup:
                        working_base = candidate
                        break
                    time.sleep(delay)

                if not working_base:
                    yield self._snapshot(host, 0, 0, 0, 0, global_vocab, words_total, note="homepage unavailable")
                    continue

                start = _strip_fragment(working_base)
                base_netloc = urlparse(start).netloc

                visited: set[str] = set()
                q = deque([(start, 0)])
                parent_map: Dict[str, Tuple[Optional[str], Optional[str]]] = {start: (None, None)}

                visited_pages = 0
                saved_pages = 0

                while q and not self._stop:
                    url, depth = q.popleft()
                    url = _strip_fragment(url)
                    if url in visited:
                        continue
                    visited.add(url)

                    session.headers["Referer"] = parent_map.get(url, (None, None))[0] or working_base
                    soup = get_soup(url, session, delay=delay)
                    if soup is None:
                        if (visited_pages + len(visited)) % heartbeat_every == 0:
                            yield self._snapshot(host, visited_pages, saved_pages, depth, len(q), global_vocab, words_total)
                        time.sleep(delay)
                        continue

                    if self._is_html_like(url):
                        visited_pages += 1

                        # Disk path via anchor-chain mirroring
                        chain = self._anchor_chain(url, parent_map)
                        if chain:
                            *dirs, leaf = chain
                        else:
                            dirs, leaf = [], self._html_stem(url)

                        target_dir = site_out_root
                        for seg in dirs:
                            target_dir = target_dir / f"{seg}_html"
                        target_dir.mkdir(parents=True, exist_ok=True)

                        author, work = infer_titles(soup, url)
                        text = clean_text(soup)
                        saved_path_str = ""
                        if text and len(text) >= 80:
                            c = tokenize_count(text)
                            global_vocab.update(c)
                            words_total += sum(c.values())

                            saved_path = target_dir / f"{safe_filename(leaf)}_html.txt"
                            header = f"# {work}\n# Author: {author}\n# Source: {url}\n\n"
                            saved_path.write_text(header + text, encoding="utf-8")
                            saved_path_str = str(saved_path.resolve())
                            saved_pages += 1

                            # store per-file stats
                            file_stats[saved_path_str] = {
                                "counter": c,
                                "total": sum(c.values()),
                                "url": url,
                                "title": work,
                            }

                        # heartbeat with top_files
                        if (visited_pages % heartbeat_every) == 0:
                            top_files = self._rank_top_files(file_stats, global_vocab, top_vocab_n, min_words_for_file, top_files_k)
                            yield self._snapshot(
                                host, visited_pages, saved_pages, depth, len(q),
                                global_vocab, words_total,
                                last_url=url, last_saved_path=saved_path_str,
                                top_files=top_files
                            )

                        if max_pages and visited_pages >= max_pages:
                            top_files = self._rank_top_files(file_stats, global_vocab, top_vocab_n, min_words_for_file, top_files_k)
                            yield self._snapshot(
                                host, visited_pages, saved_pages, depth, len(q),
                                global_vocab, words_total,
                                last_url=url, last_saved_path=saved_path_str,
                                top_files=top_files, note="reached page cap"
                            )
                            break

                    # enqueue children
                    if depth < max_depth:
                        for a in soup.select("a[href]"):
                            href = (a.get("href") or "").strip()
                            if not href or EXCLUDE_RE.search(href):
                                continue
                            nxt = _strip_fragment(urljoin(url, href))
                            if urlparse(nxt).netloc.lower() != base_netloc.lower():
                                continue
                            if nxt not in parent_map:
                                anchor_text = (a.get_text(separator=" ", strip=True) or "").strip()
                                parent_map[nxt] = (url, anchor_text if anchor_text else None)
                                q.append((nxt, depth + 1))

                    time.sleep(delay)

                # final snapshot for the site
                top_files = self._rank_top_files(file_stats, global_vocab, top_vocab_n, min_words_for_file, top_files_k)
                yield self._snapshot(host, visited_pages, saved_pages, 0, 0, global_vocab, words_total, top_files=top_files, note="site done")

    # ------- helpers -------
    def _snapshot(
        self, site: str, visited: int, saved: int, depth: int, queue: int,
        vocab: Counter, words_total: int, *,
        last_url: str = "", last_saved_path: str = "", top_files: List[Dict] | None = None, note: str = ""
    ) -> Dict:
        return {
            "site": site,
            "visited": visited,
            "saved": saved,
            "depth": depth,
            "queue": queue,
            "words_total": words_total,
            "vocab_size": len(vocab),
            "top_words": vocab.most_common(20),
            "last_url": last_url,
            "last_saved_path": last_saved_path,
            "top_files": top_files or [],
            "note": note,
        }

    def _rank_top_files(
        self,
        file_stats: Dict[str, Dict],
        global_vocab: Counter,
        top_vocab_n: int,
        min_words: int,
        top_k: int,
    ) -> List[Dict]:
        if not file_stats:
            return []

        top_vocab = {w for w, _ in global_vocab.most_common(top_vocab_n)} if top_vocab_n > 0 else set()

        scored = []
        for path, st in file_stats.items():
            total = st["total"]
            if total < min_words:
                continue
            c: Counter = st["counter"]
            in_top = sum(c[w] for w in top_vocab) if top_vocab else 0
            coverage = (in_top / total) if total else 0.0
            scored.append({
                "path": path,
                "url": st["url"],
                "title": st.get("title") or "",
                "total_words": total,
                "coverage": coverage,
            })

        scored.sort(key=lambda d: d["coverage"], reverse=True)
        return scored[:top_k]

    def _html_stem(self, page_url: str) -> str:
        p = urlparse(page_url).path
        if p.endswith("/") or p == "":
            return "index"
        name = p.rsplit("/", 1)[-1]
        name = re.sub(r"\.(s?html?)$", "", name, flags=re.IGNORECASE)
        return name or "index"

    def _anchor_chain(self, url: str, parent_map: Dict[str, Tuple[Optional[str], Optional[str]]]) -> List[str]:
        chain: List[str] = []
        cur = url
        while True:
            parent, anchor = parent_map.get(cur, (None, None))
            if parent is None:
                break
            seg = (anchor or self._html_stem(cur)).strip()
            seg = safe_filename(seg) or "page"
            chain.append(seg)
            cur = parent
        chain.reverse()
        return chain

    def _is_html_like(self, url: str) -> bool:
        return is_html_like(url)
