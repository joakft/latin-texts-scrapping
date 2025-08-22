from __future__ import annotations

import re
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

# ===================== CONFIG (edit here) =====================
OUT_DIR = Path("./data").resolve()

# You can list multiple sites to crawl (they must be same-domain BFS each).
START_URLS = [
    "http://thelatinlibrary.com/",  # friendlier over http; you can add more sites here
    # "https://example.com/",
]

DELAY = 2.0                # polite delay between requests
MAX_DEPTH = 3              # BFS depth from each homepage
MAX_PAGES = 400            # cap per site (0 = unlimited)
REQUEST_TIMEOUT = 25       # per request
RETRIES = 3                # per URL retries
HEARTBEAT_EVERY = 25       # print stats every N saved/visited pages
DEBUG_PRINT_URLS = False   # True = print every fetched URL

PARSER = "html.parser"

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

# Exclude non-content targets
EXCLUDE_HREF_PATTERNS = [
    r"\.css$", r"\.js$", r"\.jpg$", r"\.jpeg$", r"\.png$", r"\.gif$", r"\.pdf$",
    r"\.zip$", r"\.mp3$", r"\.mp4$", r"\.svg$", r"^mailto:",
    r"^https?://(?!.*thelatinlibrary\.com)",  # external (keep same-domain only for this site)
]
EXCLUDE_RE = re.compile("|".join(EXCLUDE_HREF_PATTERNS), re.IGNORECASE)

HTML_EXT_RE = re.compile(r"\.(s?html?)$", re.IGNORECASE)  # .html, .htm, .shtml

# If crawling multiple sites, create a subfolder per host to avoid collisions
INCLUDE_HOST_FOLDER = True if len(START_URLS) > 1 else False
# =============================================================


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


def _alt_bases(base: str):
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


@contextmanager
def make_session():
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    try:
        yield s
    finally:
        s.close()


def get_soup(url: str, session: requests.Session, *, retries: int = RETRIES, delay: float = DELAY) -> BeautifulSoup | None:
    """
    Fetch URL; skip forbidden (401/403/429) and 404 cleanly; retry on transient errors.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code in (401, 403, 429, 404):
                if DEBUG_PRINT_URLS:
                    print(f"[SKIP {resp.status_code}] {url}")
                return None
            resp.raise_for_status()
            return BeautifulSoup(resp.text, PARSER)
        except Exception as e:
            if attempt == retries:
                print(f"[WARN] Failed {url}: {e}")
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
    for sel in ["nav", "header", "footer", ".nav", ".header", ".footer"]:
        for tag in node.select(sel):
            tag.decompose()
    for tag in node.find_all(["script", "style"]):
        tag.decompose()
    text = node.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_titles(soup: BeautifulSoup, url: str) -> tuple[str, str]:
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else urlparse(url).path.rsplit("/", 1)[-1]
    title = re.sub(r"\s*–\s*The Latin Library\s*$", "", title)
    if " - " in title:
        author, work = title.split(" - ", 1)
    else:
        author, work = ("Unknown", title)
    return (author.strip() or "Unknown", work.strip() or "Untitled")


def html_stem_from_url(page_url: str) -> str:
    p = urlparse(page_url).path
    if p.endswith("/") or p == "":
        return "index"
    name = p.rsplit("/", 1)[-1]
    name = re.sub(r"\.(s?html?)$", "", name, flags=re.IGNORECASE)
    return name or "index"


def anchor_chain_path(url: str, parent_map: dict[str, tuple[str | None, str | None]]) -> list[str]:
    """
    Build the chain of anchor texts from the homepage to `url`.
    Each segment is sanitized and suffixed with `_html` when turned into a folder/file.
    Fallback to the page's stem when anchor text is missing.
    """
    chain: list[str] = []
    cur = url
    while True:
        entry = parent_map.get(cur)
        if not entry:
            break
        parent, anchor = entry
        # Use anchor text if present; else fallback to URL stem
        seg = anchor.strip() if anchor else html_stem_from_url(cur)
        seg = safe_filename(seg) or "page"
        chain.append(seg)
        if parent is None:
            break
        cur = parent
    chain.reverse()  # from root -> leaf
    return chain


def is_html_like(url: str) -> bool:
    path = urlparse(url).path
    return (
        path.endswith("/") or
        bool(HTML_EXT_RE.search(path)) or
        not ("." in path.split("/")[-1])  # extension-less endpoint
    )


def crawl_one_site(start_url: str):
    base = _normalize_base(start_url)
    host = urlparse(base).netloc
    out_root = OUT_DIR / (safe_filename(host) if INCLUDE_HOST_FOLDER else "")
    out_root.mkdir(parents=True, exist_ok=True)

    with make_session() as session:
        # Find a working homepage variant quickly (rotate http/https, www/non-www)
        working_base = None
        for candidate in _alt_bases(base):
            session.headers["Referer"] = candidate  # light referer
            print(f"[INFO] Trying homepage: {candidate}")
            soup = get_soup(candidate, session)
            if soup:
                working_base = candidate
                break
            time.sleep(DELAY)
        if not working_base:
            print(f"[ERROR] Homepage unavailable for {base}")
            return

        start = _strip_fragment(working_base)
        base_netloc = urlparse(start).netloc

        visited: set[str] = set()
        q = deque([(start, 0)])

        # Remember how we reached each URL: child -> (parent, anchor_text)
        # Root has (None, None)
        parent_map: dict[str, tuple[str | None, str | None]] = {start: (None, None)}

        pages_visited = 0
        pages_saved = 0
        t0 = time.time()

        print(f"[INFO] START BFS host={host} depth≤{MAX_DEPTH} max_pages={MAX_PAGES or '∞'}")
        while q:
            url, depth = q.popleft()
            url = _strip_fragment(url)
            if url in visited:
                continue
            visited.add(url)

            if DEBUG_PRINT_URLS:
                print(f"[FETCH][d={depth}] {url}")

            session.headers["Referer"] = parent_map.get(url, (None, None))[0] or working_base
            soup = get_soup(url, session)
            if soup is None:
                continue

            if is_html_like(url):
                pages_visited += 1

                # Build nested path from anchor chain
                chain = anchor_chain_path(url, parent_map)  # ['Y', 'X'] -> Y_html/X_html.txt
                if chain:
                    *dirs, leaf = chain
                else:
                    dirs, leaf = [], html_stem_from_url(url)

                # Make dirs: each with _html suffix
                target_dir = out_root
                for seg in dirs:
                    target_dir = target_dir / f"{seg}_html"
                target_dir.mkdir(parents=True, exist_ok=True)

                # Save file as <leaf>_html.txt
                author, work = infer_titles(soup, url)
                text = clean_text(soup)
                if text and len(text) >= 80:
                    out_path = target_dir / f"{safe_filename(leaf)}_html.txt"
                    header = f"# {work}\n# Author: {author}\n# Source: {url}\n\n"
                    out_path.write_text(header + text, encoding="utf-8")
                    pages_saved += 1

                if pages_visited % HEARTBEAT_EVERY == 0:
                    elapsed = time.time() - t0
                    rate = pages_visited / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[HB][{host}] visited={pages_visited} saved={pages_saved} "
                        f"depth={depth} queue={len(q)} rate={rate:.2f} pgs/s"
                    )

                if MAX_PAGES and pages_visited >= MAX_PAGES:
                    print(f"[INFO][{host}] Reached MAX_PAGES cap. Stopping this site.")
                    break

            # Discover links if within depth
            if depth < MAX_DEPTH:
                for a in soup.select("a[href]"):
                    href = (a.get("href") or "").strip()
                    if not href or EXCLUDE_RE.search(href):
                        continue
                    nxt = _strip_fragment(urljoin(url, href))
                    p = urlparse(nxt)
                    if p.netloc.lower() != base_netloc.lower():
                        continue
                    if nxt not in parent_map:
                        anchor_text = (a.get_text(separator=" ", strip=True) or "").strip()
                        parent_map[nxt] = (url, anchor_text if anchor_text else None)
                        q.append((nxt, depth + 1))

            time.sleep(DELAY)

        elapsed = time.time() - t0
        print(f"[INFO] DONE host={host} visited={pages_visited} saved={pages_saved} in {elapsed:.1f}s "
              f"({pages_visited/elapsed if elapsed>0 else 0:.2f} pgs/s)")


def crawl_all():
    print(f"[INFO] Output directory: {OUT_DIR}")
    print(f"[INFO] Delay: {DELAY}s | Depth: {MAX_DEPTH} | Max pages/site: {MAX_PAGES or '∞'}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for start in START_URLS:
        crawl_one_site(start)


if __name__ == "__main__":
    crawl_all()
