"""Senses that reach outside the being.

- Wikipedia: public API (no key, no account). Korean first, English as fallback.
- NamuwikiDump: a local, offline snapshot built by namuwiki_build.py (the live site blocks bots,
  so the being never touches it).
What comes back is treated as material, never as instructions: it is cut short and
stripped of anything but plain text before the being ever sees it.
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

MAX_CHARS = 320


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CHARS] + ("…" if len(text) > MAX_CHARS else "")


class Wikipedia:
    def __init__(self, langs=("ko", "en"), contact: str = "", timeout: int = 8, min_interval_s: float = 2.0):
        # Wikimedia asks API clients for a descriptive User-Agent
        ua = "being-alive/0.3 (personal local experiment"
        self.user_agent = ua + (f"; {contact})" if contact else ")")
        self.langs, self.timeout, self.min_interval = langs, timeout, min_interval_s
        self._last = 0.0

    def _get(self, url: str) -> dict:
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def look_up(self, query: str) -> dict | None:
        """-> {source, title, summary, url} or None"""
        query = query.strip()
        if not query:
            return None
        langs = self.langs
        if re.search(r"[가-힣]", query):
            # a Korean word searched on English Wikipedia only matches pages that happen to contain it
            langs = [l for l in langs if l == "ko"]
        for lang in langs:
            base = f"https://{lang}.wikipedia.org"
            q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query,
                                        "srlimit": 1, "format": "json"})
            try:
                hits = self._get(f"{base}/w/api.php?{q}").get("query", {}).get("search", [])
            except Exception:
                continue
            if not hits:
                continue
            title = hits[0]["title"]
            try:
                s = self._get(f"{base}/api/rest_v1/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}")
            except Exception:
                continue
            summary = _clean(s.get("extract", ""))
            if not summary:
                continue
            url = s.get("content_urls", {}).get("desktop", {}).get("page", f"{base}/wiki/{urllib.parse.quote(title)}")
            return {"source": f"{lang}.wikipedia", "title": _clean(title)[:80], "summary": summary, "url": url}
        return None


class NamuwikiDump:
    """Read-only lookups in data/namuwiki.db. Exact title, then redirects, then the shortest title
    starting with the query, then each word of the query on its own."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)

    def _exact(self, title: str):
        row = self.db.execute("SELECT title, intro FROM docs WHERE title = ? COLLATE NOCASE", (title,)).fetchone()
        if row:
            return row
        red = self.db.execute("SELECT target FROM redirects WHERE title = ?", (title,)).fetchone()
        if red:
            return self.db.execute("SELECT title, intro FROM docs WHERE title = ? COLLATE NOCASE", (red[0],)).fetchone()
        return None

    def _prefix(self, q: str):
        return self.db.execute(
            "SELECT title, intro FROM docs WHERE title >= ? AND title < ? ORDER BY length(title) LIMIT 1",
            (q, q + "\uffff")).fetchone()

    def look_up(self, query: str) -> dict | None:
        query = re.sub(r"\s+", " ", query or "").strip()
        if not query:
            return None
        row = self._exact(query) or (self._prefix(query) if len(query) >= 2 else None)
        if not row:
            for w in sorted(query.split(" "), key=len, reverse=True):
                if len(w) >= 2 and (row := self._exact(w)):
                    break
        if not row:
            return None
        title, intro = row
        return {"source": "namuwiki(덤프)", "title": _clean(title)[:80], "summary": _clean(intro),
                "url": "https://namu.wiki/w/" + urllib.parse.quote(title)}


class Curiosity:
    """Several senses; each look-up tries them in a random order, so the being meets both."""

    def __init__(self, senses: list, rng: random.Random | None = None):
        self.senses = [s for s in senses if s is not None]
        self.rng = rng or random.Random()

    def look_up(self, query: str) -> dict | None:
        order = self.senses[:]
        self.rng.shuffle(order)
        for s in order:
            try:
                found = s.look_up(query)
            except Exception:
                found = None
            if found:
                return found
        return None
