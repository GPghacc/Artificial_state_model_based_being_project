"""Build a local, offline Namuwiki index for the being's curiosity.

    pip install pyarrow
    python namuwiki_build.py

1. downloads the public Namuwiki dump (Hugging Face: heegyu/namuwiki, ~3GB, CC BY-NC-SA 2.0 KR)
   into data/ (resumes if interrupted), unless --parquet points at a file you already have
2. keeps only each article's opening (about 320 characters, markup stripped) and redirects
3. writes data/namuwiki.db (SQLite). The parquet can be deleted afterwards.

Namuwiki itself blocks automated access, so the being never touches the live site;
it only reads this snapshot. The snapshot is from around 2021-2022, so it has no recent events.
License: CC BY-NC-SA 2.0 KR. Personal, non-commercial use, with attribution (the page links each entry).
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "https://huggingface.co/datasets/heegyu/namuwiki/resolve/main/namuwiki_20210301.parquet"
INTRO_CHARS = 320
SCAN_CHARS = 8000        # only the start of each article is needed for its opening


# ------------------------------------------------------------------ markup → plain opening
_drop_line = re.compile(r"^\s*(\|\||\[include|\[목차\]|\[tableofcontents\]|#redirect|##|\[\[분류:|\[\[파일:|"
                        r"\{\{\{#!|\}\}\}|>|\[br\]$|-{4,})", re.I)
_heading = re.compile(r"^\s*=+#?\s.*\s#?=+\s*$")


def clean_intro(text: str) -> str:
    text = text[:SCAN_CHARS]
    text = re.sub(r"\{\{\{#!(wiki|html|folding|syntax)[\s\S]*?\}\}\}", " ", text)   # embedded blocks
    text = re.sub(r"\[\*[^\]]*\]", "", text)                                          # footnotes
    lines, started = [], False
    for line in text.splitlines():
        if _heading.match(line):
            if started:
                break                      # the opening ends at the first heading after some prose
            continue
        if not line.strip() or _drop_line.match(line):
            continue
        lines.append(line)
        started = True
        if sum(len(l) for l in lines) > INTRO_CHARS * 2:
            break
    s = " ".join(lines)
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)                           # [[a|b]] -> b, [[a]] -> a
    s = re.sub(r"\{\{\{[+-]?\d\s*", "", s)                                             # {{{+1 x}}}
    s = re.sub(r"\{\{\{#\w+\s*", "", s)                                                # {{{#color x}}}
    s = s.replace("{{{", "").replace("}}}", "")
    s = re.sub(r"'''?|~~|--|__|\^\^|,,", "", s)
    s = re.sub(r"\[(?:br|age|date|dday|youtube|kakaotv|nicovideo|ruby|anchor)\([^)]*\)\]|\[br\]", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:INTRO_CHARS] + ("…" if len(s) > INTRO_CHARS else "")


def redirect_target(text: str) -> str | None:
    m = re.match(r"\s*#redirect\s+(.+)", text, re.I)
    if not m:
        return None
    return m.group(1).split("#")[0].strip() or None


# ------------------------------------------------------------------ download
def download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(URL, headers={"User-Agent": "being-alive/0.3 (personal local experiment)"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:           # server ignored the range: start over
            have = 0
        total = have + int(r.headers.get("Content-Length", 0) or 0)
        mode = "ab" if have else "wb"
        with open(dest, mode) as f:
            got, t0 = have, time.time()
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    rate = (got - have) / max(1e-6, time.time() - t0) / 1e6
                    print(f"\r  내려받는 중 {got / 1e9:.2f} / {total / 1e9:.2f} GB  ({rate:.1f} MB/s)", end="", flush=True)
    print()


# ------------------------------------------------------------------ build
def build(parquet: Path, db_path: Path) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow가 필요해. 먼저 'pip install pyarrow'를 실행해줘.")

    tmp = db_path.with_suffix(".building")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    db.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE docs(title TEXT PRIMARY KEY, intro TEXT NOT NULL);
        CREATE TABLE redirects(title TEXT PRIMARY KEY, target TEXT NOT NULL);
    """)
    f = pq.ParquetFile(parquet)
    total = f.metadata.num_rows
    done = kept = redirs = 0
    t0 = time.time()
    for batch in f.iter_batches(batch_size=2000, columns=["title", "text"]):
        titles = batch.column("title").to_pylist()
        texts = batch.column("text").to_pylist()
        docs, reds = [], []
        for title, text in zip(titles, texts):
            if not title or not text:
                continue
            tgt = redirect_target(text)
            if tgt:
                reds.append((title, tgt))
                continue
            intro = clean_intro(text)
            if len(intro) >= 15:
                docs.append((title, intro))
        db.executemany("INSERT OR IGNORE INTO docs VALUES(?,?)", docs)
        db.executemany("INSERT OR IGNORE INTO redirects VALUES(?,?)", reds)
        done += len(titles)
        kept += len(docs)
        redirs += len(reds)
        el = time.time() - t0
        print(f"\r  색인 중 {done:,}/{total:,}  문서 {kept:,}  넘겨주기 {redirs:,}  ({el:.0f}s)", end="", flush=True)
    print()
    db.commit()
    db.execute("CREATE INDEX docs_nocase ON docs(title COLLATE NOCASE)")
    db.commit()
    db.execute("VACUUM")
    db.close()
    if db_path.exists():
        db_path.unlink()
    os.replace(tmp, db_path)
    print(f"완료: {db_path} ({db_path.stat().st_size / 1e6:.0f} MB), 문서 {kept:,}개")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", help="이미 받아둔 parquet 파일 경로")
    p.add_argument("--out", default=str(ROOT / "data" / "namuwiki.db"))
    p.add_argument("--keep-parquet", action="store_true", help="색인 후 parquet를 지우지 않음")
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.parquet:
        parquet = Path(a.parquet)
    else:
        parquet = ROOT / "data" / "namuwiki_20210301.parquet"
        print("나무위키 덤프를 내려받아 (약 3GB, 중간에 끊겨도 다시 실행하면 이어서 받음)")
        download(parquet)
    print("필요한 부분만 골라 색인을 만들어")
    build(parquet, out)
    if not a.parquet and not a.keep_parquet:
        parquet.unlink()
        print("원본 parquet는 지웠어 (--keep-parquet로 남길 수 있음)")


if __name__ == "__main__":
    main()
