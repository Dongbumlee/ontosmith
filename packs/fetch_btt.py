"""Collect Behind the Tech with Kevin Scott transcripts (GraphRAG paper corpus).

Source: podscripts.co mirror of the show's official transcripts
(the official site is a JS app; transcripts are not in static HTML).
Strips the mirror's "Starting point is HH:MM:SS" markers.
Per-episode files in data/btt/ + manifest.json. Resumable: existing
episode files are skipped.

Usage: python fetch_btt.py [--limit N]
"""
import json
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://podscripts.co"
PODCAST = f"{BASE}/podcasts/behind-the-tech-with-kevin-scott"
OUT = Path("data/btt")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research corpus collection)"}


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


class TextExtract(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1
        if tag in ("p", "div", "br", "li") and not self.skip:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def episode_links():
    links, page = [], 1
    while True:
        html = fetch(f"{PODCAST}?page={page}") if page > 1 else fetch(PODCAST)
        found = re.findall(r'href="(/podcasts/behind-the-tech-with-kevin-scott/[^"]+)"', html)
        new = [f"{BASE}{f}" for f in found if f"{BASE}{f}" not in links]
        if not new:
            break
        links += new
        page += 1
        if page > 12:
            break
    return links


def extract_transcript(html):
    p = TextExtract()
    p.feed(html)
    text = "".join(p.parts)
    text = re.sub(r"Starting point is \d{1,2}:\d{2}(:\d{2})?", " ", text)
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]
    lines = [l for l in lines if len(l) > 40]  # drop nav/UI crumbs
    # Transcript starts at the first long run of prose; take the tail
    # block from the first line mentioning the show intro or a speaker.
    return "\n".join(lines)


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    links = episode_links()
    print(f"episodes found: {len(links)}")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    done = {m["url"] for m in manifest}
    todo = [l for l in links if l not in done]
    if limit:
        todo = todo[:limit]
    for i, url in enumerate(todo):
        slug = url.rstrip("/").split("/")[-1]
        try:
            text = extract_transcript(fetch(url))
        except Exception as e:
            print(f"FAIL {slug}: {e}")
            continue
        (OUT / f"{slug}.txt").write_text(text)
        manifest.append({"slug": slug, "url": url, "chars": len(text),
                         "est_tokens": int(len(text) / 4)})
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"[{i+1}/{len(todo)}] {slug}: {len(text)} chars")
        time.sleep(1)
    tot = sum(m["chars"] for m in manifest)
    print(f"total: {len(manifest)} episodes, {tot} chars, ~{tot//4} tokens")


if __name__ == "__main__":
    main()
