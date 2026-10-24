"""Fetch the hands-on demo corpus from its source ($0, no key).

The demo corpus is **fetched, never bundled** (release plan
R13): Re-DocRED's document text derives from Wikipedia and
carries upstream terms, and the repo's fresh-history rule
does not carry data blobs. This script downloads the
Re-DocRED dev file from the dataset's own repository and
writes a small, deterministic subset — the first N documents
in file order — as plain-text files a profiler or batch can
consume, plus a SOURCE.md with the attribution and terms.

Usage:
    python examples/demo/fetch_demo.py [--out DIR] [--docs N]
                                       [--dry-run]

Default: 40 documents into ./demo-data/ (beside your work,
never inside this repository's tracked tree unless you point
it there). At the flagship build's realized per-document
cost ($0.03736/doc; quality record, whitepaper §5.6), building the
default subset live costs roughly $1-2 — the counted
projection in the guide (docs/hands-on.md) prices your exact
subset before any spend.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

SOURCE_URL = ("https://raw.githubusercontent.com/tonytan48/"
              "Re-DocRED/main/data/dev_revised.json")
SOURCE_NAME = "Re-DocRED dev split (Tan et al., EMNLP 2022)"
LICENSE_NOTE = (
    "Re-DocRED is distributed under the MIT license (dataset "
    "repository); its document text derives from Wikipedia "
    "and remains under the upstream CC BY-SA terms. This "
    "fetch is for the hands-on walkthrough; redistribution "
    "terms are the upstream ones, not this project's.")

# Realized flagship rates (quality record, whitepaper §5.6) —
# an expectation for planning, not a price quote.
REALIZED_USD_PER_DOC = 0.03736


def doc_text(doc: dict) -> str:
    """Document text from the dataset's tokenized sentences."""
    return "\n".join(" ".join(sent) for sent in doc["sents"])


def write_subset(docs: list[dict], out: Path, n: int) -> list[Path]:
    """Write the first n documents as doc_NNN.txt + SOURCE.md."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for i, doc in enumerate(docs[:n]):
        path = out / f"doc_{i:03d}.txt"
        path.write_text(f"{doc['title']}\n\n{doc_text(doc)}\n",
                        encoding="utf-8")
        written.append(path)
    (out / "SOURCE.md").write_text(
        f"# Demo corpus — fetched, not bundled\n\n"
        f"- Source: {SOURCE_NAME}\n- URL: {SOURCE_URL}\n"
        f"- Subset: first {len(written)} documents of the dev "
        f"file, in file order (deterministic)\n"
        f"- Terms: {LICENSE_NOTE}\n",
        encoding="utf-8")
    return written


def main(argv=None) -> int:
    """Fetch (or, with --dry-run, only describe) the subset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="demo-data",
                        help="output directory (default ./demo-data)")
    parser.add_argument("--docs", type=int, default=40,
                        help="subset size in documents (default 40)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan; download nothing")
    args = parser.parse_args(argv)
    out = Path(args.out)
    print(f"Source : {SOURCE_NAME}")
    print(f"URL    : {SOURCE_URL}")
    print(f"Terms  : {LICENSE_NOTE}")
    print(f"Subset : first {args.docs} documents -> {out}/")
    print(f"Expect : at the flagship's realized "
          f"${REALIZED_USD_PER_DOC}/doc, a live build of this "
          f"subset projects to roughly "
          f"${args.docs * REALIZED_USD_PER_DOC:.2f} before the "
          f"counted projection prices it exactly.")
    if args.dry_run:
        print("Dry run — nothing downloaded.")
        return 0
    print("Downloading dev file (one file, ~tens of MB) ...")
    with urllib.request.urlopen(SOURCE_URL) as response:
        docs = json.loads(response.read().decode("utf-8"))
    if len(docs) < args.docs:
        print(f"source dev file holds only {len(docs)} documents",
              file=sys.stderr)
        return 1
    written = write_subset(docs, out, args.docs)
    total = sum(p.stat().st_size for p in written)
    print(f"Wrote {len(written)} documents ({total:,} bytes) to {out}/")
    print("Next: ontosmith profile " + str(out) + " --family redocred")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
