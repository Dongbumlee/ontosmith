#!/usr/bin/env python
"""doc_check — deterministic document consistency checks (Task Ledger).

Checks (all offline, $0):
  1. terminology lint   — docs/*.md against docs/terms_lint.txt
  2. stale values       — superseded figures from docs/facts.yaml must
                          not appear in docs/*.md
  3. reference integrity— relative links in docs/*.md must resolve
  4. secret scan        — credential patterns anywhere in docs/ and
                          in the file list passed via --files

Usage: python scripts/doc_check.py [--files f1 f2 ...]
Exit 0 = clean, 1 = findings (printed one per line).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SECRET_PATTERNS = [
    r"sk-or-v1-[a-f0-9]{32,}",
    r"OPENROUTER_API_KEY=[A-Za-z0-9_\-]{20,}",
    r"ghp_[A-Za-z0-9]{30,}",
    r"AKIA[0-9A-Z]{16}",
]


def md_files():
    return sorted(DOCS.glob("*.md"))


def check_terms(findings):
    pats = []
    for line in (DOCS / "terms_lint.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pats.append(re.compile(line))
    for f in md_files():
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            for p in pats:
                if p.search(line):
                    findings.append(f"TERMS {f.name}:{i}: banned pattern /{p.pattern}/")


def load_waivers():
    waived = set()
    wf = DOCS / "doc_check_waivers.txt"
    if wf.exists():
        for line in wf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2:
                    waived.add((parts[0], parts[1]))
    return waived


def check_stale(findings):
    text = (DOCS / "facts.yaml").read_text()
    superseded = re.findall(r'superseded:\s*\[([^\]]*)\]', text)
    values = set()
    for group in superseded:
        values.update(v.strip().strip('"') for v in group.split(",") if v.strip())
    waived = load_waivers()
    for f in md_files():
        body = f.read_text(errors="replace")
        for v in values:
            if v and v in body and (f.name, v) not in waived:
                findings.append(f"STALE {f.name}: superseded value {v} still present")


def check_refs(findings):
    link_re = re.compile(r"\]\(([^)]+)\)")
    for f in md_files():
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            for target in link_re.findall(line):
                if target.startswith(("http", "#", "mailto:")):
                    continue
                path = (f.parent / target.split("#")[0]).resolve()
                if target.split("#")[0] and not path.exists():
                    findings.append(f"REF {f.name}:{i}: missing target {target}")


def check_secrets(findings, extra_files):
    pats = [re.compile(p) for p in SECRET_PATTERNS]
    targets = list(md_files()) + [ROOT / p for p in extra_files]
    for f in targets:
        if not f.exists() or f.is_dir():
            continue
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            for p in pats:
                if p.search(line):
                    findings.append(f"SECRET {f.relative_to(ROOT)}:{i}: credential pattern")


def main():
    findings = []
    extra = []
    if "--files" in sys.argv:
        extra = sys.argv[sys.argv.index("--files") + 1:]
    check_terms(findings)
    check_stale(findings)
    check_refs(findings)
    check_secrets(findings, extra)
    for f in findings:
        print(f)
    print(f"doc_check: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
