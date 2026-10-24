"""Build the PyPI variant of the README (generated; issue #78).

PyPI renders the package long description through
readme_renderer (the Warehouse path), and it does NOT
resolve repository-relative links or images — every
relative target in the GitHub README would 404 on the
PyPI project page, silently (the master renders with
zero renderer warnings; the baseline audit is recorded
in the project's internal run log).

So the GitHub README stays the master, and this script
generates the PyPI variant from it:

  * relative links to files  -> <base>/blob/<branch>/...
  * relative links to a directory -> <base>/tree/<branch>/...
  * relative images          -> <base>/raw/<branch>/...
  * the brand banner the master carries directly under
    its title (assets/brand/banner.png, run 21) is
    verified — the build refuses if it is missing — and
    rewritten to its absolute raw URL like every other
    image;
  * the clone-install wording is substituted with the
    PyPI form (`pip install ontosmith`) — on the PyPI
    page the reader is already there. The substitutions
    are anchored to exact master text and the build
    FAILS if an anchor is missing, so master and variant
    cannot drift apart silently;
  * the master is audited first: raw HTML, GitHub alert
    syntax, HTML comments, and fragment-only links are
    PyPI-hostile or unrewritable, and any of them aborts
    the build instead of shipping a broken page.

<base> is one declared parameter (PUBLIC_REPO_BASE):
the public owner/repo is a launch decision, and when it
is fixed this default changes in exactly one place.

Publish-time selection (the release flow, in the
ephemeral release checkout):
  1. python scripts/build_pypi_readme.py
  2. cp build/pypi/README.md README.md
  3. python -m build        # sdist + wheel carry the variant

The committed pyproject.toml keeps `readme = "README.md"`
for every build; the release checkout swaps the FILE's
content, never the metadata. Why not a pyproject
override pointing at build/pypi/README.md? Proven
broken when measured, on two independent layers: `build/`
is on the sdist prune list, so the variant file never
enters the sdist — and a wheel built FROM that sdist
then ships with `long_description` missing while twine
still exits 0. A MANIFEST.in graft cannot rescue it
(the prune runs after the graft). The root README.md,
by contrast, is in every sdist by construction, so
swapping its content in the disposable checkout is the
one selection mechanism that is correct on every build
path: tree wheel, tree sdist, and from-sdist wheel.
The variant is committed at build/pypi/README.md as
the checked artefact (force-added past the generic
build/ ignore) and kept fresh by
tests/test_doc_guards.py; the publish flow still
regenerates it in the release checkout.

Exit codes: 0 = variant + override written; 2 = audit
or anchor failure (nothing is written).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_BASE = "https://github.com/Dongbumlee/ontosmith"
DEFAULT_BRANCH = "main"

# The brand banner (run 21: logo concept A, Forged
# Triad, picked by Dongbum). The master carries it
# directly under the title as a relative image; the
# variant must carry it exactly once, in the header
# above the first section, rewritten to the absolute
# raw form. A master without it is refused — the PyPI
# page never ships bannerless by accident.
BANNER_IMAGE = "assets/brand/banner.png"
BANNER_LINE = f"![OntoSmith]({BANNER_IMAGE})"

# Anchored substitutions: (master text, PyPI text). Each
# master text must occur EXACTLY once in the master.
SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    (
        "Install from a clone of this repository (at launch this\n"
        "becomes `pip install ontosmith` on PyPI; the clone route\n"
        "is the one that works today):",
        "Install from PyPI:",
    ),
    (
        "$ pip install .",
        "$ pip install ontosmith",
    ),
    (
        "Optional surfaces install as extras: `pip install .[nlp]`\n"
        "(the pipeline candidate path's local NER), `.[viz]`.\n"
        "(The `.[neo4j]` extra installs the official driver the\n"
        "export utilities use; the build itself talks to Neo4j\n"
        "over HTTP and needs the server, not the extra.)",
        "Optional surfaces install as extras:\n"
        '`pip install "ontosmith[nlp]"`\n'
        "(the pipeline candidate path's local NER),\n"
        '`"ontosmith[viz]"`.\n'
        '(The `"ontosmith[neo4j]"` extra installs the official\n'
        "driver the export utilities use; the build itself talks\n"
        "to Neo4j over HTTP and needs the server, not the extra.)",
    ),
)

_LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")
_HTML_TAG_RE = re.compile(r"<[a-zA-Z/!][^>\n]*>")
_ALERT_RE = re.compile(r"^\s*> \[!", re.MULTILINE)
_COMMENT_RE = re.compile(r"<!--")
_FENCE_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)


class AuditFailure(Exception):
    """The master README cannot be transformed safely."""


def audit(master: str) -> list[str]:
    """PyPI-hostile constructs in the master, prose only.

    Fenced code blocks are stripped before scanning: code
    samples may legitimately contain angle brackets; prose
    may not (readme_renderer sanitizes raw HTML away, so a
    prose tag would vanish from the PyPI page silently).
    """
    prose = _FENCE_RE.sub("", master)
    problems: list[str] = []
    if _HTML_TAG_RE.search(prose):
        problems.append("raw HTML tag in prose")
    if _ALERT_RE.search(prose):
        problems.append("GitHub alert syntax (> [!...]) in prose")
    if _COMMENT_RE.search(prose):
        problems.append("HTML comment in the master")
    for match in _LINK_RE.finditer(master):
        if match.group(3).startswith("#"):
            problems.append(
                f"fragment-only link target {match.group(3)!r} "
                "cannot be rewritten to an absolute URL")
    return problems


def _rewrite_target(target: str, is_image: bool, base: str,
                    branch: str, counts: dict[str, int]) -> str:
    if target.startswith(("http://", "https://", "mailto:")):
        return target
    if is_image:
        counts["raw"] += 1
        return f"{base}/raw/{branch}/{target}"
    if target.endswith("/"):
        counts["tree"] += 1
        return f"{base}/tree/{branch}/{target.rstrip('/')}"
    counts["blob"] += 1
    return f"{base}/blob/{branch}/{target}"


def build_variant(master: str, base: str = DEFAULT_BASE,
                  branch: str = DEFAULT_BRANCH,
                  ) -> tuple[str, dict[str, int]]:
    """Transform the master README text into the variant.

    Returns (variant text, rewrite counts by URL kind).
    Raises AuditFailure on hostile constructs, a missing
    substitution anchor, or a leftover relative target.
    """
    problems = audit(master)
    if problems:
        raise AuditFailure("; ".join(problems))
    if not master.startswith("# "):
        raise AuditFailure("master does not open with an H1 title")
    if master.count(BANNER_LINE) != 1:
        raise AuditFailure(
            f"master must carry the brand banner line "
            f"{BANNER_LINE!r} exactly once (found "
            f"{master.count(BANNER_LINE)}) — the logo "
            "lands in the master first (run 21)")

    text = master
    for old, new in SUBSTITUTIONS:
        if text.count(old) != 1:
            raise AuditFailure(
                f"substitution anchor occurs {text.count(old)}x "
                f"(expected 1): {old[:60]!r}...")
        text = text.replace(old, new)

    counts = {"blob": 0, "tree": 0, "raw": 0}

    def _sub(match: re.Match[str]) -> str:
        bang, label, target = match.groups()
        new = _rewrite_target(target, bang == "!", base,
                              branch, counts)
        return f"{bang}[{label}]({new})"

    text = _LINK_RE.sub(_sub, text)

    leftovers = [m.group(3) for m in _LINK_RE.finditer(text)
                 if not m.group(3).startswith(
                     ("http://", "https://", "mailto:"))]
    if leftovers:
        raise AuditFailure(
            f"relative targets survived the rewrite: {leftovers}")

    # Brand banner LAST: the rewritten variant must carry
    # it exactly once, in the header (above the first
    # section heading) — the slot the run-19 generator
    # used to reserve is now filled, and this check is
    # what guards it.
    banner_abs = (f"![OntoSmith]({base}/raw/{branch}/"
                  f"{BANNER_IMAGE})")
    if text.count(banner_abs) != 1:
        raise AuditFailure(
            "the rewritten banner did not survive exactly "
            f"once: {banner_abs!r}")
    first_section = text.find("\n## ")
    if first_section != -1 \
            and text.index(banner_abs) > first_section:
        raise AuditFailure(
            "the brand banner sits below the first "
            "section heading, not in the header")
    return text, counts


def check_pyproject_master(pyproject: str) -> None:
    """The committed metadata must keep pointing at the master.

    The publish-time selection swaps file CONTENT in the
    release checkout; if the committed pyproject ever
    points elsewhere, master and variant have forked —
    refuse to generate.
    """
    if pyproject.count('readme = "README.md"') != 1:
        raise AuditFailure(
            'pyproject must carry exactly one '
            'readme = "README.md" line')


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Generate the PyPI README variant "
                    "from the GitHub master README.")
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="PUBLIC_REPO_BASE — the public "
                             "repository's absolute URL "
                             f"(default: {DEFAULT_BASE})")
    parser.add_argument("--branch", default=DEFAULT_BRANCH,
                        help=f"branch for blob/tree/raw URLs "
                             f"(default: {DEFAULT_BRANCH})")
    parser.add_argument("--readme", type=Path,
                        default=root / "README.md",
                        help="master README (default: repo README.md)")
    parser.add_argument("--pyproject", type=Path,
                        default=root / "pyproject.toml",
                        help="master pyproject "
                             "(default: repo pyproject.toml)")
    parser.add_argument("--out", type=Path,
                        default=root / "build" / "pypi" / "README.md",
                        help="variant output path "
                             "(default: build/pypi/README.md)")
    args = parser.parse_args(argv)

    if args.out.resolve() == args.readme.resolve():
        print("build_pypi_readme: REFUSED: --out would "
              "overwrite the master README; the content swap "
              "happens in the release checkout, not here",
              file=sys.stderr)
        return 2

    master = args.readme.read_text(encoding="utf-8")
    try:
        check_pyproject_master(
            args.pyproject.read_text(encoding="utf-8"))
        variant, counts = build_variant(master, args.base,
                                        args.branch)
    except AuditFailure as exc:
        print(f"build_pypi_readme: REFUSED: {exc}",
              file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(variant, encoding="utf-8")

    total = sum(counts.values())
    print(f"base:     {args.base} (branch {args.branch})")
    print(f"rewrote:  {total} relative targets "
          f"(blob {counts['blob']}, tree {counts['tree']}, "
          f"raw {counts['raw']})")
    print(f"applied:  {len(SUBSTITUTIONS)} anchored substitutions, "
          "brand banner verified under the title")
    print(f"wrote:    {args.out}")
    print("release:  in the release checkout — "
          "cp build/pypi/README.md README.md, then build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
