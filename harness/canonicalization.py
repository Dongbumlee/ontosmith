"""Intrinsic surface canonicalization — BioRED round 4 (§4.3).

Aliases are REGISTERED, never substituted: a roster mention
keeps its canonical (extracted) surface; aliases are additional
surfaces recorded against it, each with the rule that proposed
it. Every generation rule is text-internal — it reads only the
doc text, the roster, and the profile. No rule consults gold.

Admitted generation rules (strict classes only, per the design's
hypothesis table §3.2):

- A1 — parenthetical split: a surface "X (Y)" registers X and Y,
  each IFF attested standalone in the doc (F-R4-3 tightening,
  below).
- A2 — slash split: a surface with "/" and >= 2 non-empty
  components registers each component IFF attested standalone
  in the doc (F-R4-3 tightening, below).
- A3 — attested sub-phrase: a contiguous prefix/suffix token run
  of a surface (>= 1 token, strictly shorter) registers IFF it is
  attested standalone in the doc — it occurs at a position not
  inside an occurrence of the parent surface.
- A4 — attested qualifier extension: a text phrase consisting of
  a roster surface extended on ONE side by 1–2 tokens from the
  declared QUALIFIERS list registers, iff attested in the text
  (it is, by construction: the rule reads attested occurrences).

Not admitted (counted as unaddressed upside in reporting, never
folded in): bare-acronym generation, loose/subset sub-phrases,
orthographic invention (case/Greek/plural folding — DP-B6).

Collision guard — the acceptance test every alias must pass.
A candidate is REFUSED (and counted per rule) if:
  (a) it equals the canonical surface of a DIFFERENT roster
      mention in the same doc (type-blind: a string that is
      another mention's canonical identity cannot become this
      mention's alias — this also covers the design's clause
      (c), a same-string registration under a different type);
  (b) it is already registered as an alias of a different
      mention in the same doc.
A candidate equal to its own mention's canonical surface, or a
duplicate candidate for the same mention, is skipped silently
(not a refusal). If refusals exceed 10% of generated aliases,
the round stops for an alias-policy review before any pair
spend (design F-R4-3) — the builder reports the rate and the
caller enforces the stop.

F-R4-3 amendment (fired 2026-10-18 on the first build: 337/918
refused = 36.71%): the review found the refusals were
redundancy collisions on the dense union roster — clause (a)
179 (the string already has its own roster row), clause (b)
158 (151 of them A3 sub-phrases shared across mentions, first
registrant wins) — not entity equations slipping through; the
admitted set is collision-free by construction. Per the
design's F-R4-3 prescription, the policy is tightened exactly
as named there — standalone attestation is now required for
A1/A2 components as well (A3/A4 already required it) — and the
roster is re-measured at $0. Refusals are counted per clause
(refused_a / refused_b) so the composition stays visible.
PLACEMENT (run 7, issue #73 — the PK-03 quarantine, decided):
this module is a MEASUREMENT instrument and lives with the
harness tooling. It is not a production pipeline stage and no
production module imports it — its consumers of record are
the BioRED round-4/7 decomposition and ceiling analyses, and
in the product it serves that same measurement role (alias
-generation upside analysis) plus the pack-data wiring test.
It was carried in the pipeline package under an experiment
quarantine; the quarantine is lifted by this placement, not
by promotion — the supported production surface is
pipeline/, and this module's absence from it is deliberate.
The one domain-fitted constant it carries, the A4 QUALIFIERS
set, is Bio Medical pack content (packs/bio-medical/
fitted_qualifiers.json; the pack was validated on BioRED),
loaded below as data and injectable per call. The generation
rules themselves are domain-neutral text mechanics.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# The one production-package dependency: the shared surface
# normalizer the roster surfaces are built with.
from pipeline.pair_proposer import norm_surface


def _fitted_qualifiers_candidates() -> list[Path]:
    """Where the Bio Medical pack's fitted_qualifiers.json may
    live, in resolution order — the shipped-data candidate
    discipline (api.server.app_candidates): the source tree,
    the pip --target layout (data files beside the packages),
    the regular install layout (the interpreter's data path).
    """
    import sysconfig
    pkg_parent = Path(__file__).resolve().parent.parent
    rel = Path("packs") / "bio-medical" / "fitted_qualifiers.json"
    return [
        pkg_parent / rel,
        pkg_parent / "share" / "ontosmith" / rel,
        Path(sysconfig.get_path("data")) / "share" / "ontosmith" / rel,
    ]


def _load_fitted_qualifiers() -> frozenset:
    """The A4 qualifier set as pack data (PK-03): fitted on
    BioRED text for the decomposition instrument
    (scripts/analyze_biored_r4_decomposition.py), committed with
    the rule per design §4.3 / A4 — as data under the Bio Medical
    pack (packs/bio-medical/), not as a core-code literal."""
    for cand in _fitted_qualifiers_candidates():
        if cand.is_file():
            data = json.loads(cand.read_text())
            return frozenset(data["qualifiers"])
    raise FileNotFoundError(
        "fitted_qualifiers.json (Bio Medical pack data) was not "
        "found (looked under: "
        + ", ".join(str(c) for c in _fitted_qualifiers_candidates())
        + ")")


# The BioRED-fitted qualifier set, loaded from pack content.
# Kept as a module-level name because the round-7 scripts import
# it; new callers should inject their own set via the
# `qualifiers` parameters below.
QUALIFIERS = _load_fitted_qualifiers()

_PAREN = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")


def _spans(haystack: str, needle: str) -> list[tuple[int, int]]:
    """All (start, end) occurrence spans of needle in haystack."""
    out = []
    if not needle:
        return out
    i = haystack.find(needle)
    while i >= 0:
        out.append((i, i + len(needle)))
        i = haystack.find(needle, i + 1)
    return out


def _gen_a1(surface: str) -> list[str]:
    m = _PAREN.match(surface)
    if not m:
        return []
    x, y = m.group(1).strip(), m.group(2).strip()
    return [c for c in (x, y) if c]


def _gen_a2(surface: str) -> list[str]:
    if "/" not in surface:
        return []
    parts = [p.strip() for p in surface.split("/")]
    parts = [p for p in parts if p]
    return parts if len(parts) >= 2 else []


def _attested_standalone(cand: str, surface: str,
                         norm_text: str) -> bool:
    """True iff cand occurs in the text at a position not inside
    an occurrence of the parent surface."""
    parent_spans = _spans(norm_text, surface)
    return any(not any(ps <= s and e <= pe
                       for ps, pe in parent_spans)
               for s, e in _spans(norm_text, cand))


def _gen_a3(surface: str, norm_text: str) -> list[str]:
    toks = surface.split(" ")
    if len(toks) < 2:
        return []
    out = []
    for length in range(1, len(toks)):
        for cand in (" ".join(toks[:length]),
                     " ".join(toks[length:])):
            if not cand or cand == surface:
                continue
            if _attested_standalone(cand, surface, norm_text):
                out.append(cand)
    return out


def _gen_a4(surface: str, norm_text: str,
            qualifiers: frozenset = QUALIFIERS) -> list[str]:
    # Token spans of the normalized text (single-space separated
    # after norm_surface).
    toks: list[tuple[str, int, int]] = []
    i = 0
    for m in re.finditer(r"\S+", norm_text):
        toks.append((m.group(0), m.start(), m.end()))
    surf_toks = surface.split(" ")
    n = len(surf_toks)
    out: list[str] = []
    for i in range(0, len(toks) - n + 1):
        if [t[0] for t in toks[i:i + n]] != surf_toks:
            continue
        for side in ("left", "right"):
            for k in (1, 2):
                if side == "left":
                    ext = toks[i - k:i]
                    if len(ext) < k:
                        continue
                    span = (ext[0][1], toks[i + n - 1][2])
                else:
                    ext = toks[i + n:i + n + k]
                    if len(ext) < k:
                        continue
                    span = (toks[i][1], ext[-1][2])
                if all(t[0].lower() in qualifiers for t in ext):
                    out.append(norm_text[span[0]:span[1]])
    return out


def generate_alias_candidates(mention: dict, norm_text: str,
                              qualifiers: frozenset | None = None
                              ) -> list[tuple[str, str]]:
    """(alias, rule) candidates for one mention, in rule order
    A1, A2, A3, A4 — pre-guard, duplicates within the mention
    removed (first rule wins the provenance). A1/A2 components
    are generated only when attested standalone (the F-R4-3
    tightening); A3/A4 are attestation-gated by construction.
    `qualifiers` injects the A4 extension set (default: the
    BioRED-fitted pack data, QUALIFIERS)."""
    if qualifiers is None:
        qualifiers = QUALIFIERS
    surface = mention["surface"]
    seen: dict[str, str] = {}
    for rule, gen in (("A1", _gen_a1), ("A2", _gen_a2)):
        for cand in gen(surface):
            cand = norm_surface(cand)
            if cand and cand not in seen and cand != surface \
                    and _attested_standalone(cand, surface,
                                             norm_text):
                seen[cand] = rule
    for cand in _gen_a3(surface, norm_text):
        cand = norm_surface(cand)
        if cand and cand not in seen:
            seen[cand] = "A3"
    for cand in _gen_a4(surface, norm_text, qualifiers):
        cand = norm_surface(cand)
        if cand and cand not in seen:
            seen[cand] = "A4"
    return [(c, r) for c, r in seen.items()]


def canonicalize_doc(mentions: list[dict], text: str,
                     qualifiers: frozenset | None = None
                     ) -> tuple[list[dict], dict]:
    """Apply §4.3 to one doc's roster.

    mentions: [{surface, type}] in roster order. Returns
    (enriched mentions — same rows plus "aliases":
    [{surface, rule}] in registration order — and the doc stats
    {generated, admitted, refused, refusal_rate, per_rule}).
    `qualifiers` injects the A4 extension set (default: the
    BioRED-fitted pack data).
    """
    norm_text = norm_surface(text)
    canonical_strings = {m["surface"] for m in mentions}
    alias_owner: dict[str, tuple[str, str]] = {}
    per_rule: dict[str, dict[str, int]] = {}
    generated = admitted = refused = 0
    enriched = []
    for m in mentions:
        identity = (m["surface"], m.get("type", ""))
        aliases: list[dict] = []
        own = {m["surface"]}
        for cand, rule in generate_alias_candidates(
                m, norm_text, qualifiers):
            st = per_rule.setdefault(
                rule, {"generated": 0, "admitted": 0, "refused": 0,
                       "refused_a": 0, "refused_b": 0})
            if cand in own:
                continue  # own canonical / duplicate: silent skip
            st["generated"] += 1
            generated += 1
            owner = alias_owner.get(cand)
            clause = None
            if cand in canonical_strings and cand != m["surface"]:
                clause = "refused_a"  # another mention's canonical
            elif owner is not None and owner != identity:
                clause = "refused_b"  # another mention's alias
            if clause is not None:
                st["refused"] += 1
                st[clause] += 1
                refused += 1
                continue
            alias_owner[cand] = identity
            own.add(cand)
            aliases.append({"surface": cand, "rule": rule})
            st["admitted"] += 1
            admitted += 1
        enriched.append({**m, "aliases": aliases})
    stats = {"generated": generated, "admitted": admitted,
             "refused": refused,
             "refusal_rate": round(refused / generated, 4)
             if generated else 0.0,
             "per_rule": per_rule}
    return enriched, stats
