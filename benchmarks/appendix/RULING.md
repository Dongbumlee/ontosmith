# The adjudication ruling (2026-10-22) — ruled verdict set provenance

This appendix ships two verdict sets over the same 740
sampled items: the original adjudication protocol's
verdicts (`judgments/`) and the **ruled** verdicts
(`judgments_ruled/`), produced after the original protocol
was stress-tested and its contract completed by explicit
ruling. The record's headline figures (whitepaper §5) are
the ruled ones; `examples/demo/replay_scores.py` recomputes
both estimates from this appendix. The standing
adjudication spec the ruling amended is
[`docs/ADJUDICATION.md`](../../docs/ADJUDICATION.md); the
full account is in the whitepaper, §5.4.

## Why a ruled set exists

After the record was frozen, the 740 verdicts were
stress-tested: under a pre-registered design, a blind
re-adjudication of a stratified 200-item sample by a judge
from a different model family (openai/gpt-5.5; the
original panel was of the Anthropic line) agreed with the
shipped verdicts on 81.5% of items, against a
pre-registered 95% bar, and scored 29/40 on freshly
planted blind controls (true family 12/20, absent family
17/20). The disagreements were one-directional (35 of 37:
original supported → recheck not-supported) and
concentrated in six recurring surface situations the
as-typed contract had never defined — the seam classes.
Those six situations were settled by the rulings below,
and every flagged item — all 565, definite and possible
flags alike — was re-judged blind under the ruled contract
by the stricter judge family, so that the ruling, not the
judge, decided each seam. 186 of the 565 verdicts
changed; re-priced through the record's identical
estimator (which first reproduces the original figures
from the original verdicts), corrected precision is
0.9110 [0.8919–0.9275] on FULL-500. Judging spend,
metered from provider responses: $1.3314 for the recheck
(including a discarded transport-failed attempt) and
$3.7977 for the two ruled passes.

## The rulings, as approved (2026-10-22)

1. **DEM — ACCEPT.** An adjectival/demonym form that
   unambiguously denotes the typed entity counts as the
   typed value.
2. **RELPREC — STRICT.** A neighbouring relation does not
   support the typed relation.
3. **EVAL — ACCEPT**, under the operational reading stated
   in the packet (prominent presentation in the source
   text about the subject).
4. **CONT — SPLIT.** (a) Transitive containment ACCEPTS;
   (b) mere co-occurrence/association does NOT support
   containment; (c) a macro-region typed as an
   administrative entity is a type error — NOT supported
   as typed.
5. **IDENT — STRICT.** Truncated/conflated/
   granularity-shifted subjects do not carry the fact;
   "unclear" remains available where identity genuinely
   cannot be resolved.
6. **QUAL — STRICT.** A fact stated only with a qualifier
   does not support the unqualified typed fact.

## The packet text (verbatim)

The block below is the exact ruling text the ruled
re-adjudication's judge received, inside the recheck's
as-typed framing. Each ruled verdict in
`judgments_ruled/` records which rule (if any) actually
decided it, in its `seam_applied` field.

<!-- PACKET-BEGIN -->
ADJUDICATION RULES — read before judging. These claims come from typed
knowledge-graph extraction, and six recurring surface situations are
settled by the rulings below. Apply them exactly as written. Where a
ruling decides your verdict, report its code in "seam_applied"; if
several rulings bear on an item, report the one without which your
verdict would have been different. If no ruling is needed — the
ordinary as-typed reading decides the item — report "none".

R1 — DEM (adjectival / demonym values). If the claim's subject or
object appears in the document in an adjectival or demonym form that
unambiguously denotes the typed entity (for example, the object typed
as a country and named in the text by its demonym adjective, or the
reverse), that counts as the typed value: judge the claim exactly as
if the typed form appeared. Surface-form variation for the same
referent is not a mismatch.

R2 — RELPREC (relation precision). Support requires the typed
relation itself; a neighbouring relation does not support it.
Proposing is not founding; formalising is not developing; being
co-parents is not being spouses; being credited is not being a cast
member; belonging to a bloc is not membership in the typed body. If
the document states only the neighbouring relation, the verdict is
"not_supported".

R3 — EVAL (the relation "notable work"). A work that the document
presents prominently about the subject counts as supported: treated
at length, named among the subject's works in the lead or in a works
list, or itself a main topic of the document. A passing mention does
not count.

R4 — CONT (containment / located-in). (a) Transitive containment
counts: if the document states A is in B and states B is in C, a
claim that A is in C is supported. (b) Mere co-occurrence does not:
two entities appearing in the same document or sentence, an
association between them, or a shared event does not by itself
support a containment or located-in claim. (c) The typed endpoint
must be the container the document states: if the claim names a
macro-region (a cultural or geographic aggregate such as Latin
America) where the document states a different container, or names
an administrative entity where the document states only a
macro-region, the claim as typed is not supported.

R5 — IDENT (subject identity). The claim's subject must be the
entity the document's statements are about. If the typed subject is
truncated, conflated with another entity, or shifted in granularity
relative to that entity (a person vs. their office, a place vs. its
sub-area, a name missing its distinguishing part), those statements
do not support the claim: verdict "not_supported". If the text
genuinely does not allow you to decide whether the typed subject is
the document's entity, use "unclear".

R6 — QUAL (dropped qualifier). If the document states the fact only
with a qualifier — it held initially or formerly, or it holds for
one market, territory, or period — the unqualified claim as typed is
not supported.
<!-- PACKET-END -->
