# BioRED corpus census — the Bio Medical pack's corpus census

Measured 2026-10-16 by direct parse of `data/biored/BioRED/{Train,Dev,Test}.BioC.JSON`
(local files only, no API calls). Parsing method: entity annotations live
**inside passages**, not at document level — each document has exactly 2
passages (passage 0 = title, passage 1 = abstract) in all 600 documents.
Relations live at document level and reference **gold identifier strings**
(`infons.entity1/entity2`), so relation analysis joins endpoint IDs to the
document's own passage annotations. PMIDs are disjoint across splits
(overlap verified: 0).

## Inventory

| Split | Docs | Chars | Words | Mean chars/doc | Median | Min | Max |
|---|---|---|---|---|---|---|---|
| Train | 400 | 660,010 | 94,242 | 1,650 | 1,678 | 355 | 3,294 |
| Dev | 100 | 172,553 | 24,917 | 1,726 | 1,769 | 703 | 2,875 |
| Test | 100 | 167,959 | 24,087 | 1,680 | 1,720 | 544 | 2,700 |
| **Total** | **600** | **999,922** | **143,246** | — | — | — | — |

Abstracts have no section structure beyond title + abstract (structured
labels like `OBJECTIVE:` occur inline in the abstract text only).

## Entity mentions (passage-level annotations)

| Type | Train | Dev | Test | Total | Distinct gold IDs (union) |
|---|---|---|---|---|---|
| GeneOrGeneProduct | 4,430 | 1,087 | 1,180 | 6,697 | 1,698 |
| DiseaseOrPhenotypicFeature | 3,646 | 982 | 917 | 5,545 | 812 |
| ChemicalEntity | 2,853 | 822 | 754 | 4,429 | 658 |
| OrganismTaxon | 1,429 | 370 | 393 | 2,192 | 47 |
| SequenceVariant | 890 | 250 | 241 | 1,381 | 678 |
| CellLine | 103 | 22 | 50 | 175 | 72 |
| **Total** | **13,351** | **3,533** | **3,535** | **20,419** | 3,962 distinct ID strings overall |

**Identifier coverage: 20,356 / 20,419 mentions (99.69%) carry a gold
database identifier.** The 63 without: Gene 20, Chemical 36, Disease 7;
zero for OrganismTaxon, SequenceVariant, CellLine.

Identifier namespaces by type:

- **GeneOrGeneProduct** → NCBI Gene (numeric). Includes **529 mentions
  corpus-wide whose identifier is a comma-separated list of NCBI Gene IDs**
  (Train 367, Dev 77, Test 85) — gene-family/group mentions annotated with
  several IDs at once.
- **DiseaseOrPhenotypicFeature** → MeSH descriptors (D#####) and MeSH
  supplementary concepts (C######); some mentions carry comma-lists of MeSH
  IDs too.
- **ChemicalEntity** → MeSH D and C numbers (same namespace family as
  diseases — a MeSH string alone does not tell you the type).
- **OrganismTaxon** → NCBI Taxonomy (numeric).
- **SequenceVariant** → dbSNP rs numbers (784 mentions) and HGVS-style
  strings (594 mentions, e.g. `c|DEL|1314_1328|`), 3 other.
- **CellLine** → Cellosaurus (CVCL_####).
- One ID (`9685`) is annotated as both GeneOrGeneProduct (1 mention) and
  OrganismTaxon (3 mentions) — the only multi-type ID in the corpus.

## Relations (document level, endpoints = gold IDs)

| Type | Train | Dev | Test | Total | Share |
|---|---|---|---|---|---|
| Association | 2,192 | 560 | 635 | 3,387 | 52.1% |
| Positive_Correlation | 1,089 | 352 | 325 | 1,766 | 27.2% |
| Negative_Correlation | 763 | 216 | 171 | 1,150 | 17.7% |
| Bind | 61 | 19 | 9 | 89 | 1.4% |
| Cotreatment | 31 | 10 | 14 | 55 | 0.8% |
| Comparison | 28 | 5 | 6 | 39 | 0.6% |
| Drug_Interaction | 11 | 0 | 2 | 13 | 0.2% |
| Conversion | 3 | 0 | 1 | 4 | 0.06% |
| **Total** | **4,178** | **1,162** | **1,163** | **6,503** | 100% |

(Totals match the design doc's expected 6,503 / 3,387 / 1,766 / 1,150
exactly.) Gold also marks each relation `novel` (the paper's own finding)
vs background: Novel 4,532 (69.7%), No 1,971.

Pair-type signatures (top, unordered): Association is dominated by
Disease~Gene (1,419), Gene~Gene (580), Disease~Variant (483),
Chemical~Gene (317) — i.e. it spans nearly every type pair and behaves as
the default bucket. Positive_Correlation: Chemical~Disease 609,
Disease~Variant 380. Negative_Correlation: Chemical~Disease 438,
Chemical~Gene 265, Chemical~Chemical 174. Cotreatment (50/55),
Drug_Interaction (13/13), Comparison (39/39), Conversion (4/4) are
Chemical~Chemical almost or exactly exclusively.

**Zero gold relations have an OrganismTaxon or a CellLine endpoint.**
Relations exist only among Gene / Disease / Chemical / Variant.

**Endpoint grounding:** every relation endpoint resolves to an annotation
in the same document once multi-ID comma-lists are split into components —
653 endpoint references (in 568 relations, 8.7%) point at one component of
a list-valued annotation; **zero** endpoints are truly ungrounded.

**Span classes** (passage of each endpoint's mentions): both endpoints
mentioned in the abstract passage — 5,861 relations (90.1%); one endpoint
mentioned only in the title — 74 (1.1%); remainder are the multi-ID
component cases above (span class determinable after list splitting).

**Direction:** gold attests both endpoint orders for the same relation
type across the corpus (Association: Disease~Gene and Gene~Disease are
both common), and 24 Association + 19 Positive_Correlation unordered
pairs are attested in *both* orders. Endpoint order is serialization,
not semantic direction.

## Abbreviations, case, ambiguity — the ER terrain

- **Abbreviation definitions are pervasive:** 490/600 docs (81.7%)
  contain ≥1 parenthetical short-form definition (`Full Name (ABBR)`),
  1,325 total. Gold itself attests **861 in-document cases** of one ID
  carrying multiple surfaces including a short form — the corpus defines
  its abbreviations per abstract.
- **Short-form mentions:** 5,475 / 20,419 mentions (26.8%) are all-caps
  short forms (2–10 chars).
- **Surface ambiguity:** 302 distinct surfaces map to >1 gold ID.
  Worst case: `insulin` → 7 IDs (human gene 3630, rat 24505, mouse 16333,
  a multi-ID list, and MeSH chemical D007328 among them).
- **Case is meaningful:** 103 case-folded surface groups map to >1
  surface AND >1 gold ID. The case pattern is the species signal —
  human `CCL2` (6347) vs mouse `Ccl2` (20296) vs rat (24770);
  `PTPN22` (human 26191) vs `Ptpn22` (mouse 19260). Gene mention casing:
  ALLCAPS 3,579 (53%), mixed 1,507 (22%), lowercase 1,061 (16%),
  Capitalized 550 (8%) of 6,697.
- **OrganismTaxon is concentrated and generic:** 47 distinct IDs for
  2,192 mentions; human (9606) + mouse (10090) + rat (10116) = 2,021
  mentions (92%). The single most-mentioned gold ID in the corpus is
  9606 'human' (1,364 mentions) whose surfaces are mostly generic words:
  `patients`, `patient`, `man`, `woman`, `human`.
- **Cross-document recurrence is the norm (unlike CUAD defined terms):**
  879 gold IDs recur in ≥2 documents, 44 in ≥10 documents; 3,083 IDs
  appear in exactly one document.

## Notes for the profile

- Dev contains **zero** Conversion and zero Drug_Interaction gold
  relations (Bind only 19) — rare-type behavior is not measurable on
  the Dev slice.
- There is **no slot/attribute layer** in BioRED gold: entities + pair
  relations only. No analog of CUAD's 41 clause categories exists.
- The multi-ID comma-list convention (529 mentions) is a gold encoding
  detail an evaluator must handle; the build never sees it (DP-B1).
