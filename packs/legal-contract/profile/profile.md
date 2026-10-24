# Corpus Profile — cuad (setup service, stage 0)

**Generated:** 2026-10-15T19:26:01.231213+00:00 · **Service:** setup-service/1.0 · **Status:** PENDING_APPROVAL · **Profile hash:** `sha256:f46cd0efb7b4b85c`
**Corpus:** `data/cuad_v1/CUAD_v1` · **Draft model:** openai/gpt-5

Evidence grades: [FULL] deterministic count over the whole corpus · [SAMPLE n=…] read from the dispersed sample · [JUDGMENT] awaiting human approval.

## inventory

- [FULL] The CUAD family contains 510 files representing 509 unique-content documents in the full deterministic inventory.
  - measured: 509 (verdict: confirmed, source: draft)
- [FULL] There is exactly one duplicate-content group in the inventory (e.g., two versions of the Aduro Biotech consulting agreement).
  - measured: 509 (verdict: confirmed, source: draft)
- [FULL] Corpus holds 510 document files, 509 unique by content, 26807133 chars total.
  - measured: 509 (verdict: measured, source: scanner)

## labels & metadata

- [FULL] The gold annotations file declares 41 label categories with titles for 510 documents.
  - measured: 41 (verdict: confirmed, source: draft)
- [JUDGMENT] Audit the derived master_clauses.csv before trusting values because it is not a source-of-truth and may contain propagation errors from span gold.
  - measured: 0.8587 (verdict: confirmed, source: draft)
- [JUDGMENT] Validate that all 510 gold annotation titles can be matched to files after normalization of whitespace, case, and exhibit numbering.
  - measured: 1 (verdict: confirmed, source: draft)
- [JUDGMENT] Some titles contain the word 'Amendment' and should be flagged for special handling of scope and supersession.
  - measured: 7 (verdict: confirmed, source: draft)

## entity terrain

- [FULL] Across the full 510-document corpus, 396 documents (77.65%) include an 'Exhibit' header with a number.
  - measured: 396 (verdict: revised, source: draft)
- [FULL] Parenthetical defined-term patterns (e.g., Company, Service Provider) are pervasive in this corpus.
  - measured: 6001 (verdict: confirmed, source: draft)
- [FULL] Parties named in this corpus are predominantly organizations (e.g., International Business Machines Corporation, Philips Medical Systems Nederland B.V., T-Mobile Netherlands B.V.) with occasional individuals in officer or signatory roles.
  - measured: 156 (verdict: confirmed, source: draft)

## identity & aliases

- [JUDGMENT] Role nouns such as Company, Consultant, Employer, Executive, Service Provider, Service Recipient, Party, Parties, Party A/B, and Affiliate are not identities and must be resolved to their defining named entities within each document.
- [JUDGMENT] Alias chains that define multiple labels for the same party (e.g., 'Application Provider' or 'e-centives') should be unified to a single canonical node per document.

## relation terrain

- [JUDGMENT] Governing law clauses are expected to be common and should be detected via headings or phrases like 'This Agreement shall be governed by'.
  - measured: 367 (verdict: confirmed, source: draft)
- [FULL] Across the full 510-document corpus, 272 documents (53.33%) contain the token 'WHEREAS'.
  - measured: 272 (verdict: revised, source: draft)
- [JUDGMENT] Termination, confidentiality, IP ownership, limitation of liability, force majeure, data protection/privacy, and governing law/dispute resolution are recurrent clause families in this corpus and should be targeted for relation extraction.
  - measured: 467 (verdict: confirmed, source: draft)
- [JUDGMENT] Headings like 'Non-Competition', 'Exclusivity', and 'Right of First Offer' appear as dedicated sections and can be mapped to competitive restriction relations.
  - measured: 154 (verdict: confirmed, source: draft)

## attribution terrain

- [JUDGMENT] Typed signature markers using '/s/' appear in some filings and should be recognized as signature attributions.
  - measured: 350 (verdict: confirmed, source: draft)

## value formats

- [FULL] Across the full corpus, effective-date spans are parseable at a 59.28% rate (265 of 447 spans), indicating substantial variation in date expressions.
  - measured: 0.5928 (verdict: revised, source: draft)
- [JUDGMENT] Currency values generally use a dollar sign and thousands separators (e.g., '$1,000') and may appear in monthly or annualized compensation contexts.
  - measured: 273 (verdict: confirmed, source: draft)
- [JUDGMENT] Section numbering varies between arabic levels (1., 1.1., 1.1.1) and lettered schedules/appendices, so hierarchical parsing must support mixed schemes.

## document families

- [FULL] Across the full 510-document corpus, 508 documents (99.61%) contain an agreement-type cue such as 'agreement', 'non-competition', or 'joint filing'.
  - measured: 508 (verdict: revised, source: draft)

## text-hygiene traps

- [JUDGMENT] Redactions and placeholders appear in multiple styles, including '[*****]', '[*]', '***', '[•]', and '[address removed]', and should be normalized to a single redaction token rather than extracted as values.
  - measured: 144 (verdict: confirmed, source: draft)
- [JUDGMENT] SEC-source boilerplate like 'Source: [COMPANY], [FILING], [DATE]' lines and repeated exhibit headers should be stripped before span alignment.
  - measured: 351 (verdict: confirmed, source: draft)
- [JUDGMENT] OCR artifacts and all-caps headings (e.g., 'TIHS' for 'THIS', random digit/letter noise) occur in certain PDFs and can confound literal matching unless case-folding and fuzzy tolerance are applied.

## category-to-task mapping

- [JUDGMENT] Map CUAD 'Effective Date' to a single normalized date value extracted from definitions or the cover heading, tolerating ordinal and DMY/M DY/Y-M-D formats.
- [JUDGMENT] Map CUAD 'Parties' to named-entity extraction of legal names and jurisdictions from the preamble, with aliases bound to canonical per-document nodes.
- [JUDGMENT] Map CUAD 'Governing Law' to jurisdiction extraction from the governing law or dispute resolution section heading/body.
- [JUDGMENT] Map CUAD 'Term' to a duration relation capturing start trigger, end trigger, and renewal/auto-renewal conditions from the Term section.
- [JUDGMENT] Map CUAD 'Confidentiality' to a clause-presence boolean plus scope (parties, affiliates, data types) and carve-outs if spans are available.
- [JUDGMENT] Map CUAD 'IP Ownership' to a directional relation (assignor->assignee, background vs. foreground IP) extracted from IP sections and definitions.
- [JUDGMENT] Map CUAD 'Limitation of Liability' to a numeric/qualifier capture (cap amount, exclusions, 'consequential damages' carve-outs).

## labels

- [FULL] [labels] category_census: {"n_categories": 41, "categories": ["Affiliate License-Licensee", "Affiliate License-Licensor", "Agreement Date", "Anti-Assignment", "Audit Rights", "Cap On Liability", "Change Of Control", "Competitive Restriction Exception", "Covenant Not To Sue", "Document Name", "Effective Date", "Exclusivity", 
  - measured: 41 (verdict: measured, source: scanner)
- [FULL] [labels] csv_flag_audit: {"rows": 510, "answer_columns": 40, "non_empty_cells": 19599, "yes_no_cells": 16830, "yes_no_share": 0.8587, "distinct_other_values": 1605, "other_examples": {"New York": 97, "perpetual": 69, "successive 1 year": 60, "California": 55, "Delaware": 40, "Texas": 28, "SPONSORSHIP AGREEMENT": 20, "Florid
  - measured: 0.8587 (verdict: measured, source: scanner)
- [FULL] [labels] annotation_titles: {"n_gold_titles": 510, "n_unmatched_exact": 1, "unmatched_exact": ["LECLANCHE\u0301 S.A. - JOINT DEVELOPMENT AND MARKETING AGREEMENT"], "n_unmatched_normalized": 1, "unmatched_normalized": ["LECLANCHE\u0301 S.A. - JOINT DEVELOPMENT AND MARKETING AGREEMENT"], "n_files_without_gold_title": 1, "files_w
  - measured: 1 (verdict: measured, source: scanner)

## hygiene

- [FULL] Redaction marker [***] appears in 65 of 510 documents (12089 occurrences).
  - measured: 65 (verdict: measured, source: scanner)
- [FULL] Spaced redaction marker [ * * * ] appears in 2 documents (16 occurrences).
  - measured: 2 (verdict: measured, source: scanner)

## identity

- [FULL] [identity] defined_terms: {"docs_with": 471, "docs_total": 510, "total_captures": 6001}
  - measured: 6001 (verdict: measured, source: scanner)
- [FULL] [identity] defined_term_scope: {"terms_total": 3581, "terms_in_ge2_docs": 502, "terms_with_differing_bindings": 459, "share_differing": 0.9143, "examples": [{"term": ".pdf", "bindings": ["SIGNATURES TO THIS AGREEMENT TRANSMITTED BY FACSIMILE BY ELECTRONIC MAIL IN PORTABLE DOCUMENT FORMAT", "THE DELIVERY OF SIGNED COUNTERPARTS BY 
  - measured: 459 (verdict: measured, source: scanner)
- [FULL] [identity] party_recurrence: {"distinct_parties": 1895, "parties_in_ge2_docs": 156, "top_recurring": {"DISTRIBUTOR": 25, "LICENSEE": 18, "LICENSOR": 14, "CUSTOMER": 13, "SPONSOR": 12, "CONSULTANT": 11, "CLIENT": 11, "PARTY A": 11, "PARTY B": 11, "AGENT": 9}, "spot_check": {}}
  - measured: 156 (verdict: measured, source: scanner)

## values

- [FULL] [values] date_parse_rate: {"category": "Expiration Date", "spans": 467, "parseable": 102, "parse_rate": 0.2184, "unparseable_examples": ["The term of this  Agreement  shall be ten (10)                            years (the \"Term\")  which shall  commence on the date                            upon which the Company  deliver
  - measured: 0.2184 (verdict: measured, source: scanner)

## families

- [FULL] [families] amendment_titles: {"docs_with_amendment_in_title": 7, "examples": ["ADMA BioManufacturing, LLC -  Amendment #3 to Manufacturing Agreement ", "BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT AGREEMENT - First Amendment", "FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT_AMENDMENT", "FE
  - measured: 7 (verdict: measured, source: scanner)

## Decision points (human approval required)

- Audit the derived master_clauses.csv before trusting values because it is not a source-of-truth and may contain propagation errors from span gold.
- Validate that all 510 gold annotation titles can be matched to files after normalization of whitespace, case, and exhibit numbering.
- Role nouns such as Company, Consultant, Employer, Executive, Service Provider, Service Recipient, Party, Parties, Party A/B, and Affiliate are not identities and must be resolved to their defining named entities within each document.
- Alias chains that define multiple labels for the same party (e.g., 'Application Provider' or 'e-centives') should be unified to a single canonical node per document.
- Redactions and placeholders appear in multiple styles, including '[*****]', '[*]', '***', '[•]', and '[address removed]', and should be normalized to a single redaction token rather than extracted as values.
- SEC-source boilerplate like 'Source: [COMPANY], [FILING], [DATE]' lines and repeated exhibit headers should be stripped before span alignment.
- OCR artifacts and all-caps headings (e.g., 'TIHS' for 'THIS', random digit/letter noise) occur in certain PDFs and can confound literal matching unless case-folding and fuzzy tolerance are applied.
- Some titles contain the word 'Amendment' and should be flagged for special handling of scope and supersession.
- Section numbering varies between arabic levels (1., 1.1., 1.1.1) and lettered schedules/appendices, so hierarchical parsing must support mixed schemes.
- Map CUAD 'Effective Date' to a single normalized date value extracted from definitions or the cover heading, tolerating ordinal and DMY/M DY/Y-M-D formats.
- Map CUAD 'Parties' to named-entity extraction of legal names and jurisdictions from the preamble, with aliases bound to canonical per-document nodes.
- Map CUAD 'Governing Law' to jurisdiction extraction from the governing law or dispute resolution section heading/body.
- Map CUAD 'Term' to a duration relation capturing start trigger, end trigger, and renewal/auto-renewal conditions from the Term section.
- Map CUAD 'Confidentiality' to a clause-presence boolean plus scope (parties, affiliates, data types) and carve-outs if spans are available.
- Map CUAD 'IP Ownership' to a directional relation (assignor->assignee, background vs. foreground IP) extracted from IP sections and definitions.
- Map CUAD 'Limitation of Liability' to a numeric/qualifier capture (cap amount, exclusions, 'consequential damages' carve-outs).
