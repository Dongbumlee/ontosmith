# Coverage account — the imported FIBO pack vs CUAD's clause categories

What the `fibo-contracts` pack can see in the corpus it was
scoped for, measured against the corpus's own annotation
scheme: CUAD's 41 clause categories (the categories the
fitted `legal-contract` pack was built around). Measured
2026-10-26 from the emitted pack and the category list; no
build is involved — this is a schema-coverage account, not
a quality measurement.

Verdicts: **covered** = a dedicated pack type or relation
carries the category's semantics. **Partial** = the
content is extracted, but only into a generic bucket
(`ContractProvision` / `ContractualCommitment`) or as an
unlinked type (a bare `Date`) — the graph records that
something is there, not what it is. **Not covered** = no
type receives it.

| CUAD category | FIBO pack receiver | Verdict |
|---|---|---|
| Agreement Date | Date via HasExecutionDate / HasEffectiveDate | covered |
| Effective Date | Date via HasEffectiveDate | covered |
| Governing Law | Jurisdiction via HasGoverningJurisdiction | covered |
| Notice Period To Terminate Renewal | NotificationProvision | covered |
| Renewal Term | ExtensionProvision | covered |
| Termination For Convenience | TerminationProvision | covered |
| Third Party Beneficiary | HasBeneficiary (Contract → party) | covered |
| Document Name | the Contract entity's own surface (no name type) | partial |
| Expiration Date | Date, but no expiration relation links it | partial |
| Warranty Duration | Warranty (the duration itself stays in the text) | partial |
| Anti-Assignment | ContractProvision bucket (isAssignable was demoted to a note) | partial |
| Audit Rights | ContractualCommitment / ContractProvision | partial |
| Cap On Liability | ContractProvision | partial |
| Change Of Control | ContractProvision | partial |
| Competitive Restriction Exception | ContractProvision | partial |
| Covenant Not To Sue | ContractualCommitment | partial |
| Exclusivity | ContractProvision | partial |
| Insurance | ContractualCommitment | partial |
| Ip Ownership Assignment | ContractProvision | partial |
| Irrevocable Or Perpetual License | ContractProvision | partial |
| Joint Ip Ownership | ContractProvision | partial |
| License Grant | ContractProvision | partial |
| Liquidated Damages | ContractProvision | partial |
| Minimum Commitment | ContractualCommitment | partial |
| Most Favored Nation | ContractProvision | partial |
| No-Solicit Of Customers | ContractualCommitment | partial |
| No-Solicit Of Employees | ContractualCommitment | partial |
| Non-Compete | ContractualCommitment | partial |
| Non-Disparagement | ContractualCommitment | partial |
| Non-Transferable License | ContractProvision | partial |
| Post-Termination Services | ContractualCommitment | partial |
| Price Restrictions | ContractProvision | partial |
| Revenue/Profit Sharing | ContractProvision | partial |
| Rofr/Rofo/Rofn | ContractProvision | partial |
| Source Code Escrow | ContractProvision (agent via HasThirdParty) | partial |
| Uncapped Liability | ContractProvision | partial |
| Unlimited/All-You-Can-Eat-License | ContractProvision | partial |
| Volume Restriction | ContractProvision | partial |
| Affiliate License-Licensee | Organization, unlinked as licensee | partial |
| Affiliate License-Licensor | Organization, unlinked as licensor | partial |

**7 of 41 covered by a dedicated type or relation (17%);
34 partial; 0 with no receiver at all.** The zero is the
catch-all's doing, not coverage: `ContractProvision`
receives every clause, so nothing is invisible — but for
34 of 41 categories the graph cannot say what the clause
*is*. Retrieval by clause type — the fitted pack's whole
design — is unavailable for those 34 under the imported
schema.

The asymmetry runs the other way as well. The imported
pack types things CUAD does not annotate at all:
provisions as typed entities (termination / extension /
disclosure / notification / netting / warranty /
representation), supersession and subordination between
contracts, principal vs counterparty vs third-party
roles, beneficiaries. A build under it would answer
questions the fitted pack's graph does not carry — a
different graph, not merely a lesser one.
