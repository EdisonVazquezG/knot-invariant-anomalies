# Review of the saved-result collection — 18 September 2026

The author supplied `release_evidence_20260918T201246Z.zip` after running the
release checker and collector on the existing Drive results. This is a review
of stored data and source provenance, not a new execution of the scientific
pipeline.

## Collection and provenance

- 1,081 collected files: every byte count and SHA-256 matches the collection
  manifest; the ZIP CRC check also passes.
- All 25 explicitly requested result directories are present.
- The Drive check reports that all five requested raw tables and eight frozen
  inputs exist. These large inputs are not copied into the collection; their
  existence does not establish public availability or reproduce a model fit.
- All 44 historical notebook script calls resolve, and the checker reports no
  Python syntax errors.
- All 75 source hashes recorded at collection time match the preceding release
  candidate. The figure and collector fixes described below are subsequent edits.
- 334 research-output files are included unchanged under `paper_results/`.
  The other 747 files are installed third-party dependencies from `_deps` and
  are deliberately excluded from this research-output copy. Their environment
  and original provenance records are retained.

## Reconstructed results

`scripts/audit_saved_results.py` reproduces the following from the supplied
memberships, scores and partition labels:

| Check | Result |
|---|---|
| Fixed-cardinality Jaccard table | All 50 ordered comparisons agree with stable-ID memberships |
| Corrected mirror experiment | All seven membership counts, overlaps, Jaccard values and recovery proportions agree |
| Width-threshold lists | Counts 48/220, 9/31 and 5/31 agree with the stored lists |
| Grouped partition | 313,230 distinct IDs; zero stored joint-input groups span partitions |
| Common test population | 6,982 knots |
| Selections reconstructed from saved per-view scores | All six 31-member selections agree, including ranking within the common pool |
| Selected width and alternation summaries | All six reported profiles agree |
| Common-test overlap of the two refits | 28 shared knots; union 34; Jaccard 0.823529 |
| Canonical two-bin null summary | Four endpoints, 5,000 draws; p-value floor 1/5001 and within-family Holm 4/5001 agree |
| Diagram certificate | Saved status `FULL_VERIFIED`, two verified pairs, no recorded errors |

The partition check uses the stored exact-input group labels; it does not
recompute their equality from raw coefficients. The selected outcomes are
checked against saved outcome columns, not a new homology calculation. The
null check verifies summary bookkeeping, not the omitted permutation draws.

The former raw–Mahalanobis Jaccard discrepancy is now traceable directly to
the two saved designs: **0.512690** compares the variable-size five-view cohorts
(263 and 333 members, overlap 202); **0.621622** compares the fixed-size cohorts
(60 each, overlap 46). They are different comparisons, not rounding variants.

## Fixes and usable outputs

The previous figure code expected random seed labels 30–34, whereas the real
mirror table uses 20260830–20260834. The code now uses the full identifiers
while retaining compact axis labels. All four figures have been generated;
empirical figures use the actual supplied tables. No model refitting or
selection changes were needed.

The collector now excludes installed `_deps` trees from future collections.
This changes packaging only; the author's original uploaded collection is
preserved separately.

The exact four-endpoint canonical null table is included at
`paper_results/29b_final_joint_null/paper_facing_joint_null_results.csv`.
It can supply the supplementary table without another permutation run.

## What this closes, and what it does not

The main saved results above can now be inspected and replayed locally from
this package. No additional collection or scientific experiment is required
to perform those checks or regenerate the four figures.

This package still does not certify a fresh end-to-end reconstruction from raw
data. The historical notebook's failed exploratory geometry balance gate
remains visible (shared-group SMD 0.288980 against a 0.10 gate). The collector
covered 25 named directories, not every historical analysis or every potential
supplementary output. Large raw data and binary model checkpoints are external.
A source commit/tag has not yet been created by this preparation step.
