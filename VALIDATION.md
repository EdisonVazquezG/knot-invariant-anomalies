# Release-candidate validation — 18 September 2026

Completed in the preparation workspace:

- Editable package installation and primary CLI help passed.
- **11 tests passed:** eight existing tests and three regressions covering
  compression-sweep isolation/environment restoration and safe evidence collection.
- All existing **24 numerical package modules** are byte-identical to the upload.
- For **31 scripts**, restoring the previous default path literals gives an
  identical Python AST. Three other existing scripts have documented orchestration
  changes (baseline input lookup, compression sweep, notebook evidence audit).
- All **44 historical script calls** resolve to supplied files.
- Both working notebooks pass notebook schema and transformed cell syntax checks.
- The original recorded notebook is byte-identical to the supplied notebook,
  including its failed balance assertion.
- All **six evidence archives** match their recorded SHA-256 hashes and ZIP CRCs.
- Diagram verification and grouped-partition scripts are byte-identical to their
  respective evidence-archive copies.
- Figures 1, 2 and 4 were regenerated and visually inspected. Figures 2 and 4 use
  the author's saved tables. Figure 3's CSV-to-render path passed with synthetic
  test values; its real source CSV is not present in this upload, so its empirical
  figure was not regenerated here.

Not performed:

- Full raw-data execution on 313,230 knots, model refitting or new statistical tests.
- Independent rerun of mathematical verification in Sage/Khoca/Regina.
- Verification that every final paper table is included in a complete public
  derived-data archive.
- GitHub publication, creation of a commit/tag or DOI registration.

`docs/validation_report.json` records the preparation environment and automated
checks. `docs/REPRODUCIBILITY.md` explains the historical execution issue and the
next inexpensive collection step.

## Update after the Drive result collection

- All 1,081 collected files passed SHA-256 and byte-count checks.
- 334 research-output files are bundled; 747 installed dependency files are excluded.
- All 75 collected source hashes matched the preceding candidate before the
  subsequent figure-label and collector packaging fixes.
- Ten saved-result checks pass, including reconstruction of all six grouped-split
  selections from stored per-view scores, all 50 Jaccard comparisons, threshold
  lists and seven mirror-assignment memberships.
- All four figures are now generated from the available sources; the empirical
  mirror figure has replaced the earlier synthetic smoke test.

These additional checks do not refit models or execute the full raw-data pipeline.
See `docs/saved_results_check.json` and `docs/RELEASE_EVIDENCE_REVIEW.md`.
