# Reproducibility and release status

## Three distinct checks

1. **Software checks:** Python syntax, package tests, historical script references,
   evidence ZIP hashes/CRC, working notebook syntax. These can run without data.
2. **Result replay:** regenerate figures and audit selected tables from the frozen
   output tree. This requires saved results but does not require fitting models.
3. **Fresh scientific execution:** reconstruct the full paper from raw data in a
   specified environment. This has not been verified in this release preparation.

`verify_release.py` reports the first scope and, if paths are supplied, checks
whether selected inputs exist. Its success is not a claim that scope 3 passed.
`collect_release_evidence.py` reports absent folders and skipped large files.
It never substitutes a similarly named file from another sensitivity run.

## Environments

Install the package with `.[notebook,plots,dev]` for working notebooks, tables,
figures and tests. `.[ae]` adds TensorFlow for the autoencoder analyses. The
geometry analyses use SnapPy and the 15-crossing knot extension. Diagram
verification additionally requires the mathematical packages and SageMath
specified by `notebooks/stage46_independent_diagram_verification.py`; do not
assume a Python package named `sage` supplies SageMath.

The recorded grouped-split environment used Python 3.13.15, NumPy 2.1.3,
pandas 2.2.3, SciPy 1.16.3 and scikit-learn 1.6.1. These versions are historical
provenance, not a complete lockfile. Individual original manifests inside
`evidence/` are authoritative for each run. `docs/validation_report.json`
records the separate environment used to check this release candidate.

The exact diagram script and the grouped-split script are byte-identical to the
copies in their respective evidence archives. Other scripts with path-only
edits have new source hashes; do not attribute the old runs to the new hashes.

## Known historical execution issue

Recorded notebook cell 135 (zero-based) asserts that the maximum absolute norm
SMD of the chosen geometry matches is below 0.10. The saved shared-group value
is **0.288980**, so the assertion correctly fails. It is not a dependency or
syntax failure. The working notebook retains the assertion. This historical
exploratory comparison must not be described as passing its balance gate.
A full Run all check would stop here. Resolving which historical exploratory
blocks belong in a final execution route requires the final output inventory;
removing the gate or changing the cutoff is not a reproduction fix.

The full notebook also contains recovery cells, versioned alternative analyses,
optional packages and downloads. Their historical order is preserved rather
than asserted to be an independently verified execution order.

## Improvements in this candidate

- Missing direct dependencies (Parquet, joblib and threadpoolctl) are declared.
- Working paths can be configured using `KNOT_PROJECT_DIR`, `KNOT_DATA_DIR` and
  `KNOT_OUTPUT_DIR`; historical defaults remain available.
- The baseline script resolves the frozen split and phenotype by explicit paths.
- The residual-compression sweep runs each child analysis in an isolated
  namespace and restores its environment variable even on failure.
- The notebook audit finds inline analyses by their headings, not cell numbers.
- Inline conditional and relative withheld-outcome analyses are also available
  as named scripts; execute them with `%run -i` after the baseline state exists.
- Figures load their named tables and export PNG, PDF and provenance records.
- Original numerical source modules, seeds, thresholds and definitions are
  preserved. Existing result files are not overwritten by the release checker
  or collector.

## Saved-result replay now available

The Drive collection has been received and reviewed. The supplied memberships,
partition assignments, per-view scores and summaries are included under
`paper_results/`; no further collection is required for the completed checks.
Run `scripts/audit_saved_results.py --out saved_results_check.json` to repeat
them and `scripts/make_paper_figures.py --root paper_results --out figures`
to regenerate all four figures. See `RELEASE_EVIDENCE_REVIEW.md` for scope.

The original release checker and collector remain available for future runs.
They do not certify raw-data execution. The collector now excludes installed
third-party dependency trees, while retaining research outputs and provenance.

The remaining release steps are review/integration in the existing repository,
a real commit/tag, and documentation of which outputs and external inputs the
release supports. A fresh full execution can only be claimed once performed.
