# Knot invariant score analyses

Code, saved results and computational evidence for a multiview analysis of
313,230 prime knots with 3–15 crossings. The study examines how anomaly scores
and invariant representations shape selected knot populations, and relates
these selections to homological and topological properties.

## Quick start: inspect the saved results

Use Python 3.10 or later. From the repository root:

```bash
python -m pip install -e ".[plots]"
python scripts/audit_saved_results.py --out saved_results_check.json
python scripts/make_paper_figures.py --root paper_results --out figures
```

These commands check the bundled results and generate the four manuscript
figures without fitting models or downloading the raw atlas. Figure 1 is
illustrative; Figures 2–4 use the saved result tables.

The result audit checks cohort memberships and overlaps, selected width
profiles, stored partition assignments, selections reconstructed from saved
scores, recorded verification status, and file integrity. Its full scope is
documented in [the results review](docs/RELEASE_EVIDENCE_REVIEW.md).

## Repository contents

| Path | Contents |
|---|---|
| `src/consensus_hardness/` | Core analysis package |
| `scripts/` | Analysis entry points, result checks and figure generation |
| `notebooks/` | Research notebooks and analysis scripts |
| `notebooks/archive/paper_run_recorded.ipynb` | Historical notebook with saved outputs |
| `configs/` | Analysis configuration |
| `paper_results/` | 334 preserved research-output files from the reviewed collection |
| `figures/` | Manuscript figures in PDF and PNG |
| `evidence/` | Verification archives and provenance records |
| `tests/` | Software tests |
| `docs/RESULTS_MAP.md` | Mapping between scientific claims, code and outputs |
| `docs/REPRODUCIBILITY.md` | Inputs, environments and execution scope |

## Data and analysis workflows

The source coefficient and homology tables are available in
[*Knot Invariants Data*, version v1](https://doi.org/10.5281/zenodo.16631322).
The raw tables, full processed atlas and fitted checkpoints are stored
separately from this repository. The bundled outputs support the checks and
figures above; analysis workflows require the additional inputs and
environments described in [the reproducibility guide](docs/REPRODUCIBILITY.md).

For notebook and development dependencies:

```bash
python -m pip install -e ".[notebook,plots,dev]"
python scripts/verify_release.py --out release_check.json
python -m pytest -q
```

Specialized diagram and homology computations require their documented
software environments in addition to the Python dependencies above.

## Validation scope

The bundled saved-result audit passed all ten checks, and the four figures
were regenerated from the supplied inputs. This validates the checked saved
outputs rather than a complete rerun from raw data. The research notebook
preserves a failed balance diagnostic in an exploratory geometry analysis
and is not a verified uninterrupted workflow. Details and execution routes
are recorded in [the reproducibility guide](docs/REPRODUCIBILITY.md).

## Version identification

Identify the exact code version used in an analysis by its Git commit:

```bash
git rev-parse HEAD
```

When citing this repository, include its URL and the release tag or full
commit identifier. Cite the source dataset separately using its DOI above.
An archival software DOI, when available, should identify the corresponding
release.
