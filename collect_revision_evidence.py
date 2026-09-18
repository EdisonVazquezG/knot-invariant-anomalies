#!/usr/bin/env python3
"""Collect existing paper-run evidence; never refit models or modify caches.

Colab (Drive must already be mounted):
    %run "/content/collect_revision_evidence.py"
Alternative run directory:
    %run "/content/collect_revision_evidence.py" --root "/path/to/corrected_run"

Only the explicitly named files below are read. Missing files are recorded,
not replaced with similarly named files from sensitivity runs. The resulting
ZIP is local; this script does not upload or transmit it.
"""
import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid
import zipfile

DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/"
    "processed_consensus_hardness/corrected_run_20260819"
)

# Core files establish the example, selected identities, and width checks.
CORE = {
    "33B_example_pair_extraction": [
        "best_example_pair.csv", "best_example_pair_paper_table.csv",
        "best_example_pair_nonzero_coefficients.csv",
        "ranked_checked_example_pairs.csv",
    ],
    "34_turaev_width_consequence_audit": [
        "WKh_ge4_selected_knot_lists.csv", "WKh_ge4_selected_count_audit.csv",
    ],
    "35_khovanov_grading_width_audit": [
        "stage34_threshold_knot_encoding_audit.csv",
    ],
}
SUPPORTING = {
    "33B_example_pair_extraction": ["best_example_pair_summary.txt"],
    "33_view_ablation_and_fibers": [
        "heldout_view_ablation_n31_summary.csv",
        "heldout_view_ablation_n31_selected_ids.csv",
        "heldout_view_ablation_pairwise_jaccard.csv",
        "exact_input_fiber_summary.csv",
        "same_homfly_different_theta_khovanov_examples.csv",
    ],
    "34_turaev_width_consequence_audit": ["turaev_consequence_logic.json"],
    "35_khovanov_grading_width_audit": [
        "khovanov_width_encoding_summary.csv", "grading_candidate_reproduction.csv",
        "stage35_gate.json",
    ],
    "18B_mirror_representation_robustness": ["target_free_canonicalization_audit.csv"],
    "32B_multiplier_gcm_calibration": [
        "multiplier_gcm_calibration_summary.csv",
        "multiplier_gcm_calibration_decision.csv",
    ],
    "40_revision_audit_direct": [
        "conditional_cohort_comparison.csv", "conditional_cohort_phenotypes.csv",
        "jaccard_with_cardinalities.csv", "manifest.json",
    ],
    "41_theta_preprocessing_direct_fixed": ["theta_sensitivity_summary.csv", "manifest.json"],
    "42_crossing15_grid_direct": ["crossing15_grid_summary.csv", "manifest.json"],
}


def collect(root, output_dir):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {root}. Mount Drive or pass --root.")
    output_dir = Path(output_dir).expanduser().resolve()
    records, contents = [], []
    for required, groups in ((True, CORE), (False, SUPPORTING)):
        for folder, names in groups.items():
            for name in names:
                relative = Path(folder) / name
                path = root / relative
                record = {"path": relative.as_posix(), "core": required}
                try:
                    payload = path.read_bytes()
                except OSError as exc:
                    record.update(status="unavailable", error=str(exc))
                    print(f"MISSING ({'core' if required else 'supporting'}): {relative}")
                else:
                    record.update(status="included", bytes=len(payload),
                                  sha256=hashlib.sha256(payload).hexdigest())
                    if name.endswith(".csv"):
                        try:
                            rows = csv.reader(io.StringIO(payload.decode("utf-8-sig")))
                            record["columns"] = next(rows, [])
                            record["data_rows"] = sum(1 for row in rows if row)
                        except (UnicodeError, csv.Error) as exc:
                            record["csv_inspection_error"] = str(exc)
                    contents.append((relative.as_posix(), payload))
                records.append(record)
    if not contents:
        raise FileNotFoundError("No requested files found. Verify --root; no ZIP was created.")
    now = datetime.now(timezone.utc)
    manifest = {
        "created_utc": now.isoformat(), "source_root": str(root),
        "complete_core": all(r["status"] == "included" for r in records if r["core"]),
        "note": "Existing files only. Hashes identify exported bytes, not their computational provenance.",
        "files": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"revision_evidence_{now:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, payload in contents:
            bundle.writestr(name, payload)
        bundle.writestr("collection_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"\nIncluded {len(contents)} existing files. Core complete: {manifest['complete_core']}")
    print(f"ZIP ready: {archive}")
    print("Upload this ZIP for the document/code consistency check.")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    collect(args.root, args.output_dir)


if __name__ == "__main__":
    main()
