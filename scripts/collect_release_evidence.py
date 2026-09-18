#!/usr/bin/env python3
"""Collect existing small results and source hashes; never execute analyses."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from verify_release import RAW, FROZEN, REPO, check, sha

# Explicit run subdirectories: never select the first recursive name match.
DIRECTORIES = [
    "23_anomaly_score_baselines", "23B_size_matched_score_sensitivity",
    "23C_relative_no_khovanov_withheld", "23D_conditional_no_khovanov_withheld_n31",
    "26_bootstrap_confidence_intervals", "29b_final_joint_null",
    "31B_corrected_four_view_mirror_withheld_khovanov", "31_crossing_number_extrapolation",
    "32B_multiplier_gcm_calibration", "33_view_ablation_and_fibers",
    "33B_example_pair_extraction", "34_turaev_width_consequence_audit",
    "35_khovanov_grading_width_audit", "36_euler_cancellation_endpoint",
    "37_mirror_symmetrized_score", "38_residual_compression_sensitivity",
    "39_per_norm_matching_balance", "40_revision_audit_direct",
    "41_theta_preprocessing_direct_fixed", "42_crossing15_grid_direct",
    "43_split_tuple_audit", "44_theta_float64_verification",
    "45_exact_fiber_khovanov_audit", "46_independent_diagram_verification",
    "47_fiber_grouped_validation",
]
SUFFIXES = {".csv", ".json", ".txt", ".md", ".py", ".sage"}


def collect(root, out, data_dir=None, max_bytes=20*2**20, hash_inputs=False):
    root=root.resolve();out=out.resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    if out.is_relative_to(root):
        raise ValueError("Write the evidence ZIP outside the frozen run directory")
    if out.exists():raise FileExistsError(f"Choose a new output name: {out}")
    report,ok=check(root,data_dir)
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),
        "root":str(root),"analysis_executed":False,"fresh_full_execution_verified":False,
        "max_file_bytes":max_bytes,"files":[],"missing_directories":[],"skipped":[],"inputs":[]}
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("release_check.json",json.dumps(report,indent=2)+"\n")
        # Source hashes tie outputs to the inspected working tree, without claiming
        # that this tree generated the historical data.
        sources={str(p.relative_to(REPO)):sha(p) for base in ['src','notebooks','scripts']
                 for p in sorted((REPO/base).rglob('*.py'))}
        z.writestr("current_source_hashes.json",json.dumps(sources,indent=2)+"\n")
        for directory in DIRECTORIES:
            folder=root/directory
            if not folder.is_dir():manifest['missing_directories'].append(directory);continue
            for path in sorted(folder.rglob('*')):
                if any(part in {'_deps', '__pycache__', '.venv', 'site-packages'}
                       for part in path.relative_to(folder).parts):
                    continue
                if not path.is_file():continue
                rel=path.relative_to(root).as_posix()
                if not path.resolve().is_relative_to(root):
                    manifest['skipped'].append({'path':rel,'reason':'outside run root'});continue
                size=path.stat().st_size
                if path.suffix.lower() not in SUFFIXES or size>max_bytes:
                    manifest['skipped'].append({'path':rel,'bytes':size,'reason':'format or size limit'});continue
                payload=path.read_bytes()
                manifest['files'].append({'path':rel,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
                z.writestr('results/'+rel,payload)
        for base,names,kind in [(root,FROZEN,'frozen'),(data_dir,RAW,'raw')]:
            if base is None:continue
            for name in names:
                path=base/name;row={'kind':kind,'path':str(path),'exists':path.is_file()}
                if path.is_file():
                    row['bytes']=path.stat().st_size
                    if hash_inputs:row['sha256']=sha(path)
                manifest['inputs'].append(row)
        z.writestr('collection_manifest.json',json.dumps(manifest,indent=2)+'\n')
    return manifest,ok


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-file-mib',type=int,default=20)
    p.add_argument('--hash-inputs',action='store_true',help='Also read and hash large raw/checkpoint files; slower on Drive')
    a=p.parse_args()
    if a.max_file_mib<=0:p.error('--max-file-mib must be positive')
    result,ok=collect(a.root,a.out,a.data_dir,a.max_file_mib*2**20,a.hash_inputs)
    print('Existing result files collected:',len(result['files']))
    print('Missing result directories:',len(result['missing_directories']))
    print('No models fitted; no stored results modified.')
    print('ZIP:',a.out)
    if not ok:print('Some required inputs are absent; see release_check.json inside the ZIP.')

if __name__=='__main__':main()
