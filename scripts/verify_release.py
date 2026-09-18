#!/usr/bin/env python3
"""Check source completeness and supplied evidence without fitting models."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from importlib import metadata
from pathlib import Path
import re
import shlex
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
RAW = ["Alexander_upto17.csv", "Jones_upto17_MIRRORS.csv", "HomflyPt_upto15_MIRRORS.csv", "theta_upto15.csv", "even_KH_upto17.pkl"]
FROZEN = [
    "17_final_paper_outputs/final_hard_regime_atlas.parquet",
    "07_heldout_ae_target_free/scores/heldout_ae_seed_0.npz",
    "20_mathematical_phenotype/complete_mathematical_phenotype_atlas.parquet",
    "23_anomaly_score_baselines/checkpoints/Alexander_test_scores.npz",
    "23_anomaly_score_baselines/checkpoints/Jones_test_scores.npz",
    "23_anomaly_score_baselines/checkpoints/HOMFLY_PT_test_scores.npz",
    "23_anomaly_score_baselines/checkpoints/Theta_test_scores.npz",
    "33_view_ablation_and_fibers/heldout_view_ablation_n31_selected_ids.csv",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def notebook_calls(nb, repo=REPO):
    rows = []
    for i,c in enumerate(nb["cells"]):
        source = "".join(c.get("source", []))
        for line in re.findall(r"^%run\s+([^\n]+)", source, re.M):
            args = shlex.split(line)
            names = [Path(v).name for v in args if v.endswith(".py")]
            for name in names:
                paths = [repo/name, repo/"notebooks"/name]
                rows.append({"cell":i,"script":name,"present":any(p.is_file() for p in paths)})
    return rows


def check(root=None, data_dir=None):
    syntax = []
    for folder in ["src", "notebooks", "scripts", "tests"]:
        for p in sorted((REPO/folder).rglob("*.py")):
            try:
                ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
            except (SyntaxError, UnicodeError) as e:
                syntax.append({"file":str(p.relative_to(REPO)),"error":str(e)})
    recorded = json.loads((REPO/"notebooks/archive/paper_run_recorded.ipynb").read_text())
    calls = notebook_calls(recorded)
    errors = [{"cell":i,"name":o.get("ename"),"message":o.get("evalue")}
              for i,c in enumerate(recorded["cells"]) for o in c.get("outputs",[])
              if o.get("output_type")=="error"]
    archive_checks = []
    for entry in json.loads((REPO/"evidence/archives.json").read_text()):
        path = REPO/"evidence"/entry["file"]
        ok = path.is_file() and sha(path)==entry["sha256"]
        corrupt_member = None
        if ok:
            with zipfile.ZipFile(path) as z:
                corrupt_member = z.testzip()
            ok = corrupt_member is None
        archive_checks.append({"file":entry["file"],"hash_and_crc_ok":ok,"corrupt_member":corrupt_member})
    versions = {}
    for name in ["numpy","pandas","scipy","scikit-learn","statsmodels","pyarrow","joblib","threadpoolctl","matplotlib","ipython","pytest"]:
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]=None
    raw = [{"path":str(data_dir/name),"exists":(data_dir/name).is_file()} for name in RAW] if data_dir else []
    frozen = [{"path":str(root/name),"exists":(root/name).is_file()} for name in FROZEN] if root else []
    software_ok = not syntax and all(r["present"] for r in calls) and all(r["hash_and_crc_ok"] for r in archive_checks)
    inputs_ok = all(r["exists"] for r in raw+frozen)
    report = dict(status="SOFTWARE_CHECKS_PASS" if software_ok else "SOFTWARE_CHECKS_FAIL",
        syntax_errors=syntax, recorded_notebook_script_calls=calls,
        recorded_notebook_errors=errors, evidence_archives=archive_checks,
        installed_versions=versions, python=sys.version, raw_inputs=raw, frozen_inputs=frozen,
        requested_inputs_present=inputs_ok,
        fresh_full_execution_verified=False,
        scope="Static source checks, archive integrity and optional input existence. Saved outputs are historical evidence, not a new run.")
    return report, software_ok and inputs_ok


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,help="Existing processed run; read only")
    p.add_argument("--data-dir",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();report,ok=check(a.root,a.data_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+"\n")
    print(report["status"])
    print(f"Recorded script calls: {len(report['recorded_notebook_script_calls'])}; missing: {sum(not r['present'] for r in report['recorded_notebook_script_calls'])}")
    print(f"Historical error outputs retained: {len(report['recorded_notebook_errors'])}")
    print(f"Evidence archives checked: {len(report['evidence_archives'])}")
    if not report['requested_inputs_present']:
        for entry in report['raw_inputs']+report['frozen_inputs']:
            if not entry['exists']:print('MISSING:',entry['path'])
    print("Fresh full execution: not verified by this check")
    print("Report:",a.out)
    raise SystemExit(0 if ok else 2)

if __name__=="__main__":main()
