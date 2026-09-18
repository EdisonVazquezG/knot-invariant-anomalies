"""Regression checks for run isolation and evidence collection, without atlas data."""
from pathlib import Path
import importlib.util
import json
import runpy
import shutil
import sys
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from collect_release_evidence import collect


def test_sweep_isolates_child_globals_and_restores_environment(tmp_path, monkeypatch):
    # Deliberately let the child bind the parent's path variable: previously
    # exec(..., globals()) made subsequent sweep directories nest recursively.
    script = tmp_path / "stage38_residual_compression_sensitivity.py"
    shutil.copyfile(REPO / "notebooks" / script.name, script)
    child = tmp_path / "stage23_anomaly_score_baselines.py"
    child.write_text('''from pathlib import Path
import os
import numpy as np
import pandas as pd
ROOT = Path(OUTPUT_DIR)
OUT = ROOT / "23_anomaly_score_baselines"
SWEEP_ROOT = OUT
actual = int(os.environ["STAGE23_RESIDUAL_DIM"])
(OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
np.savez(OUT / "checkpoints/A_test_scores.npz", residual_sketch_dimension=[actual])
score_family_summary = pd.DataFrame([
    {"method":"raw_sse","family":"all","n":2},
    {"method":"residual_mahalanobis","family":"all","n":actual}])
score_norm_diagnostics = score_family_summary.copy()
score_selected_ids = pd.DataFrame([
    {"method":"raw_sse","family":"all","knot_id_base":"a"},
    {"method":"residual_mahalanobis","family":"all","knot_id_base":"b"}])
''')
    root=tmp_path/'run';monkeypatch.setenv('KNOT_OUTPUT_DIR',str(root))
    monkeypatch.setenv('STAGE38_SWEEP','32,64')
    monkeypatch.setenv('STAGE23_RESIDUAL_DIM','91')
    runpy.run_path(str(script),run_name='__main__')
    sweep=root/'38_residual_compression_sensitivity'
    assert (sweep/'residual_dim_32/checkpoints/A_test_scores.npz').is_file()
    assert (sweep/'residual_dim_64/checkpoints/A_test_scores.npz').is_file()
    assert not (sweep/'residual_dim_32/residual_dim_64').exists()
    assert __import__('os').environ['STAGE23_RESIDUAL_DIM']=='91'
    assert not (root/'23_anomaly_score_baselines').exists()


def test_collection_preserves_input_and_reports_missing_files(tmp_path):
    root=tmp_path/'run';folder=root/'47_fiber_grouped_validation';folder.mkdir(parents=True)
    original=b'eligible_n,selected_n\n6982,31\n';(folder/'summary.csv').write_bytes(original)
    out=tmp_path/'evidence.zip'
    report,ok=collect(root,out)
    assert not ok # missing frozen inputs are recorded, never silently accepted
    assert (folder/'summary.csv').read_bytes()==original
    with zipfile.ZipFile(out) as z:
        assert z.read('results/47_fiber_grouped_validation/summary.csv')==original
        assert not json.loads(z.read('collection_manifest.json'))['analysis_executed']
    with pytest.raises(FileExistsError):collect(root,out)


def test_collector_refuses_to_write_into_frozen_run(tmp_path):
    with pytest.raises(ValueError):collect(tmp_path,tmp_path/'new_evidence.zip')
