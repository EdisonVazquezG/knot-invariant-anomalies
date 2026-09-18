# %%
"""Stage 47.0.0 — exact-fiber grouped validation and candidate provenance.

Standalone file: no import from any other stage, no Sage/Khoca dependency.
Run in paper_run:
    %run "/content/drive/MyDrive/consensus_hardness_refactored/notebooks/stage47_fiber_grouped_validation.py"

Optional preflight: append --check-only. Cheap provenance only:
    append --provenance-only (uses a separate output directory).
Paths: KNOT_DATA_DIR / KNOT_OUTPUT_DIR, or --data-dir / --root.
Requires numpy, pandas, scipy, scikit-learn and pyarrow (existing pipeline).
Use a high-RAM runtime; old notebook X_dict is not reused or modified.

Design fixed BEFORE examining new outcomes:
* Group source-integer (Alexander,Jones,HOMFLY-PT,Theta) tuples exactly.
* Assign WHOLE groups to approx. 70/15/15; match original row-count targets.
* One fixed split seed, no seed search. No Khovanov in grouping/fitting/ranking.
* Refit BOTH original and grouped splits in float64 with fixed k=4/10/32/10,
  PCA seed=20261123; train-only scaling/PCA, validation-only 100 norm bins.
* Compare own-test top-31 profiles. Membership Jaccard for the split effect
  uses common test IDs, re-ranks EACH view within that SAME pool, then C3.
* Retain historical scores as a separate precision/implementation comparison.
* Candidate membership is retrospective evidence, NOT proof of discovery.
* No randomization p-values, no autonomous conclusion of robustness.

Resumes only manifest-matched checkpoints, never searches recursively for
ambiguous cache names, never replaces earlier stages. The review ZIP omits
large fit checkpoints; keep the complete output directory in Drive.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import zipfile
import shutil

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

VERSION = "47.0.0"
ID = "knot_id_base"
VIEWS = ("Alexander", "Jones", "HOMFLY-PT", "Theta")
SOURCES = {"Alexander": ("Alexander_upto17.csv", "A"),
           "Jones": ("Jones_upto17_MIRRORS.csv", "J"),
           "HOMFLY-PT": ("HomflyPt_upto15_MIRRORS.csv", "a"),
           "Theta": ("theta_upto15.csv", "T")}
META = {"knot_id", "knot_id_clean", ID, "number_of_crossings", "table_number",
        "is_alternating", "signature", "minimum_exponent", "maximum_exponent", "s_invariant"}
FIXED_K = dict(zip(VIEWS, (4,10,32,10)))
PARTS = ("train_idx", "val_idx", "test_idx")
EXAMPLES = ("14n00036", "14n11437", "14n08809", "15n113508")
DEFAULT_DATA = Path("/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants")

# Self-contained source-reader and exact-equality helpers used by the audited
# tuple verification. Hashes identify buckets; np.unique resolves actual rows.
def safe(s):
    return s.replace("-", "_").replace(" ", "_")

def required(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}. Check --root/--data-dir (Drive mounted?).")
    return path

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)

def atomic_npz(path, **payload):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)

def freeze(out, manifest):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(f"Inputs, code or configuration changed. Choose a new --out; preserving {out}.")
    else:
        if any(out.iterdir()):
            raise RuntimeError(f"Nonempty output has no manifest: {out}. Choose a new --out.")
        atomic_json(path, manifest)

def load_frozen(args):
    atlas_path = required(args.atlas or args.root / "17_final_paper_outputs/final_hard_regime_atlas.parquet")
    split_path = required(args.split or args.root / "07_heldout_ae_target_free/scores/heldout_ae_seed_0.npz")
    atlas = (pd.read_csv(atlas_path) if atlas_path.suffix == ".csv" else pd.read_parquet(atlas_path)).reset_index(drop=True)
    if ID not in atlas or atlas[ID].isna().any():
        raise ValueError("Missing atlas identifiers")
    ids = atlas[ID].astype(str).to_numpy(dtype=str)
    if len(set(ids)) != len(ids) or any("!" in x or x != x.strip() for x in ids):
        raise ValueError("Atlas must contain unique, normalized canonical base IDs")
    with np.load(split_path, allow_pickle=False) as q:
        split = {}
        for key in ("train_idx", "val_idx", "test_idx"):
            value = q[key]
            if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"Invalid split indices: {key}")
            split[key] = value.astype(np.int64)
        if "test_ids" in q and not np.array_equal(q["test_ids"].astype(str), ids[split["test_idx"]]):
            raise ValueError("Split test IDs do not match atlas row order")
    if any(len(v) == 0 for v in split.values()) or not np.array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(len(ids))):
        raise ValueError("Frozen indices must cover the atlas exactly once")
    # Validate notebook row order if a live aligned meta is present; never reuse X_dict.
    try:
        from IPython import get_ipython
        shell = get_ipython()
        live = shell.user_ns.get("meta") if shell else None
        if isinstance(live, pd.DataFrame) and ID in live and not np.array_equal(live[ID].astype(str).to_numpy(), ids):
            raise ValueError("Live notebook meta differs from frozen atlas order; reload the matching alignment or use a fresh session.")
    except ImportError:
        pass
    return atlas, ids, split, [atlas_path, split_path]

@contextmanager
def source_matrix(path, view, ids, work_dir, chunk_rows):
    """Stream the canonical integer coefficients to a temporary local memmap.

    Matches signature-first, unmarked-second representative selection. Missing
    numeric entries become zero as in the published pipeline, with counts logged.
    Fractional or unsafe floating coefficients are rejected, not rounded.
    """
    path = required(path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    lookup = pd.Series(np.arange(len(ids)), index=ids)
    seen = np.full(len(ids), -1, dtype=np.int8)
    columns = None
    invalid_count = 0
    with tempfile.TemporaryDirectory(prefix="knot_coefficients_", dir=work_dir) as tmp:
        X = None
        chunks = pd.read_csv(path, chunksize=chunk_rows, dtype={"knot_id": str}) if path.suffix == ".csv" else [pd.read_pickle(path)]
        for number, raw in enumerate(chunks, 1):
            if "knot_id" not in raw:
                raise ValueError(f"No knot_id in {path}")
            names = raw.knot_id.astype(str).str.strip()
            base = names.str.replace("!", "", regex=False)
            positions = base.map(lookup)
            keep = positions.notna()
            if not keep.any():
                continue
            frame = raw.loc[keep].copy()
            pos = positions[keep].to_numpy(np.int64)
            sig = pd.to_numeric(frame["signature"], errors="coerce") if "signature" in frame else pd.Series(np.nan, index=frame.index)
            priority = 2 * (sig.to_numpy() >= 0).astype(np.int8) + (~names[keep].str.contains("!", regex=False)).to_numpy(np.int8)
            cols = [str(c) for c in frame.columns if c not in META and str(c).startswith(SOURCES[view][1])]
            if columns is None:
                if not cols:
                    raise ValueError(f"No feature coordinates for {view}")
                columns = cols
                X = np.lib.format.open_memmap(Path(tmp) / "coefficients.npy", mode="w+", dtype=np.int64, shape=(len(ids), len(cols)))
            if cols != columns:
                raise ValueError("Feature columns changed while reading")
            values = np.empty((len(frame), len(cols)), dtype=np.int64)
            for j, col in enumerate(cols):
                numeric = pd.to_numeric(frame[col], errors="coerce")
                invalid_count += int(numeric.isna().sum())
                numeric = numeric.fillna(0)
                a = numeric.to_numpy()
                if np.issubdtype(a.dtype, np.integer):
                    if (a > np.iinfo(np.int64).max).any():
                        raise ValueError(f"Integer overflow in {col}")
                elif not np.isfinite(a).all() or (np.abs(a) > 2**53).any() or not np.equal(a, np.trunc(a)).all():
                    raise ValueError(f"Noninteger or unsafe floating source values in {col}")
                values[:, j] = a
            # Process the two mirror priorities independently; source duplicates
            # with the same priority must agree exactly, even across chunks.
            for pr in np.unique(priority):
                indices = np.flatnonzero(priority == pr)
                pp = pos[indices]
                if len(np.unique(pp)) != len(pp):
                    raise ValueError(f"Duplicate source IDs at equal representative priority in {path}")
                tied = seen[pp] == pr
                if tied.any() and not np.array_equal(np.asarray(X[pp[tied]]), values[indices[tied]]):
                    raise ValueError("Conflicting repeated source representatives")
                better = pr > seen[pp]
                X[pp[better]] = values[indices[better]]
                seen[pp[better]] = pr
            if number % 25 == 0:
                print(f"  {view}: scanned {number * chunk_rows:,} rows; aligned {(seen >= 0).sum():,}/{len(ids):,}", flush=True)
        if columns is None or (seen < 0).any():
            raise ValueError(f"Missing canonical source rows: {ids[seen < 0][:10].tolist()}")
        X.flush()
        info = dict(view=view, rows=len(ids), features=len(columns), source_non_numeric_or_missing_entries_filled_zero=invalid_count,
                    representative_rule="nonnegative signature, then unmarked identifier")
        try:
            yield X, columns, info
        finally:
            del X
            gc.collect()

def exact_groups(X, as_float32=False, batch=4096, hashes=None):
    """Hash to find candidates, then partition every duplicate bucket exactly."""
    if hashes is None:
        hashes = np.empty(len(X), dtype=np.uint64)
        for start in range(0, len(X), batch):
            block = np.asarray(X[start:start+batch], dtype=np.float32 if as_float32 else np.int64)
            hashes[start:start+len(block)] = pd.util.hash_pandas_object(pd.DataFrame(block), index=False).to_numpy(np.uint64)
    order = np.argsort(hashes, kind="stable")
    cuts = np.r_[0, 1 + np.flatnonzero(np.diff(hashes[order]) != 0), len(order)]
    labels = np.empty(len(X), dtype=np.int64)
    next_label = 0
    collision_buckets = 0
    for start, stop in zip(cuts[:-1], cuts[1:]):
        ix = order[start:stop]
        if len(ix) == 1:
            labels[ix] = next_label
            next_label += 1
        else:
            block = np.asarray(X[ix], dtype=np.float32 if as_float32 else np.int64)
            unique, inverse = np.unique(block, axis=0, return_inverse=True)
            collision_buckets += int(len(unique) > 1)
            labels[ix] = next_label + inverse
            next_label += len(unique)
    return labels, collision_buckets

def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = Path(os.environ.get('KNOT_DATA_DIR', str(DEFAULT_DATA)))
    p.add_argument('--data-dir', type=Path, default=data)
    p.add_argument('--root', type=Path, default=Path(os.environ.get('KNOT_OUTPUT_DIR', str(data/'processed_consensus_hardness/corrected_run_20260819'))))
    p.add_argument('--atlas', type=Path)
    p.add_argument('--split', type=Path)
    p.add_argument('--phenotype', type=Path)
    p.add_argument('--scores-dir', type=Path)
    p.add_argument('--cohorts-file', type=Path)
    p.add_argument('--stage43-dir', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--work-dir', type=Path, default=Path(tempfile.gettempdir()))
    p.add_argument('--split-seed', type=int, default=20260917)
    p.add_argument('--pca-seed', type=int, default=20261123)
    p.add_argument('--n-selected', type=int, default=31)
    p.add_argument('--expected-n', type=int, default=313230)
    p.add_argument('--chunk-rows', type=int, default=4096)
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--provenance-only', action='store_true')
    p.add_argument('--rebuild-groups', action='store_true')
    return p


def aggregate(scores):
    a = np.asarray(scores, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4 or not np.isfinite(a).all():
        raise ValueError('C3 requires four finite score columns')
    ranks = np.column_stack([rankdata(a[:,j], method='average')/len(a) for j in range(4)])
    return np.sort(ranks, axis=1)[:,1]


def top_n(score, ids, n):
    if not 0 < n <= len(ids) or len(score) != len(ids) or not np.isfinite(score).all():
        raise ValueError('Invalid selection cardinality or score')
    mask = np.zeros(len(ids), bool)
    mask[np.lexsort((ids, score))[-n:]] = True
    return mask


def calibrate(val_sse, val_norm, test_sse, test_norm, bins=100):
    edges = np.unique(np.quantile(val_norm, np.linspace(0,1,bins+1)))
    if len(edges)<2: edges=np.array([-np.inf,np.inf])
    edges[0],edges[-1]=-np.inf,np.inf
    vb=np.searchsorted(edges[1:-1],val_norm,side='right')
    tb=np.searchsorted(edges[1:-1],test_norm,side='right')
    populated=np.unique(vb)
    result=np.empty(len(test_sse))
    for b in np.unique(tb):
        ref=np.sort(val_sse[vb==b])
        if not len(ref): ref=np.sort(val_sse[vb==populated[np.argmin(np.abs(populated-b))]])
        result[tb==b]=np.searchsorted(ref,test_sse[tb==b],side='right')/(len(ref)+1)
    return result,edges


def joint_groups(per_view):
    return np.unique(np.column_stack([per_view[v] for v in VIEWS]), axis=0, return_inverse=True)[1]


def grouped_split(labels, ids, targets, seed):
    """Random whole-group allocation, weighted by rows, without endpoint access.

    Start from group order defined by smallest knot ID, shuffle once, then choose
    contiguous cut boundaries nearest the two target cumulative row counts.
    Partition sizes can differ from targets by approximately one largest fiber.
    """
    labels=np.asarray(labels,dtype=np.int64)
    groups, inverse, sizes=np.unique(labels,return_inverse=True,return_counts=True)
    if len(groups)<3: raise ValueError('Need at least three exact fibers')
    id_order=np.argsort(ids,kind='stable')
    canonical=pd.unique(inverse[id_order])
    order=np.random.default_rng(seed).permutation(canonical)
    cumulative=np.r_[0,np.cumsum(sizes[order])]
    c1=int(np.argmin(np.abs(cumulative[1:-1]-targets[0])))+1
    c1=min(c1,len(groups)-2)
    choices=np.arange(c1+1,len(groups))
    c2=int(choices[np.argmin(np.abs(cumulative[choices]-sum(targets[:2])))])
    part_group=np.empty(len(groups),dtype=np.int8)
    for code,ix in enumerate((order[:c1],order[c1:c2],order[c2:])):part_group[ix]=code
    return {key:np.flatnonzero(part_group[inverse]==code) for code,key in enumerate(PARTS)}


def split_audit(labels, split, design, family):
    labels=np.asarray(labels)
    counts={k:np.bincount(labels[ix],minlength=int(labels.max())+1) for k,ix in split.items()}
    tr,va,te=(counts[k] for k in PARTS)
    spanning=(tr>0).astype(int)+(va>0)+(te>0)
    return dict(design=design,input_family=family,n_groups=len(tr),
        groups_spanning_partitions=int((spanning>1).sum()),
        val_seen_in_train_n=int((tr[labels[split['val_idx']]]>0).sum()),
        test_seen_in_train_n=int((tr[labels[split['test_idx']]]>0).sum()),
        test_seen_in_train_or_val_n=int(((tr+va)[labels[split['test_idx']]]>0).sum()))


def validate_hash_binding(manifest, paths, strict=True):
    result=[]
    for path in paths:
        expected={h for name,h in manifest.get('input_sha256',{}).items() if Path(name).name==path.name}
        if not expected and not strict:continue
        if len(expected)!=1 or digest(path) not in expected:
            raise ValueError(f'Provenance mismatch for {path}; select matching inputs. No ambiguous cache chosen.')
        result.append(str(path))
    return result


def load_historical(args, ids, split):
    folder=args.scores_dir or args.root/'23_anomaly_score_baselines/checkpoints'
    teids=ids[split['test_idx']]
    scores=[];paths=[];identity=[]
    for view in VIEWS:
        path=required(folder/(safe(view)+'_test_scores.npz'))
        with np.load(path,allow_pickle=False) as q:
            a=np.asarray(q['test_conditional_percentile_100'],float)
            if a.shape!=(len(teids),) or not np.isfinite(a).all():raise ValueError(f'Invalid scores: {path}')
            if 'test_ids' in q and not np.array_equal(q['test_ids'].astype(str),teids):raise ValueError(f'Score ID mismatch: {path}')
            if int(q['pca_k'][0])!=FIXED_K[view]:raise ValueError(f'Unexpected PCA dimension: {view}')
            identity.append('explicit_test_ids' if 'test_ids' in q else 'frozen_split_order_and_cohort_reproduction')
            scores.append(a)
        paths.append(path)
    cp=required(args.cohorts_file or args.root/'33_view_ablation_and_fibers/heldout_view_ablation_n31_selected_ids.csv')
    df=pd.read_csv(cp,dtype={ID:str})
    if not {'selection',ID}<=set(df):raise ValueError(f'Unexpected columns: {cp}')
    members=df.loc[df.selection=='four_view_3of4',ID]
    if not len(members) or members.isna().any() or members.duplicated().any() or not set(members)<=set(teids):raise ValueError('Invalid saved four_view_3of4 cohort')
    scores=np.column_stack(scores)
    reproduced=set(teids[top_n(aggregate(scores),teids,len(members))])
    if reproduced!=set(members):raise ValueError('Historical conditional cohort does not reproduce exactly; no refitting performed')
    return scores,paths+[cp],dict(saved_cohort_n=len(members),saved_cohort_reproduced=True,score_identity_checks=dict(zip(VIEWS,identity)))


def load_outcomes(args,atlas,ids):
    path=required(args.phenotype or args.root/'20_mathematical_phenotype/complete_mathematical_phenotype_atlas.parquet')
    df=pd.read_csv(path,dtype={ID:str}) if path.suffix=='.csv' else pd.read_parquet(path)
    if ID not in df or df[ID].isna().any() or df[ID].duplicated().any():raise ValueError('Invalid outcome IDs')
    df[ID]=df[ID].astype(str)
    if not set(ids)<=set(df[ID]):raise ValueError('Missing atlas outcomes')
    df=df.set_index(ID).loc[ids]
    cols=[c for c in ['khovanov_q_minus_2t_diagonal_count','kh_diagonal_count','khovanov_diagonal_count'] if c in df]
    if not cols:raise ValueError('No stored Khovanov width column')
    width=pd.to_numeric(df[cols[0]],errors='raise').to_numpy(float)
    alt=pd.to_numeric(atlas.is_alternating,errors='raise').to_numpy(float)
    if not np.isfinite(width).all() or (width<1).any() or not np.equal(width,np.trunc(width)).all():raise ValueError('Invalid width values')
    if not np.isin(alt,[0,1]).all():raise ValueError('Invalid alternation values')
    return width,alt,path


def fit_view(X, split, k, args):
    train=split['train_idx']
    if not 0<k<min(len(train),X.shape[1]):raise ValueError('Too few rows/features for fixed PCA dimension')
    Z=np.empty((len(train),X.shape[1]),dtype=np.float64)
    for start in range(0,len(train),args.batch_size):
        block=np.asarray(X[train[start:start+args.batch_size]])
        if ((block>2**53)|(block<-(2**53))).any():raise ValueError('Coefficients exceed exact float64 range')
        Z[start:start+len(block)]=block
    scaler=StandardScaler(copy=False)
    scaler.fit_transform(Z)
    with threadpool_limits(limits=args.threads):
        pca=PCA(n_components=k,svd_solver='randomized',random_state=args.pca_seed,copy=False).fit(Z)
    del Z;gc.collect()
    evr=float(pca.explained_variance_ratio_.sum())
    if not np.isfinite(evr) or not 0<=evr<=1+1e-12:raise ValueError('Invalid float64 PCA explained variance')
    def evaluate(indices):
        errors=np.empty(len(indices));norms=np.empty(len(indices))
        for start in range(0,len(indices),args.batch_size):
            raw=np.asarray(X[indices[start:start+args.batch_size]])
            if ((raw>2**53)|(raw<-(2**53))).any():raise ValueError('Coefficients exceed exact float64 range')
            z=scaler.transform(raw.astype(np.float64))
            with threadpool_limits(limits=args.threads):
                residual=z-pca.inverse_transform(pca.transform(z))
            errors[start:start+len(z)]=np.sum(residual*residual,axis=1,dtype=np.float64)
            norms[start:start+len(z)]=np.log1p(np.sum(z*z,axis=1,dtype=np.float64))
        if not np.isfinite(errors).all() or not np.isfinite(norms).all():raise ValueError('Nonfinite evaluation')
        return errors,norms
    vs,vn=evaluate(split['val_idx']);ts,tn=evaluate(split['test_idx'])
    score,edges=calibrate(vs,vn,ts,tn)
    return dict(test_score=score,test_raw_sse=ts,test_log_norm=tn,val_raw_sse=vs,val_log_norm=vn,
        calibration_edges=edges,pca_k=np.array([k]),evr=np.array([evr]),scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,scaler_var=scaler.var_,pca_mean=pca.mean_,
        pca_components=pca.components_,pca_explained_variance=pca.explained_variance_,
        fit_dtype=np.array(['float64']))


def report_selection(scores,pool,ids,width,alt,n,design,evaluation,group_labels=None):
    poolids=ids[pool];c3=aggregate(scores);mask=top_n(c3,poolids,n);chosen=pool[mask]
    cutoff=float(c3[mask].min())
    row=dict(design=design,evaluation_pool=evaluation,eligible_n=len(pool),selected_n=n,
        mean_W_F=float(width[chosen].mean()),P_W_F_ge_3=float((width[chosen]>=3).mean()),
        P_W_F_ge_4=float((width[chosen]>=4).mean()),alternating_fraction=float(alt[chosen].mean()),
        eligible_mean_W_F=float(width[pool].mean()),eligible_alternating_fraction=float(alt[pool].mean()),
        cutoff_C3=cutoff,boundary_tied_total=int((c3==cutoff).sum()),boundary_tied_selected=int(((c3==cutoff)&mask).sum()))
    if group_labels is not None:row['selected_distinct_fibers']=int(len(np.unique(group_labels[chosen])))
    members=pd.DataFrame({ID:ids[chosen],'design':design,'evaluation_pool':evaluation,'C3':c3[mask],
                          'W_F':width[chosen],'is_alternating':alt[chosen]})
    return row,members,set(ids[chosen]),c3,mask


def compare(a,b,label,pool,n):
    return dict(comparison=label,evaluation_pool=pool,n_per_selection=n,intersection=len(a&b),union=len(a|b),
                jaccard=len(a&b)/len(a|b),comparison_uses_identical_eligible_ids=True)


def provenance_rows(ids, split, scores, n, name):
    rows=[];te=split['test_idx'];teids=ids[te];position={x:i for i,x in enumerate(teids)}
    c3=aggregate(scores);mask=top_n(c3,teids,n)
    percentile=rankdata(c3,method='average')/len(c3)
    parts={ids[i]:k.removesuffix('_idx') for k,ix in split.items() for i in ix}
    for knot in EXAMPLES:
        row={ID:knot,'design':name,'partition':parts.get(knot,'absent'),
             'eligible_in_test':knot in position,'selected_top_n':None,'C3':None,'C3_percentile_in_test':None,
             'discovery_provenance':'retrospective_exact_fiber_audit; diagram_based_certification',
             'detector_discovery_demonstrated':False}
        if knot in position:
            i=position[knot];row.update(selected_top_n=bool(mask[i]),C3=float(c3[i]),C3_percentile_in_test=float(percentile[i]),
                rank_first_possible=int((c3>c3[i]).sum()+1),rank_last_possible=int((c3>=c3[i]).sum()))
            for j,v in enumerate(VIEWS):row[safe(v)+'_conditional_percentile']=float(scores[i,j])
        rows.append(row)
    return rows


def bundle(out):
    script=out/'stage47_fiber_grouped_validation.py'
    if Path(__file__).resolve()!=script.resolve():shutil.copy2(__file__,script)
    excluded={'split_assignments.csv','all_test_scores.csv'}
    files=[p for p in sorted(out.iterdir()) if p.is_file() and p.suffix in {'.csv','.json','.txt','.py'} and p.name not in excluded]
    atomic_json(out/'review_file_hashes.json',{p.name:digest(p) for p in files if p.name!='review_file_hashes.json'})
    with zipfile.ZipFile(out/'stage47_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,p.name)
        if not any(p.name=='review_file_hashes.json' for p in files):z.write(out/'review_file_hashes.json','review_file_hashes.json')
    print('Review ZIP:',out/'stage47_review.zip',flush=True)


def main():
    args=parser().parse_args()
    if min(args.batch_size,args.chunk_rows,args.threads,args.n_selected,args.expected_n)<1:raise ValueError('Counts must be positive')
    args.root=args.root.expanduser().resolve();args.data_dir=args.data_dir.expanduser().resolve()
    out=args.out or args.root/('47_fiber_provenance_only' if args.provenance_only else '47_fiber_grouped_validation')
    out=out.expanduser().resolve()
    atlas,ids,frozen,paths=load_frozen(args)
    if len(ids)!=args.expected_n:raise ValueError(f'Expected {args.expected_n} atlas knots; got {len(ids)}. Check inputs.')
    if len(frozen['test_idx'])<args.n_selected:raise ValueError('Test smaller than requested selection')
    historical,score_paths,history_info=load_historical(args,ids,frozen)
    paths+=score_paths
    print(f'Stage {VERSION}; output: {out}',flush=True)
    print('Frozen cohort reproduced:',history_info['saved_cohort_n'],'knots. No recursive cache search.',flush=True)
    legacy_dir=args.stage43_dir or args.root/'43_split_tuple_audit'
    legacy_group=legacy_dir/'all_tuple_group_memberships.npz'
    legacy_manifest=legacy_dir/'manifest.json'
    source_paths={};cached_groups=None
    if not args.provenance_only:
        source_paths={v:required(args.data_dir/SOURCES[v][0]) for v in VIEWS}
        paths+=list(source_paths.values())
        if legacy_group.exists() and not args.rebuild_groups:
            lm=json.loads(required(legacy_manifest).read_text())
            validate_hash_binding(lm,paths[:2]+list(source_paths.values()))
            validate_hash_binding(lm,score_paths,strict=False)
            with np.load(legacy_group,allow_pickle=False) as q:
                if not np.array_equal(q['ids'].astype(str),ids):raise ValueError('Exact-group cache ID order mismatch')
                cached_groups={v:q['source_integer__'+safe(v)].copy() for v in VIEWS}
            for v,g in cached_groups.items():
                if g.shape!=(len(ids),) or not np.issubdtype(g.dtype,np.integer) or (g<0).any() or g.max()>=len(ids):raise ValueError(f'Invalid exact groups: {v}')
            paths += [legacy_group,legacy_manifest]
            print('Reusing source-hash-bound integer groups from the exact-input audit.',flush=True)
        elif not args.rebuild_groups:
            print('Full exact-group cache absent: groups will be reconstructed from original coefficients.',flush=True)
    if args.check_only:
        if not args.provenance_only:required(args.phenotype or args.root/'20_mathematical_phenotype/complete_mathematical_phenotype_atlas.parquet')
        print('PREFLIGHT PASSED. Full mode fits 8 PCA models; checkpointed models are reused. Run without --check-only.')
        return
    # Outcomes are read for summaries, but never passed to grouping, splitting,
    # fitting, calibration or selection functions.
    width,alt,phenotype_path=load_outcomes(args,atlas,ids);paths+=[phenotype_path,Path(__file__).resolve()]
    manifest=dict(version=VERSION,mode='provenance_only' if args.provenance_only else 'grouped_validation',
        split_seed=args.split_seed,pca_seed=args.pca_seed,k=FIXED_K,n_selected=args.n_selected,norm_bins=100,
        fit_dtype='float64',selection_rule='third_largest_test_empirical_percentile; descending_lexicographic_ID_ties',
        grouped_split_rule='randomized whole exact fibers; nearest cumulative row-count cuts',
        source_group_mode='reused_hash_bound_cache' if cached_groups is not None else 'exact_source_reconstruction',
        batch_size=args.batch_size,chunk_rows=args.chunk_rows,threads=args.threads,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        input_sha256={str(p.resolve()):digest(p) for p in paths},historical_validation=history_info)
    freeze(out,manifest)
    status=dict(version=VERSION,status='RUNNING',detector_discovery_demonstrated=False)
    atomic_json(out/'run_summary.json',status)
    prov=provenance_rows(ids,frozen,historical,args.n_selected,'historical_random')
    pd.DataFrame(prov).to_csv(out/'example_provenance.csv',index=False)
    print('Historical candidate provenance saved; non-test scores remain unavailable, not zero.',flush=True)
    if args.provenance_only:
        status.update(status='PROVENANCE_COMPLETE',grouped_split_run=False)
        atomic_json(out/'run_summary.json',status);bundle(out);return
    cache=out/'checkpoints';cache.mkdir(exist_ok=True)
    per_view={} if cached_groups is None else cached_groups
    if cached_groups is None:
        for v in VIEWS:
            gp=cache/(safe(v)+'_exact_groups.npz')
            if gp.exists():
                with np.load(gp,allow_pickle=False) as q:
                    if not np.array_equal(q['ids'].astype(str),ids):raise ValueError('Cached group IDs differ')
                    per_view[v]=q['groups'].copy()
                continue
            print('Exact grouping:',v,flush=True)
            with source_matrix(source_paths[v],v,ids,args.work_dir,args.chunk_rows) as (X,cols,info):
                groups,collisions=exact_groups(X,batch=args.chunk_rows)
                per_view[v]=groups
                info['hash_collision_buckets_resolved']=collisions
                atomic_json(out/(safe(v)+'_source_audit.json'),info)
                atomic_npz(gp,ids=ids,groups=groups)
    joint=joint_groups(per_view)
    targets=[len(frozen[k]) for k in PARTS]
    grouped=grouped_split(joint,ids,targets,args.split_seed)
    for split in [frozen,grouped]:
        if not np.array_equal(np.sort(np.concatenate([split[k] for k in PARTS])),np.arange(len(ids))):raise ValueError('Partition coverage error')
        if min(len(split[k]) for k in PARTS)<=max(FIXED_K.values()):raise ValueError('Partition too small')
    audits=[split_audit(g,sp,design,family) for design,sp in [('frozen',frozen),('grouped',grouped)]
            for family,g in [*per_view.items(),('joint_four_view',joint)]]
    joint_row=next(r for r in audits if r['design']=='grouped' and r['input_family']=='joint_four_view')
    if joint_row['groups_spanning_partitions']!=0:raise AssertionError('Grouped partition leaks exact joint inputs')
    pd.DataFrame(audits).to_csv(out/'split_input_overlap.csv',index=False)
    summary=[];assign=pd.DataFrame({ID:ids,'exact_joint_fiber':joint})
    for name,sp in [('frozen',frozen),('grouped',grouped)]:
        assignment=np.empty(len(ids),dtype='U10')
        for target,key in zip(targets,PARTS):
            assignment[sp[key]]=key.removesuffix('_idx')
            summary.append(dict(design=name,partition=key.removesuffix('_idx'),n=len(sp[key]),target_n=target,
                fraction=len(sp[key])/len(ids),distinct_fibers=len(np.unique(joint[sp[key]]))))
        assign[name+'_partition']=assignment
    assign.to_csv(out/'split_assignments.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'split_sizes.csv',index=False)
    atomic_npz(cache/'grouped_split.npz',ids=ids,joint_groups=joint,**grouped)
    common=np.intersect1d(frozen['test_idx'],grouped['test_idx'])
    if len(common)<args.n_selected:raise ValueError('Common test too small; do not tune seed from outcomes. Specify an independent design.')
    # Each new fit uses the same implementation/precision: only partitions differ.
    designs={'frozen_refit_float64':frozen,'grouped_refit_float64':grouped}
    scores={name:np.empty((len(sp['test_idx']),4)) for name,sp in designs.items()}
    numeric=[]
    for j,v in enumerate(VIEWS):
        missing=[name for name in designs if not (cache/(name+'__'+safe(v)+'.npz')).exists()]
        if missing:
            print('Loading original coefficients for refit:',v,flush=True)
            with source_matrix(source_paths[v],v,ids,args.work_dir,args.chunk_rows) as (X,cols,info):
                atomic_json(out/(safe(v)+'_source_audit.json'),info)
                for name in missing:
                    print(f'  Fitting {name}: k={FIXED_K[v]}, shape={len(designs[name]["train_idx"]):,} x {X.shape[1]}',flush=True)
                    r=fit_view(X,designs[name],FIXED_K[v],args)
                    atomic_npz(cache/(name+'__'+safe(v)+'.npz'),test_ids=ids[designs[name]['test_idx']],feature_names=np.asarray(cols,str),**r)
        for name,sp in designs.items():
            with np.load(cache/(name+'__'+safe(v)+'.npz'),allow_pickle=False) as q:
                if not np.array_equal(q['test_ids'].astype(str),ids[sp['test_idx']]) or str(q['fit_dtype'][0])!='float64':raise ValueError('Invalid fitted checkpoint')
                scores[name][:,j]=q['test_score']
                numeric.append(dict(design=name,view=v,k=int(q['pca_k'][0]),explained_variance_ratio=float(q['evr'][0]),
                    raw_sse_min=float(q['test_raw_sse'].min()),raw_sse_finite=bool(np.isfinite(q['test_raw_sse']).all()),
                    conditional_fraction_ge_099=float((q['test_score']>=.99).mean()),
                    score_lognorm_spearman=float(spearmanr(q['test_score'],q['test_log_norm']).statistic)))
    pd.DataFrame(numeric).to_csv(out/'numeric_and_calibration_checks.csv',index=False)
    all_designs={'historical_random':(frozen,historical),**{name:(sp,scores[name]) for name,sp in designs.items()}}
    rows=[];members=[];sets={};score_exports=[]
    for name,(sp,sc) in all_designs.items():
        for scope,pool in [('own_test',sp['test_idx']),('common_test',common)]:
            # Re-rank per view WITHIN the same common pool, not ranks inherited
            # from different tests. Both models held these knots out of fitting.
            lookup=pd.Index(sp['test_idx']).get_indexer(pool)
            if (lookup<0).any():raise AssertionError('Evaluation pool is not held out')
            row,member,selected,c3,mask=report_selection(sc[lookup],pool,ids,width,alt,args.n_selected,name,scope,joint)
            rows.append(row);members.append(member);sets[(name,scope)]=selected
            if scope=='own_test':
                export=pd.DataFrame({ID:ids[pool],'design':name,'C3':c3,'selected':mask})
                for j,v in enumerate(VIEWS):export[safe(v)+'_conditional_percentile']=sc[:,j]
                score_exports.append(export)
        if name!='historical_random':prov+=provenance_rows(ids,sp,sc,args.n_selected,name)
    pd.DataFrame(rows).to_csv(out/'headline_profiles.csv',index=False)
    pd.concat(members,ignore_index=True).to_csv(out/'selected_ids.csv',index=False)
    pd.concat(score_exports,ignore_index=True).to_csv(out/'all_test_scores.csv',index=False)
    pd.DataFrame(prov).to_csv(out/'example_provenance.csv',index=False)
    comparisons=[compare(sets[('historical_random','own_test')],sets[('frozen_refit_float64','own_test')],
            'historical_vs_same_split_refit_precision_implementation','original_test',args.n_selected),
        compare(sets[('frozen_refit_float64','common_test')],sets[('grouped_refit_float64','common_test')],
            'split_effect_same_implementation_common_test','common_test',args.n_selected)]
    pd.DataFrame(comparisons).to_csv(out/'membership_comparisons.csv',index=False)
    primary=pd.DataFrame(rows)
    primary=primary[(primary.evaluation_pool=='common_test') & (primary.design!='historical_random')].copy()
    primary.to_csv(out/'main_text_comparison.csv',index=False)
    (out/'interpretation.txt').write_text(
        'MAIN TEXT: main_text_comparison.csv compares two float64 refits on identical held-out IDs.\n'
        'Use membership_comparisons.csv split_effect row with this table. All endpoint contrasts are descriptive.\n'
        'SI: headline_profiles.csv also reports each full test and the historical scores separately.\n'
        'The common-test pool is smaller, so top-31 is a different selection fraction than each full test.\n'
        'Split grouping uses only exact four-polynomial inputs. Models remain per-view: component repetition can persist.\n'
        'Fixed PCA dimensions are inherited from the benchmark; this does not reselect k from training.\n'
        'This is one additional split within the same atlas, not independent external replication.\n'
        'Historical/refitted disagreement includes precision and implementation/environment, not partition changes.\n'
        'Pair provenance is retrospective and limited to the saved four-view conditional benchmark.\n'
        'Training/validation examples have no frozen test score. Blank means ineligible/unavailable, not zero.\n'
        'The original exact-fiber audit identified the pairs; diagram computations certified them.\n'
        'High rank or membership found now does not demonstrate detector-led discovery.\n'
        'This stage does not recompute a width-adjusted randomization p-value, matched geometry, or autoencoders.\n'
        'Keep checkpoints, split_assignments.csv and all_test_scores.csv in Drive; omitted from review ZIP.\n')
    sizes=np.bincount(joint)
    status.update(status='COMPLETE',grouped_split_run=True,atlas_n=len(ids),common_test_n=len(common),
        nontrivial_fibers=int((sizes>1).sum()),knots_in_nontrivial_fibers=int(sizes[sizes>1].sum()),
        largest_fiber=int(sizes.max()),joint_groups_spanning_partitions=0,n_selected=args.n_selected,
        existing_stages_modified=False,primary_comparison='two matched float64 refits on common test',
        inference='descriptive; no p-values or automatic robustness verdict')
    atomic_json(out/'run_summary.json',status)
    print(primary.to_string(index=False),flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)
    bundle(out);print('STAGE 47 COMPLETE',flush=True)


if __name__=='__main__':
    main()
