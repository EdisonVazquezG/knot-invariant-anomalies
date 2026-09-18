# %%
"""Stage 45: exact (HOMFLY-PT, Theta) fibers versus stored Khovanov data.

%run "/path/to/notebooks/stage45_exact_fiber_khovanov_audit.py"
Keep the original stage43_split_tuple_audit.py alongside this script.
Requires Stage 43's full NPZ checkpoint retained in Drive, not its review ZIP.
No training, randomized test, or new partition. Default scopes: test and atlas.
All candidate input equalities are rechecked against original integer sources.
Differences in F_ are compared both directly and modulo (q,t)->(-q,-t).
Torsion comparisons are archival diagnostics only, not mirror-normalized claims.
"""
from __future__ import annotations

import gc
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import re
import zipfile

import numpy as np
import pandas as pd

HELPER_PATH = Path(__file__).resolve().with_name("stage43_split_tuple_audit.py")
if not HELPER_PATH.is_file():
    raise FileNotFoundError("Put stage43_split_tuple_audit.py in the same folder as Stage 45.")
spec = importlib.util.spec_from_file_location("stage45_stage43_helpers", HELPER_PATH)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
ID = h.ID
FAMILIES = {
    "HOMFLY_PT_plus_Theta": ("HOMFLY-PT", "Theta"),
    "four_view_noKh": h.VIEWS[:4],
}
WIDTH_NAMES = ("khovanov_q_minus_2t_diagonal_count", "kh_diagonal_count", "khovanov_diagonal_count")
PAIR_COLUMNS = ["scope", "input_family", "tuple_group", "knot_a", "knot_b",
                "split_a", "split_b", "same_four_view_inputs", "W_F_a", "W_F_b",
                "width_diff", "F_direct_diff", "F_diff_mod_reflection",
                "F_reflection_only", "stored_torsion_direct_diff", "torsion_available"]
MEMBER_COLUMNS = ["scope", "input_family", "tuple_group", ID, "split", "W_F",
                  "F_rank", "F_support", "stored_union_diagonal_count",
                  "F_tuple_group", "F_reflection_orbit", "stored_torsion_tuple_group"]


def partition_equal(a, b):
    """Equivalence of partitions, independently of arbitrary numeric labels."""
    pairs = np.unique(np.column_stack([a, b]), axis=0)
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))


def validate_labels(a, n, name):
    if a.ndim != 1 or len(a) != n or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"Invalid cached labels for {name}")
    if (a < 0).any() or (a >= n).any():
        raise ValueError(f"Cached labels out of range for {name}")
    return a.astype(np.int64)


def repeated_groups(labels, positions):
    frame = pd.DataFrame({"g": labels[positions], "pos": positions})
    for g, part in frame.groupby("g", sort=True):
        if len(part) > 1:
            yield int(g), part.pos.to_numpy(np.int64)


def recorded_hash(manifest, path):
    matches = [v for k, v in manifest["input_sha256"].items() if Path(k).name == path.name]
    if len(matches) != 1:
        raise ValueError(f"Stage 43 provenance is missing or ambiguous for {path.name}")
    return matches[0]


def strict_integer_matrix(frame, columns):
    result = np.empty((len(frame), len(columns)), dtype=np.int64)
    for j, name in enumerate(columns):
        a = pd.to_numeric(frame[name], errors="raise").to_numpy()
        if pd.isna(a).any():
            raise ValueError(f"Missing rank in {name}; do not interpret a missing entry as zero.")
        if np.issubdtype(a.dtype, np.integer):
            if (a < 0).any() or (a > np.iinfo(np.int64).max).any():
                raise ValueError(f"Invalid nonnegative rank in {name}")
        elif (not np.isfinite(a).all() or (a < 0).any() or (a > 2**53).any()
              or not np.equal(a, np.trunc(a)).all()):
            raise ValueError(f"Noninteger or unsafe rank in {name}")
        result[:, j] = a
    return result


def parse_grading(name):
    prefix = next((p for p in ("F_", "T2_", "T4_") if name.startswith(p)), None)
    if prefix is None:
        raise ValueError(f"Unknown coordinate family: {name}")
    tail = name[len(prefix):]
    q = re.search(r"(?:^|_)q=?(-?\d+)", tail, re.I)
    t = re.search(r"(?:^|_)t=?(-?\d+)", tail, re.I)
    if q and t:
        return prefix[:-1], int(q.group(1)), int(t.group(1))
    numbers = re.findall(r"-?\d+", tail)
    if len(numbers) != 2:
        raise ValueError(f"Cannot parse exactly two gradings in {name}")
    # Audited Stage 35 convention: index1=q, index2=t, not the reverse.
    return prefix[:-1], int(numbers[0]), int(numbers[1])


def read_khovanov(path, ids):
    print("Reading original Khovanov table (high-RAM runtime recommended).", flush=True)
    raw = pd.read_csv(path, dtype={"knot_id": str}) if path.suffix == ".csv" else pd.read_pickle(path)
    if "knot_id" not in raw:
        raise ValueError("Khovanov table has no knot_id")
    names = raw.knot_id.astype(str).str.strip()
    base = names.str.replace("!", "", regex=False)
    mask = base.isin(ids)
    frame = raw.loc[mask].copy()
    frame["_base45"] = base[mask].to_numpy()
    nm = names[mask]
    sig = pd.to_numeric(frame["signature"], errors="coerce") if "signature" in frame else pd.Series(np.nan, index=frame.index)
    frame["_priority45"] = 2 * (sig.to_numpy() >= 0).astype(np.int8) + (~nm.str.contains("!", regex=False)).to_numpy(np.int8)
    del raw, names, base, mask, nm
    gc.collect()
    if frame.duplicated(["_base45", "_priority45"]).any():
        raise ValueError("Duplicate Khovanov representatives at equal priority")
    frame = frame.sort_values("_priority45", kind="stable").drop_duplicates("_base45", keep="last").set_index("_base45")
    if not set(ids) <= set(frame.index):
        raise ValueError("Some candidate knots have no archived Khovanov row")
    frame = frame.loc[ids]
    columns = [str(c) for c in frame if str(c).startswith(("F_", "T2_", "T4_"))]
    other = [str(c) for c in frame if re.match(r"T\d+_", str(c)) and c not in columns]
    if other:
        raise ValueError(f"Additional torsion families require an explicit extension: {other[:5]}")
    if not any(c.startswith("F_") for c in columns):
        raise ValueError("No F_ coordinates")
    X = strict_integer_matrix(frame, columns)
    selected_source_ids = frame.knot_id.astype(str).tolist()
    del frame
    gc.collect()
    grading = [parse_grading(c) for c in columns]
    if len(set(grading)) != len(grading):
        raise ValueError("Duplicate parsed coordinates")
    return X, columns, grading, selected_source_ids


def label_keys(keys):
    lookup, labels = {}, []
    for key in keys:
        if key not in lookup:
            lookup[key] = len(lookup)
        labels.append(lookup[key])
    return np.asarray(labels, dtype=np.int64)


def kh_metrics(X, columns, grading):
    free = np.array([f == "F" for f, q, t in grading])
    torsion = ~free
    diagonals = np.array([q - 2*t for f, q, t in grading])
    free_keys, orbit_keys = [], []
    widths, union_widths, ranks, supports = [], [], [], []
    free_indices = np.flatnonzero(free)
    for row in X:
        nz = free_indices[row[free_indices] != 0]
        key = tuple(sorted((grading[j][1], grading[j][2], int(row[j])) for j in nz))
        reflected = tuple(sorted((-q, -t, v) for q, t, v in key))
        free_keys.append(key)
        orbit_keys.append(min(key, reflected))
        widths.append(len({q - 2*t for q, t, v in key}))
        union_widths.append(len(set(diagonals[row != 0].tolist())))
        ranks.append(sum(v for q, t, v in key))
        supports.append(len(nz))
    if torsion.any():
        _, tlabels = np.unique(X[:, torsion], axis=0, return_inverse=True)
    else:
        tlabels = np.zeros(len(X), dtype=np.int64)
    return pd.DataFrame({"W_F": widths, "F_rank": ranks, "F_support": supports,
                         "stored_union_diagonal_count": union_widths,
                         "F_tuple_group": label_keys(free_keys),
                         "F_reflection_orbit": label_keys(orbit_keys),
                         "stored_torsion_tuple_group": tlabels}), bool(torsion.any())


def write_csv(out, name, frame):
    temp = out / (name + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(out / name)


def main():
    parser = h.parser(__doc__)
    parser.add_argument("--stage43-dir", type=Path)
    parser.add_argument("--phenotype", type=Path)
    parser.add_argument("--scope", choices=["both", "test", "atlas"], default="both")
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()
    if args.chunk_rows < 1 or args.max_examples < 0:
        parser.error("Require positive --chunk-rows and nonnegative --max-examples")
    args.root = args.root.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    s43 = args.stage43_dir or args.root / "43_split_tuple_audit"
    groups_path = h.required(s43 / "all_tuple_group_memberships.npz")
    manifest_path = h.required(s43 / "manifest.json")
    prior = json.loads(manifest_path.read_text())
    if prior.get("stage") != 43 or prior.get("version") != 1:
        raise ValueError("Expected the verified Stage 43 version-1 manifest")
    atlas, ids, split, frozen_paths = h.load_frozen(args)
    phenotype_path = h.required(args.phenotype or args.root / "20_mathematical_phenotype/complete_mathematical_phenotype_atlas.parquet")
    sources = {v: h.required(args.data_dir / h.SOURCES[v][0]) for v in h.VIEWS}
    with np.load(groups_path, allow_pickle=False) as z:
        if not np.array_equal(z["ids"].astype(str), ids):
            raise ValueError("Stage 43 group IDs differ from the frozen atlas")
        for k, ix in split.items():
            if not np.array_equal(z[k], ix):
                raise ValueError(f"Stage 43 split differs: {k}")
        labels = {v: validate_labels(z["source_integer__" + h.safe(v)], len(ids), v) for v in h.VIEWS}
        families = {v: validate_labels(z["source_integer__" + h.safe(v)], len(ids), v) for v in FAMILIES}
    for name, views in FAMILIES.items():
        _, rebuilt = np.unique(np.column_stack([labels[v] for v in views]), axis=0, return_inverse=True)
        if not partition_equal(families[name], rebuilt):
            raise ValueError(f"Inconsistent joint/per-view Stage 43 groups: {name}")
    scopes = {"test": split["test_idx"], "atlas": np.arange(len(ids))}
    if args.scope != "both":
        scopes = {args.scope: scopes[args.scope]}
    scan_positions = scopes.get("atlas", scopes.get("test"))
    groups = list(repeated_groups(families["HOMFLY_PT_plus_Theta"], scan_positions))
    positions = np.sort(np.concatenate([ix for _, ix in groups])) if groups else np.array([], dtype=np.int64)
    target_ids = ids[positions]
    print(f"Atlas: {len(ids):,}; candidate fibers: {len(groups):,}; candidate knots: {len(positions):,}; scope: {args.scope}")
    print("Preflight uses exact paths; original coefficients will be checked before reporting any candidate.")
    if args.check_only:
        print("Preflight complete. Run without --check-only for source hashing and the audit.")
        return
    # Verify the frozen sources against Stage 43, even if Drive shortcut paths moved.
    hashes = {}
    for f in frozen_paths + list(sources.values()) + [HELPER_PATH]:
        print("Verifying Stage 43 source hash:", f.name, flush=True)
        value = h.digest(f)
        if value != recorded_hash(prior, f):
            raise ValueError(f"Source changed since Stage 43: {f}. Use the original frozen files.")
        hashes[str(f.resolve())] = value
    baseline = args.root / "33_view_ablation_and_fibers/exact_input_fiber_summary.csv"
    extra_paths = [groups_path, manifest_path, phenotype_path, Path(__file__)]
    if baseline.exists():
        extra_paths.append(baseline)
    for f in extra_paths:
        hashes[str(f.resolve())] = h.digest(f)
    out = (args.out or args.root / "45_exact_fiber_khovanov_audit").expanduser().resolve()
    manifest = dict(stage=45, version=1, scope=args.scope, max_examples=args.max_examples,
                    chunk_rows=args.chunk_rows, python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, input_sha256=hashes,
                    description="Exact source tuples; rational/free homology modulo grading reflection; stored torsion diagnostic only")
    h.freeze(out, manifest)
    matrices, input_columns, audits = {}, {}, []
    if len(positions):
        for view in h.VIEWS[:4]:
            print("Checking candidate source inputs:", view, flush=True)
            with h.source_matrix(sources[view], view, target_ids, args.work_dir, args.chunk_rows) as (X, cols, info):
                if info["source_non_numeric_or_missing_entries_filled_zero"]:
                    raise ValueError(f"Missing/non-numeric coefficients in {view}; cannot certify exact input equality")
                fresh, collisions = h.exact_groups(X, batch=args.chunk_rows)
                if not partition_equal(labels[view][positions], fresh):
                    raise ValueError(f"Stage 43/source partition mismatch for {view}")
                matrices[view] = np.asarray(X).copy()
                input_columns[view] = cols
                info["stage43_partition_reproduced"] = True
                info["hash_collision_buckets_resolved"] = int(collisions)
                audits.append(info)
        Xkh, khcols, grading, kh_source_ids = read_khovanov(sources["Khovanov"], target_ids)
        metrics, torsion_available = kh_metrics(Xkh, khcols, grading)
        if not partition_equal(labels["Khovanov"][positions], metrics.F_tuple_group.to_numpy()):
            raise ValueError("Stage 43 free-Khovanov equality groups do not match the source ranks")
        ph = pd.read_csv(phenotype_path, dtype={ID: str}) if phenotype_path.suffix == ".csv" else pd.read_parquet(phenotype_path)
        if ID not in ph or ph[ID].isna().any() or ph[ID].astype(str).duplicated().any():
            raise ValueError("Invalid phenotype identifiers")
        ph[ID] = ph[ID].astype(str)
        col = next((c for c in WIDTH_NAMES if c in ph), None)
        if col is None:
            raise ValueError("No frozen width column in phenotype")
        expected = pd.to_numeric(ph.set_index(ID).reindex(target_ids)[col], errors="raise").to_numpy()
        if not np.array_equal(expected, metrics.W_F.to_numpy()):
            raise ValueError("Recomputed q-2t occupied-diagonal counts differ from frozen phenotype")
        metrics.insert(0, ID, target_ids)
        metrics["source_khovanov_id"] = kh_source_ids
        write_csv(out, "candidate_knot_homology.csv", metrics)
        h.atomic_npz(out / "verified_candidate_coefficients.npz", ids=target_ids, Khovanov=Xkh,
                     Khovanov_columns=np.array(khcols, dtype=str),
                     **{h.safe(k): v for k, v in matrices.items()},
                     **{h.safe(k)+"_columns": np.array(v, dtype=str) for k, v in input_columns.items()})
        h.atomic_json(out / "source_verification.json", dict(candidate_knots=len(positions),
                      input_views=audits, free_groups_reproduced=True, widths_match_frozen=True,
                      width_convention="occupied index1-2*index2 diagonals; index1=q, index2=t",
                      torsion_families=sorted({f for f, q, t in grading if f != "F"}),
                      representation_rule="nonnegative signature, then unmarked ID",
                      independent_homology_recomputation=False))
    else:
        Xkh, khcols, grading = np.empty((0, 0), dtype=np.int64), [], []
        metrics = pd.DataFrame(columns=MEMBER_COLUMNS[3:] + ["source_khovanov_id"])
        torsion_available = False
        write_csv(out, "candidate_knot_homology.csv", metrics)
        h.atomic_json(out / "source_verification.json", dict(candidate_knots=0, status="no_nontrivial_input_fibers", independent_homology_recomputation=False))
    loc = {int(global_pos): local_pos for local_pos, global_pos in enumerate(positions)}
    split_names = np.empty(len(ids), dtype="U10")
    for k, ix in split.items():
        split_names[ix] = k.replace("_idx", "")
    pair_records, member_records, group_records, summary_records = [], [], [], []
    for scope, ix in scopes.items():
        for family, flabels in families.items():
            subset_pairs, subset_groups = [], []
            for group, members in repeated_groups(flabels, ix):
                members = sorted(members, key=lambda j: ids[j])
                local = [loc[int(j)] for j in members]
                for j, l in zip(members, local):
                    r = metrics.iloc[l]
                    member_records.append(dict(scope=scope, input_family=family, tuple_group=group,
                        **{ID: ids[j]}, split=split_names[j], **{k: int(r[k]) for k in MEMBER_COLUMNS[5:]}))
                gpairs = []
                for (a, la), (b, lb) in itertools.combinations(zip(members, local), 2):
                    ra, rb = metrics.iloc[la], metrics.iloc[lb]
                    direct = bool(ra.F_tuple_group != rb.F_tuple_group)
                    orbit_diff = bool(ra.F_reflection_orbit != rb.F_reflection_orbit)
                    row = dict(scope=scope, input_family=family, tuple_group=group,
                        knot_a=ids[a], knot_b=ids[b], split_a=split_names[a], split_b=split_names[b],
                        same_four_view_inputs=bool(families["four_view_noKh"][a] == families["four_view_noKh"][b]),
                        W_F_a=int(ra.W_F), W_F_b=int(rb.W_F), width_diff=bool(ra.W_F != rb.W_F),
                        F_direct_diff=direct, F_diff_mod_reflection=orbit_diff,
                        F_reflection_only=direct and not orbit_diff,
                        stored_torsion_direct_diff=bool(ra.stored_torsion_tuple_group != rb.stored_torsion_tuple_group),
                        torsion_available=torsion_available)
                    if row["width_diff"] and not orbit_diff:
                        raise AssertionError("Reflection-invariant width cannot differ within one reflection orbit")
                    gpairs.append(row)
                pair_records.extend(gpairs)
                subset_pairs.extend(gpairs)
                gr = dict(scope=scope, input_family=family, tuple_group=group, n_knots=len(members),
                          n_pairs=len(gpairs), n_width_values=int(metrics.iloc[local].W_F.nunique()),
                          n_free_reflection_orbits=int(metrics.iloc[local].F_reflection_orbit.nunique()))
                for key in ("width_diff", "F_diff_mod_reflection", "F_reflection_only", "stored_torsion_direct_diff"):
                    gr[key+"_pairs"] = sum(int(r[key]) for r in gpairs)
                subset_groups.append(gr)
                group_records.append(gr)
            sr = dict(scope=scope, input_family=family, population_n=len(ix), n_nontrivial_fibers=len(subset_groups),
                      n_knots_in_nontrivial_fibers=sum(r["n_knots"] for r in subset_groups), n_pairs=len(subset_pairs))
            for key in ("width_diff", "F_diff_mod_reflection", "F_reflection_only", "stored_torsion_direct_diff"):
                sr[key+"_pairs"] = sum(int(r[key]) for r in subset_pairs)
                sr[key+"_fibers"] = sum(int(r[key+"_pairs"] > 0) for r in subset_groups)
            summary_records.append(sr)
    pairs = pd.DataFrame(pair_records, columns=PAIR_COLUMNS)
    summary = pd.DataFrame(summary_records)
    write_csv(out, "fiber_summary.csv", summary)
    write_csv(out, "fiber_members.csv", pd.DataFrame(member_records, columns=MEMBER_COLUMNS))
    group_columns = ["scope", "input_family", "tuple_group", "n_knots", "n_pairs", "n_width_values", "n_free_reflection_orbits"] + [k+"_pairs" for k in ("width_diff", "F_diff_mod_reflection", "F_reflection_only", "stored_torsion_direct_diff")]
    write_csv(out, "fiber_details.csv", pd.DataFrame(group_records, columns=group_columns))
    write_csv(out, "all_fiber_pairs.csv", pairs)
    flags = pairs[["width_diff", "F_diff_mod_reflection", "F_reflection_only", "stored_torsion_direct_diff"]].astype(bool)
    candidates = pairs.loc[flags.any(axis=1)].copy()
    if len(candidates):
        candidates = candidates.sort_values(["width_diff", "F_diff_mod_reflection", "same_four_view_inputs", "knot_a", "knot_b"], ascending=[False, False, False, True, True], kind="stable")
    write_csv(out, "discordant_pairs.csv", candidates)
    examples = candidates.drop_duplicates(["knot_a", "knot_b"]).head(args.max_examples)
    (out / "examples").mkdir(exist_ok=True)
    id_local = {v: i for i, v in enumerate(target_ids)}
    for n, (_, r) in enumerate(examples.iterrows(), 1):
        a, b = id_local[r.knot_a], id_local[r.knot_b]
        obj = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.to_dict().items()}
        obj["verification_status"] = "source-archive candidate; requires independent computation and literature check"
        obj["inputs"] = {}
        for view in h.VIEWS[:4]:
            same = bool(np.array_equal(matrices[view][a], matrices[view][b]))
            cols, arr = input_columns[view], matrices[view]
            obj["inputs"][view] = dict(equal=same,
                knot_a={c: int(v) for c, v in zip(cols, arr[a]) if v != 0},
                knot_b={c: int(v) for c, v in zip(cols, arr[b]) if v != 0})
        obj["khovanov"] = {"convention": "q=index1, t=index2; free reflection negates both gradings",
            "knot_a": {c: int(v) for c, v in zip(khcols, Xkh[a]) if v != 0},
            "knot_b": {c: int(v) for c, v in zip(khcols, Xkh[b]) if v != 0}}
        h.atomic_json(out / "examples" / f"example_{n:02d}.json", obj)
    if baseline.exists() and "test" in scopes:
        old = pd.read_csv(baseline)
        old = old[old.fiber_family.eq("HOMFLY_PT_plus_Theta")]
        now = summary[(summary.scope == "test") & (summary.input_family == "HOMFLY_PT_plus_Theta")]
        if len(old) == 1 and len(now) == 1:
            comp = dict(expected_fibers=int(old.iloc[0].n_nontrivial_fibers), observed_fibers=int(now.iloc[0].n_nontrivial_fibers),
                        expected_knots=int(old.iloc[0].n_knots_in_nontrivial_fibers), observed_knots=int(now.iloc[0].n_knots_in_nontrivial_fibers))
            comp["matches_stage33_counts"] = comp["expected_fibers"] == comp["observed_fibers"] and comp["expected_knots"] == comp["observed_knots"]
            h.atomic_json(out / "stage33_count_comparison.json", comp)
            if not comp["matches_stage33_counts"]:
                print("NOTE: source-integer counts differ from Stage 33. Inspect stage33_count_comparison.json before citing 159/321.")
    (out / "interpretation.txt").write_text(
        "Stage 45 searches exact input fibers; no models are trained and no new holdout is created.\n"
        "Test and full-atlas scopes are descriptive and must be reported separately.\n"
        "width_diff: different occupied free-part q-2t diagonal counts, checked against the frozen endpoint.\n"
        "F_diff_mod_reflection: free-rank vectors differ even after simultaneous q,t sign reversal.\n"
        "F_reflection_only: direct difference removed by grading reflection; not evidence of different mirror-independent homology.\n"
        "stored_torsion_direct_diff: raw archived T2_/T4_ difference ONLY; no integral mirror normalization or completeness claim.\n"
        "The four_view_noKh rows certify equality of all four actual source inputs, not an assumed specialization relation.\n"
        "A verified pair with identical inputs and different width rules out perfect deterministic width recovery for both inputs.\n"
        "Archive candidates are not independently recomputed knot invariants and are not automatically novel results.\n"
        "If no discordance is found, report absence in these fibers only, not a general determination theorem.\n"
        "Examples prioritize width, then free-homology discordance, then actual four-view equality; all pair counts are retained.\n"
        "The review ZIP omits the local NPZ coefficient checkpoint; retain it in Drive.\n")
    with zipfile.ZipFile(out / "stage45_review.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file() and f.suffix in {".csv", ".json", ".txt"}:
                z.write(f, f.relative_to(out))
    print(summary.to_string(index=False))
    print("STAGE 45 COMPLETE:", out)
    print("Share:", out / "stage45_review.zip")


if __name__ == "__main__":
    main()
