# %% [markdown]
# Stage 32 — Representation-specific robustness audits before MLST submission
#
# Standalone: DOES NOT depend on objects from a previous Colab runtime.
# It reads persistent files from Google Drive / the corrected run, like Stage 30.
#
# Audits:
#   A) Theta sparsity / zero-padding audit.
#   B) Stored Khovanov F_-only width versus full stored F_/T2_/T4_ diagonal
#      support on the canonical universe (default) or a reproducible sample set by STAGE32_KH_SAMPLE_N.
#
# This stage does NOT alter frozen selections or paper-facing models.
# It only writes diagnostic CSV files.

from __future__ import annotations

from pathlib import Path
import os
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DEFAULT_ROOT = Path(
    (__import__("os").environ.get('KNOT_OUTPUT_DIR', '/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants/processed_consensus_hardness/corrected_run_20260819') + '')
)
DEFAULT_DATA_ROOT = Path(
    (__import__("os").environ.get('KNOT_DATA_DIR', '/content/drive/MyDrive/Colab Notebooks/data_invariants/Invariants') + '')
)

# Same convention as Stage 30: if OUTPUT_DIR exists in the notebook, use it;
# otherwise use the persistent corrected-run path.
ROOT = Path(
    globals().get(
        "OUTPUT_DIR",
        os.environ.get("STAGE32_OUTPUT_ROOT", str(DEFAULT_ROOT)),
    )
)
if not ROOT.exists():
    raise FileNotFoundError(
        f"Corrected-run root not found: {ROOT}\n"
        "Set STAGE32_OUTPUT_ROOT if your corrected run is elsewhere."
    )

DATA_ROOT = Path(os.environ.get("STAGE32_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
OUT = ROOT / "32_representation_robustness_audits"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_N = int(os.environ.get("STAGE32_KH_SAMPLE_N", "313230"))
SEED = int(os.environ.get("STAGE32_SEED", "20260831"))
CHUNK_ROWS = int(os.environ.get("STAGE32_CHUNK_ROWS", "5000"))
ID_CANDIDATES = (
    "knot_id_base", "knot_id", "knot", "Knot", "name", "Name", "id", "ID"
)

# Manual overrides are optional. Auto-discovery is attempted first.
THETA_OVERRIDE = os.environ.get("STAGE32_THETA_SOURCE", "").strip()
KH_OVERRIDE = os.environ.get("STAGE32_KHOVANOV_SOURCE", "").strip()


# ---------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------
def _existing_unique(paths):
    seen = set()
    out = []
    for p in paths:
        p = Path(p)
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if p.exists() and key not in seen:
            seen.add(key)
            out.append(p)
    return out


SEARCH_ROOTS = _existing_unique([
    ROOT,
    DATA_ROOT,
    ROOT.parent,
    ROOT.parent.parent,
])


def find_one(filename: str) -> Path:
    """Find an exact frozen artifact under ROOT."""
    found = sorted(ROOT.rglob(filename))
    if not found:
        raise FileNotFoundError(f"Could not find {filename!r} under {ROOT}")
    return found[0]


def read_columns(path: Path):
    """Read only column names, avoiding a full wide-file load."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    if suffix in (".parquet", ".pq"):
        try:
            import pyarrow.parquet as pq
            return list(pq.ParquetFile(path).schema.names)
        except Exception:
            return list(pd.read_parquet(path).columns)
    raise ValueError(f"Unsupported tabular file: {path}")


def candidate_files(patterns):
    files = []
    for root in SEARCH_ROOTS:
        for pat in patterns:
            try:
                files.extend(root.rglob(pat))
            except Exception:
                pass
    return _existing_unique(files)


def choose_id_col(columns):
    for c in ID_CANDIDATES:
        if c in columns:
            return c
    return None


# ---------------------------------------------------------------------
# Canonical atlas metadata (persistent artifact, not runtime memory)
# ---------------------------------------------------------------------
atlas_path = find_one("final_hard_regime_atlas.parquet")
atlas = pd.read_parquet(atlas_path).reset_index(drop=True)
ID_COL = next((c for c in ID_CANDIDATES if c in atlas.columns), None)
if ID_COL is None:
    raise RuntimeError(
        "Could not identify the canonical knot ID column in final_hard_regime_atlas.parquet"
    )
CANONICAL_IDS = set(atlas[ID_COL].astype(str))


# ---------------------------------------------------------------------
# Locate Theta source
# ---------------------------------------------------------------------
def locate_theta_source() -> Path:
    if THETA_OVERRIDE:
        p = Path(THETA_OVERRIDE)
        if not p.exists():
            raise FileNotFoundError(f"STAGE32_THETA_SOURCE does not exist: {p}")
        return p

    # Exact archive filename described in the manuscript gets highest priority.
    exact = candidate_files(["theta_upto15.csv", "theta_upto15.parquet"])
    if exact:
        return exact[0]

    candidates = candidate_files([
        "*theta*.csv", "*Theta*.csv", "*theta*.parquet", "*Theta*.parquet"
    ])
    scored = []
    for p in candidates:
        try:
            cols = read_columns(p)
        except Exception:
            continue
        # Prefer files with roughly the expected 841-coordinate payload.
        ncols = len(cols)
        score = abs(ncols - 847)  # current source: 6 metadata columns + 841 Theta coefficients
        scored.append((score, -ncols, str(p), p))
    if scored:
        scored.sort()
        return scored[0][-1]

    raise FileNotFoundError(
        "Could not auto-locate the Theta source. Set, for example:\n"
        "os.environ['STAGE32_THETA_SOURCE'] = "
        "'/content/drive/MyDrive/.../theta_upto15.csv'"
    )


# ---------------------------------------------------------------------
# Locate Khovanov source with F_/T2_/T4_ coordinates
# ---------------------------------------------------------------------
# Flexible q,t parser. Examples supported include:
# F_q12_t3, T2_q-4_t-1, F_q12__t3, etc.
KH_RE = re.compile(r"^(F_|T2_|T4_).*?q(-?\d+).*?t(-?\d+)$", re.IGNORECASE)


def parse_kh_coordinates(columns):
    info = []
    for c in map(str, columns):
        m = KH_RE.fullmatch(c)
        if m:
            fam_raw, q, t = m.group(1), int(m.group(2)), int(m.group(3))
            fam = fam_raw.upper()
            # Normalize family capitalization back to manuscript notation.
            if fam == "F_":
                fam = "F_"
            elif fam == "T2_":
                fam = "T2_"
            elif fam == "T4_":
                fam = "T4_"
            info.append((c, fam, q, t, q - 2 * t))
    return info


def locate_khovanov_source():
    if KH_OVERRIDE:
        p = Path(KH_OVERRIDE)
        if not p.exists():
            raise FileNotFoundError(f"STAGE32_KHOVANOV_SOURCE does not exist: {p}")
        cols = read_columns(p)
        info = parse_kh_coordinates(cols)
        if not info:
            raise RuntimeError(
                f"The supplied Khovanov file has no parsable F_/T2_/T4_ q,t columns: {p}"
            )
        return p, info

    candidates = candidate_files([
        "*khov*.csv", "*Khov*.csv", "*KHOV*.csv",
        "*khov*.parquet", "*Khov*.parquet", "*KHOV*.parquet",
        "*even*.csv", "*even*.parquet",
    ])
    scored = []
    for p in candidates:
        try:
            cols = read_columns(p)
            info = parse_kh_coordinates(cols)
        except Exception:
            continue
        if not info:
            continue
        fams = {x[1] for x in info}
        # Prefer a file containing all three stored coordinate families and many coords.
        family_bonus = sum(f in fams for f in ("F_", "T2_", "T4_"))
        scored.append((-family_bonus, -len(info), str(p), p, info))

    if scored:
        scored.sort(key=lambda x: x[:3])
        _, _, _, p, info = scored[0]
        return p, info

    raise FileNotFoundError(
        "Could not auto-locate a Khovanov source containing F_/T2_/T4_ q,t columns. "
        "Set, for example:\n"
        "os.environ['STAGE32_KHOVANOV_SOURCE'] = "
        "'/content/drive/MyDrive/.../khovanov_file.csv'"
    )


THETA_PATH = locate_theta_source()
KH_PATH, KH_COORD_INFO = locate_khovanov_source()

pd.DataFrame([
    {"artifact": "canonical_atlas", "path": str(atlas_path)},
    {"artifact": "theta_source", "path": str(THETA_PATH)},
    {"artifact": "khovanov_source", "path": str(KH_PATH)},
]).to_csv(OUT / "stage32_source_manifest.csv", index=False)

print("Stage 32 sources:")
print("  canonical atlas:", atlas_path)
print("  Theta source:   ", THETA_PATH)
print("  Khovanov source:", KH_PATH)


# ---------------------------------------------------------------------
# Generic chunk reader
# ---------------------------------------------------------------------
def iter_chunks(path: Path, usecols=None, chunksize=CHUNK_ROWS):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from pd.read_csv(path, usecols=usecols, chunksize=chunksize)
        return
    if suffix in (".parquet", ".pq"):
        # Parquet iteration by row group if pyarrow is available.
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=chunksize, columns=usecols):
                yield batch.to_pandas()
            return
        except Exception:
            # Fallback: one full read. This is only used if pyarrow iteration is unavailable.
            frame = pd.read_parquet(path, columns=usecols)
            for lo in range(0, len(frame), chunksize):
                yield frame.iloc[lo:lo + chunksize].copy()
            return
    raise ValueError(f"Unsupported tabular file: {path}")


# =====================================================================
# A) Theta sparsity / padding audit
# =====================================================================
theta_columns = read_columns(THETA_PATH)
theta_id_col = choose_id_col(theta_columns)

# The Theta source contains six metadata columns followed by a 29 x 29 = 841
# coefficient grid.  Identify coefficients by their coordinate names rather than
# by numeric dtype so metadata such as number_of_crossings/table_number can never
# leak into this audit.
theta_coord_re = re.compile(r"^T1-?\d+_T2-?\d+$")
theta_features = [c for c in theta_columns if theta_coord_re.fullmatch(str(c))]

# Fallback for an alternate export that preserves the same metadata but renames
# coefficient columns.  Keep this conservative and explicit.
if not theta_features:
    if THETA_PATH.suffix.lower() == ".csv":
        probe = pd.read_csv(THETA_PATH, nrows=200)
    else:
        probe = pd.read_parquet(THETA_PATH).head(200)
    excluded = set(ID_CANDIDATES) | {
        "crossing_number", "crossings", "number_of_crossings", "table_number",
        "signature", "s_invariant", "is_alternating"
    }
    theta_features = [
        c for c in probe.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(probe[c])
    ]

if not theta_features:
    raise RuntimeError(f"No Theta numeric features identified in {THETA_PATH}")

# Keep the audit transparent if auto-inference differs from the expected 841.
expected_theta_dim = 841

col_nnz = np.zeros(len(theta_features), dtype=np.int64)
col_sum = np.zeros(len(theta_features), dtype=np.float64)
col_sumsq = np.zeros(len(theta_features), dtype=np.float64)
row_nnz_parts = []
n_theta_rows = 0
canonical_theta_rows = 0

usecols = list(theta_features)
if theta_id_col is not None and theta_id_col not in usecols:
    usecols = [theta_id_col] + usecols

for chunk in iter_chunks(THETA_PATH, usecols=usecols):
    if theta_id_col is not None:
        ids = chunk[theta_id_col].astype(str)
        # If IDs are directly compatible with the canonical atlas, audit exactly the
        # 313,230 working knots. If not, process the source rows; the manifest records it.
        mask = ids.isin(CANONICAL_IDS)
        if mask.any():
            canonical_theta_rows += int(mask.sum())
            work = chunk.loc[mask, theta_features]
        else:
            work = chunk[theta_features]
    else:
        work = chunk[theta_features]

    if len(work) == 0:
        continue
    x = work.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("Theta source contains non-finite values in coefficient columns")
    nz = x != 0
    col_nnz += nz.sum(axis=0)
    col_sum += x.sum(axis=0)
    col_sumsq += np.square(x).sum(axis=0)
    row_nnz_parts.append(nz.sum(axis=1))
    n_theta_rows += len(work)

if n_theta_rows == 0:
    raise RuntimeError("Theta audit processed zero rows")

row_nnz = np.concatenate(row_nnz_parts)
col_mean = col_sum / n_theta_rows
col_var = np.maximum(col_sumsq / n_theta_rows - col_mean ** 2, 0.0)
zero_var = col_var <= 1e-14
never_nonzero = col_nnz == 0

overall_zero_fraction = float(1.0 - col_nnz.sum() / (n_theta_rows * len(theta_features)))

# Conservative diagnostics only; these are not inferential thresholds.
theta_summary = pd.DataFrame([{
    "theta_source": str(THETA_PATH),
    "n_rows_audited": int(n_theta_rows),
    "canonical_id_rows_seen": int(canonical_theta_rows),
    "n_theta_features_detected": int(len(theta_features)),
    "expected_theta_features": expected_theta_dim,
    "dimension_matches_expected_841": bool(len(theta_features) == expected_theta_dim),
    "overall_zero_fraction": overall_zero_fraction,
    "n_never_nonzero_columns": int(never_nonzero.sum()),
    "fraction_never_nonzero_columns": float(never_nonzero.mean()),
    "n_zero_variance_columns": int(zero_var.sum()),
    "fraction_zero_variance_columns": float(zero_var.mean()),
    "median_nonzero_coords_per_row": float(np.median(row_nnz)),
    "q05_nonzero_coords_per_row": float(np.quantile(row_nnz, .05)),
    "q95_nonzero_coords_per_row": float(np.quantile(row_nnz, .95)),
    "median_coordinate_prevalence": float(np.median(col_nnz / n_theta_rows)),
    "q05_coordinate_prevalence": float(np.quantile(col_nnz / n_theta_rows, .05)),
    "q95_coordinate_prevalence": float(np.quantile(col_nnz / n_theta_rows, .95)),
}])
theta_summary.to_csv(OUT / "theta_sparsity_summary.csv", index=False)

pd.DataFrame({
    "feature": list(map(str, theta_features)),
    "nonzero_count": col_nnz,
    "nonzero_fraction": col_nnz / n_theta_rows,
    "variance": col_var,
    "never_nonzero": never_nonzero,
    "zero_variance": zero_var,
}).sort_values(["nonzero_fraction", "feature"]).to_csv(
    OUT / "theta_coordinate_prevalence.csv", index=False
)

pd.DataFrame({"nonzero_coordinates_per_row": row_nnz}).describe(
    percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]
).to_csv(OUT / "theta_row_sparsity_distribution.csv")


# =====================================================================
# B) Khovanov F_-only vs full stored F_/T2_/T4_ diagonal-support width
# =====================================================================
kh_columns = read_columns(KH_PATH)
kh_id_col = choose_id_col(kh_columns)
coord_df = pd.DataFrame(
    KH_COORD_INFO, columns=["column", "family", "q", "t", "diagonal"]
)
coord_df.to_csv(OUT / "khovanov_full_stored_coordinate_map.csv", index=False)

f_cols = coord_df.loc[coord_df["family"].eq("F_"), "column"].tolist()
full_cols = coord_df["column"].tolist()
if not f_cols:
    raise RuntimeError("No F_ coordinates parsed from the Khovanov source")

# Report source-family counts; manuscript expectation is 373 F_, 337 T2_, 16 T4_.
family_counts = (
    coord_df.groupby("family").size().rename("n_coordinates").reset_index()
)
family_counts.to_csv(OUT / "khovanov_stored_family_counts.csv", index=False)

diag_by_col = dict(zip(coord_df["column"], coord_df["diagonal"]))
f_diags = np.asarray([diag_by_col[c] for c in f_cols], dtype=int)
full_diags = np.asarray([diag_by_col[c] for c in full_cols], dtype=int)


def widths_from_array(arr: np.ndarray, diags: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape[0], dtype=np.int16)
    for i in range(arr.shape[0]):
        active = np.flatnonzero(arr[i] != 0)
        if active.size:
            out[i] = np.unique(diags[active]).size
    return out


# Uniform reproducible priority sample while streaming the full source.
rng = np.random.default_rng(SEED)
reservoir = pd.DataFrame(columns=["_key", "source_id", "stored_width_F_only", "stored_width_full_F_T2_T4"])
source_rows_seen = 0
canonical_kh_rows_seen = 0

kh_usecols = list(full_cols)
if kh_id_col is not None:
    kh_usecols = [kh_id_col] + kh_usecols

for chunk in iter_chunks(KH_PATH, usecols=kh_usecols):
    source_rows_seen += len(chunk)
    if kh_id_col is not None:
        src_ids = chunk[kh_id_col].astype(str)
        in_canonical = src_ids.isin(CANONICAL_IDS)
        canonical_kh_rows_seen += int(in_canonical.sum())
        # Audit ONLY the exact canonical working universe.  Chunks containing no
        # canonical rows are discarded rather than entering the reservoir.
        chunk = chunk.loc[in_canonical].copy()
        src_ids = chunk[kh_id_col].astype(str)
    else:
        src_ids = pd.Series([f"row_{source_rows_seen-len(chunk)+i}" for i in range(len(chunk))])

    if len(chunk) == 0:
        continue

    arr_f = chunk[f_cols].to_numpy(dtype=np.float64, copy=False)
    arr_full = chunk[full_cols].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(arr_f).all() or not np.isfinite(arr_full).all():
        raise ValueError("Khovanov source contains non-finite coordinate values")

    wf = widths_from_array(arr_f, f_diags)
    wfull = widths_from_array(arr_full, full_diags)
    keys = rng.random(len(chunk))
    cand = pd.DataFrame({
        "_key": keys,
        "source_id": src_ids.to_numpy(dtype=str),
        "stored_width_F_only": wf.astype(int),
        "stored_width_full_F_T2_T4": wfull.astype(int),
    })
    reservoir = pd.concat([reservoir, cand], ignore_index=True)
    if len(reservoir) > SAMPLE_N:
        reservoir = reservoir.nsmallest(SAMPLE_N, "_key").reset_index(drop=True)

sample_out = reservoir.nsmallest(min(SAMPLE_N, len(reservoir)), "_key").copy()
sample_out["full_minus_F"] = (
    sample_out["stored_width_full_F_T2_T4"].astype(int)
    - sample_out["stored_width_F_only"].astype(int)
)
sample_out = sample_out.drop(columns=["_key"])
sample_out.to_csv(OUT / f"khovanov_width_concordance_sample{len(sample_out)}.csv", index=False)

w_f = sample_out["stored_width_F_only"].to_numpy(dtype=float)
w_full = sample_out["stored_width_full_F_T2_T4"].to_numpy(dtype=float)
diff = w_full - w_f
spearman = pd.Series(w_f).corr(pd.Series(w_full), method="spearman")

kh_summary = pd.DataFrame([{
    "khovanov_source": str(KH_PATH),
    "source_rows_seen": int(source_rows_seen),
    "canonical_id_rows_seen": int(canonical_kh_rows_seen),
    "sample_n": int(len(sample_out)),
    "n_F_coordinates": int((coord_df.family == "F_").sum()),
    "n_T2_coordinates": int((coord_df.family == "T2_").sum()),
    "n_T4_coordinates": int((coord_df.family == "T4_").sum()),
    "exact_width_agreement_fraction": float(np.mean(w_f == w_full)),
    "spearman_F_vs_full_stored_width": float(spearman),
    "mean_F_only_width": float(np.mean(w_f)),
    "mean_full_stored_width": float(np.mean(w_full)),
    "mean_full_minus_F": float(np.mean(diff)),
    "fraction_full_width_larger": float(np.mean(diff > 0)),
    "max_full_minus_F": int(np.max(diff)) if len(diff) else 0,
}])
kh_summary.to_csv(OUT / "khovanov_width_concordance_summary.csv", index=False)


# ---------------------------------------------------------------------
# Compact decision sheet for manuscript revision
# ---------------------------------------------------------------------
theta_row = theta_summary.iloc[0]
kh_row = kh_summary.iloc[0]

decision = pd.DataFrame([
    {
        "audit": "Theta sparsity/padding",
        "key_result": (
            f"zero_fraction={theta_row['overall_zero_fraction']:.4f}; "
            f"never_nonzero={int(theta_row['n_never_nonzero_columns'])}/"
            f"{int(theta_row['n_theta_features_detected'])}; "
            f"zero_variance={int(theta_row['n_zero_variance_columns'])}"
        ),
        "interpretation_rule": (
            "Never-nonzero/zero-variance columns diagnose literal padding; high sparsity without "
            "dead columns is a different issue and should be described as sparse support rather than padding."
        ),
    },
    {
        "audit": "Khovanov stored-width convention",
        "key_result": (
            f"exact_agreement={kh_row['exact_width_agreement_fraction']:.4f}; "
            f"spearman={kh_row['spearman_F_vs_full_stored_width']:.4f}; "
            f"fraction_full_larger={kh_row['fraction_full_width_larger']:.4f}"
        ),
        "interpretation_rule": (
            "High agreement supports F_-part width as a faithful stored proxy; otherwise retain the "
            "operational wording prominently and consider recomputing the endpoint with full stored support."
        ),
    },
])
decision.to_csv(OUT / "stage32_decision_summary.csv", index=False)

print("\nTheta sparsity summary:")
print(theta_summary.to_string(index=False))
print("\nKhovanov stored-family counts:")
print(family_counts.to_string(index=False))
print("\nKhovanov width concordance summary:")
print(kh_summary.to_string(index=False))
print("\nSaved Stage 32 to:", OUT)
