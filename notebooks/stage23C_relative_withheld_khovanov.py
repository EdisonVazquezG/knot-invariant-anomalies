# ============================================================
# Stage 23C
# Relative-error withheld-Khovanov analysis
#
# Question:
# Does a size-matched no-Khovanov relative-error selection
# localize larger stored Khovanov structure?
#
# Run AFTER:
#   stage23_anomaly_score_baselines.py
#
# Uses:
#   - same frozen train/validation/test split
#   - Khovanov excluded from selection
#   - n = 31, matching the no-Khovanov conditional benchmark
#   - fixed-cardinality joint-label randomization
#   - crossing, alternation, |sigma|
#   - four selection-view amplitudes
#   - withheld Khovanov norm
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import HistGradientBoostingRegressor


# ------------------------------------------------------------
# 0. Preconditions
# ------------------------------------------------------------

REQUIRED = (
    "meta",
    "test_idx",
    "val_idx",
    "score_by_view",
    "test_kh_diag",
    "test_kh_support",
    "INVARIANTS",
    "NO_KHOVANOV",
    "ROOT",
)

missing = [x for x in REQUIRED if x not in globals()]

if missing:
    raise RuntimeError(
        "Run stage23_anomaly_score_baselines.py first. "
        f"Missing: {missing}"
    )

OUT = Path(ROOT) / "23C_relative_no_khovanov_withheld"
OUT.mkdir(parents=True, exist_ok=True)

N_TEST = len(test_idx)
N_SELECT = 31

SEED = 20260901

# Paper-facing null
PAPER_BINS = 2
PAPER_REPS = 5000

# Resolution audit
GRID_BINS = (20, 10, 5, 2, 1)
GRID_REPS = 2000

NO_KHOVANOV = tuple(
    x for x in INVARIANTS
    if x != "Khovanov"
)

assert len(NO_KHOVANOV) == 4


# ------------------------------------------------------------
# 1. Construct relative-error C3 score WITHOUT Khovanov
# ------------------------------------------------------------
#
# Important:
# Per-view anomaly scores live on different numerical scales.
# We first convert each relative-NRE score to its empirical
# percentile rank within the frozen test set.
#
# C3 = third-largest of the four no-Khovanov per-view ranks.
# For four views this is equivalent to the second-smallest rank:
# to score highly, at least 3/4 views must score highly.
# ------------------------------------------------------------

def empirical_percentile_rank(x):
    x = np.asarray(x, dtype=float)

    if not np.isfinite(x).all():
        raise ValueError("Non-finite score detected")

    return rankdata(
        x,
        method="average",
    ) / len(x)


relative_rank_by_view = {}

for name in NO_KHOVANOV:

    score = np.asarray(
        score_by_view[name]["test_relative_nre"],
        dtype=float,
    )

    relative_rank_by_view[name] = empirical_percentile_rank(
        score
    )


rank_matrix = np.column_stack([
    relative_rank_by_view[name]
    for name in NO_KHOVANOV
])

# Ascending:
# [lowest, second-lowest, second-highest, highest]
#
# Requiring 3 of 4 views to be high is represented by
# the second-lowest score = third-largest score.
relative_c3 = np.sort(
    rank_matrix,
    axis=1,
)[:, 1]


# Stable top-n selection
order = np.lexsort((
    np.arange(N_TEST),
    -relative_c3,
))

selected_local_idx = order[:N_SELECT]

selected = np.zeros(
    N_TEST,
    dtype=bool,
)

selected[selected_local_idx] = True

assert selected.sum() == N_SELECT


# ------------------------------------------------------------
# 2. Observed withheld Khovanov outcomes
# ------------------------------------------------------------

kh_diag = np.asarray(
    test_kh_diag,
    dtype=float,
)

kh_support = np.asarray(
    test_kh_support,
    dtype=float,
)


def endpoint_values(mask):

    diag = kh_diag[mask]
    support = kh_support[mask]

    return {
        "n": int(mask.sum()),

        "kh_diagonal_mean":
            float(np.mean(diag)),

        "kh_diagonal_ge_3_prop":
            float(np.mean(diag >= 3)),

        "kh_diagonal_ge_4_prop":
            float(np.mean(diag >= 4)),

        "kh_support_mean":
            float(np.mean(support)),
    }


observed = endpoint_values(selected)

print("\nObserved relative-error no-Khovanov selection:")
print(pd.DataFrame([observed]).to_string(index=False))


# ------------------------------------------------------------
# 3. Sanity check against existing Stage 23B descriptive result
# ------------------------------------------------------------
#
# Current Stage 23B result expected approximately:
#
# mean diagonal     = 3.129032
# P(W_Kh >= 3)      = 0.967742
# P(W_Kh >= 4)      = 0.161290
#
# If these do NOT reproduce, STOP:
# the exact Stage23B C3 aggregation implementation differs
# from the rank-C3 reconstruction above.
# ------------------------------------------------------------

EXPECTED = {
    "kh_diagonal_mean": 3.129032,
    "kh_diagonal_ge_3_prop": 0.967742,
    "kh_diagonal_ge_4_prop": 0.161290,
}

print("\nStage-23B reproduction check:")

for metric, expected in EXPECTED.items():

    actual = observed[metric]

    print(
        f"{metric:28s} "
        f"actual={actual:.6f} "
        f"expected≈{expected:.6f} "
        f"diff={actual-expected:+.6f}"
    )


max_difference = max(
    abs(observed[k] - v)
    for k, v in EXPECTED.items()
)

if max_difference > 1e-5:

    raise RuntimeError(
        "\nThe selection does not exactly reproduce Stage 23B. "
        "Do NOT run inference until the C3 ranking construction "
        "is made identical to stage23B_size_matched_score_sensitivity.py."
    )


# ------------------------------------------------------------
# 4. Structural covariates
# ------------------------------------------------------------

test_meta = meta.iloc[test_idx].reset_index(drop=True)

crossing = (
    test_meta["number_of_crossings"]
    .to_numpy(int)
)

alternating = (
    test_meta["is_alternating"]
    .to_numpy(int)
)

signature_abs = (
    test_meta["signature"]
    .abs()
    .to_numpy()
)


# ------------------------------------------------------------
# 5. Validation-derived amplitude bins
# ------------------------------------------------------------
#
# Edges are estimated on VALIDATION only and then applied to TEST.
# We include:
#
#   Alexander norm
#   Jones norm
#   HOMFLY-PT norm
#   Theta norm
#   withheld Khovanov norm
#
# This keeps the withheld Khovanov representation OUT of selection,
# while allowing its amplitude to be treated as a nuisance variable.
# ------------------------------------------------------------

ALL_NORM_VIEWS = (
    "Alexander",
    "Jones",
    "HOMFLY-PT",
    "Theta",
    "Khovanov",
)


def validation_quantile_bins(
    val_values,
    test_values,
    n_bins,
):

    val_values = np.asarray(
        val_values,
        dtype=float,
    )

    test_values = np.asarray(
        test_values,
        dtype=float,
    )

    if n_bins == 1:
        return np.zeros(
            len(test_values),
            dtype=np.int16,
        )

    edges = np.quantile(
        val_values,
        np.linspace(
            0,
            1,
            n_bins + 1,
        ),
    )

    edges = np.unique(edges)

    if len(edges) <= 2:
        return np.zeros(
            len(test_values),
            dtype=np.int16,
        )

    # interior edges only
    bins = np.searchsorted(
        edges[1:-1],
        test_values,
        side="right",
    )

    return bins.astype(np.int16)


def make_joint_strata(n_bins):

    data = {
        "crossing": crossing,
        "alternating": alternating,
        "signature_abs": signature_abs,
    }

    for name in ALL_NORM_VIEWS:

        data[f"norm_{name}"] = validation_quantile_bins(
            score_by_view[name]["val_log_norm"],
            score_by_view[name]["test_log_norm"],
            n_bins=n_bins,
        )

    frame = pd.DataFrame(data)

    # One exact COMMON stratum code per test knot
    codes, _ = pd.factorize(
        pd.MultiIndex.from_frame(frame),
        sort=False,
    )

    return codes.astype(np.int32)


# ------------------------------------------------------------
# 6. Fixed-cardinality stratified randomization
# ------------------------------------------------------------

def prepare_sampler(strata, selected_mask):

    strata = np.asarray(strata)
    selected_mask = np.asarray(
        selected_mask,
        dtype=bool,
    )

    payload = []

    fixed_selected = 0

    for code in np.unique(strata):

        members = np.flatnonzero(
            strata == code
        )

        n_selected = int(
            selected_mask[members].sum()
        )

        if n_selected == 0:
            continue

        payload.append(
            (
                members,
                n_selected,
            )
        )

        if n_selected == len(members):
            fixed_selected += n_selected

    return payload, fixed_selected


def sample_fixed_cardinality(
    sampler,
    rng,
):

    mask = np.zeros(
        N_TEST,
        dtype=bool,
    )

    for members, n_selected in sampler:

        if n_selected == len(members):

            chosen = members

        else:

            chosen = rng.choice(
                members,
                size=n_selected,
                replace=False,
            )

        mask[chosen] = True

    if mask.sum() != N_SELECT:
        raise AssertionError(
            (
                mask.sum(),
                N_SELECT,
            )
        )

    return mask


METRICS = (
    "kh_diagonal_mean",
    "kh_diagonal_ge_3_prop",
    "kh_diagonal_ge_4_prop",
    "kh_support_mean",
)


def run_null(
    n_bins,
    n_reps,
    seed,
):

    strata = make_joint_strata(
        n_bins
    )

    sampler, fixed_selected = prepare_sampler(
        strata,
        selected,
    )

    selected_cell_sizes = np.array([
        np.sum(strata == strata[i])
        for i in np.flatnonzero(selected)
    ])

    rng = np.random.default_rng(
        seed
    )

    values = {
        metric: np.empty(
            n_reps,
            dtype=float,
        )
        for metric in METRICS
    }

    for b in range(n_reps):

        null_selected = sample_fixed_cardinality(
            sampler,
            rng,
        )

        result = endpoint_values(
            null_selected
        )

        for metric in METRICS:
            values[metric][b] = result[metric]

        if (b + 1) % 500 == 0:
            print(
                f"{n_bins:2d} bins: "
                f"{b+1:,}/{n_reps:,}"
            )

    summary_rows = []

    for metric in METRICS:

        null_values = values[metric]

        obs = observed[metric]

        n_ge = int(
            np.sum(
                null_values >= obs
            )
        )

        p_emp = (
            n_ge + 1
        ) / (
            n_reps + 1
        )

        summary_rows.append({
            "bins_per_view": n_bins,
            "metric": metric,
            "selected_n": N_SELECT,
            "observed": obs,
            "null_mean":
                float(np.mean(null_values)),
            "null_sd":
                float(
                    np.std(
                        null_values,
                        ddof=1,
                    )
                ),
            "null_q025":
                float(
                    np.quantile(
                        null_values,
                        0.025,
                    )
                ),
            "null_q975":
                float(
                    np.quantile(
                        null_values,
                        0.975,
                    )
                ),
            "n_ge_observed": n_ge,
            "n_randomizations": n_reps,
            "p_emp": p_emp,
            "median_selected_cell_size":
                float(
                    np.median(
                        selected_cell_sizes
                    )
                ),
            "fixed_selected":
                int(fixed_selected),
        })

    return (
        pd.DataFrame(summary_rows),
        values,
    )


# ------------------------------------------------------------
# 7. Grid trajectory
# ------------------------------------------------------------

grid_frames = []

for n_bins in GRID_BINS:

    print(
        "\n" + "=" * 70
    )
    print(
        f"Relative-error no-Khovanov null: {n_bins} bins/view"
    )
    print(
        "=" * 70
    )

    frame, _ = run_null(
        n_bins=n_bins,
        n_reps=GRID_REPS,
        seed=SEED + n_bins,
    )

    grid_frames.append(
        frame
    )


grid_summary = pd.concat(
    grid_frames,
    ignore_index=True,
)

grid_summary.to_csv(
    OUT /
    "relative_no_khovanov_grid_trajectory.csv",
    index=False,
)

print(
    "\nGRID TRAJECTORY — MEAN DIAGONAL"
)

display(
    grid_summary.loc[
        grid_summary["metric"].eq(
            "kh_diagonal_mean"
        ),
        [
            "bins_per_view",
            "selected_n",
            "observed",
            "null_mean",
            "p_emp",
            "median_selected_cell_size",
            "fixed_selected",
        ],
    ]
    .sort_values(
        "bins_per_view",
        ascending=False,
    )
)


# ------------------------------------------------------------
# 8. Paper-facing 2-bin null with 5,000 randomizations
# ------------------------------------------------------------

print(
    "\n" + "=" * 70
)
print(
    "FINAL 2-BIN PAPER-FACING NULL"
)
print(
    "=" * 70
)

paper_summary, paper_null = run_null(
    n_bins=PAPER_BINS,
    n_reps=PAPER_REPS,
    seed=SEED,
)


# Holm correction across four declared endpoints
reject, p_holm, _, _ = multipletests(
    paper_summary["p_emp"].to_numpy(),
    method="holm",
)

paper_summary["holm_p"] = p_holm
paper_summary["holm_reject_005"] = reject


# Standardized null effect
paper_summary["z_null"] = (
    paper_summary["observed"]
    - paper_summary["null_mean"]
) / paper_summary["null_sd"]


paper_summary.to_csv(
    OUT /
    "relative_no_khovanov_final_joint_null.csv",
    index=False,
)

print(
    "\nFINAL STRATIFIED RESULT"
)

display(
    paper_summary[
        [
            "metric",
            "observed",
            "null_mean",
            "null_q025",
            "null_q975",
            "z_null",
            "p_emp",
            "holm_p",
            "median_selected_cell_size",
            "fixed_selected",
        ]
    ]
)


# ------------------------------------------------------------
# 9. Continuous-amplitude held-out sensitivity
# ------------------------------------------------------------
#
# Stronger complement:
#
# nuisance model is trained on VALIDATION knots only,
# then applied once to the frozen TEST set.
#
# This means selected test outcomes are not used to fit the
# amplitude-adjustment model.
# ------------------------------------------------------------

# Reload full phenotype atlas so validation Khovanov outcomes are available
if "phenotype_path" not in globals():
    matches = list(
        Path(ROOT).rglob(
            "complete_mathematical_phenotype_atlas.parquet"
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely locate phenotype atlas"
        )

    phenotype_path = matches[0]


phenotype_full = pd.read_parquet(
    phenotype_path
)

phenotype_full[ID_COL] = (
    phenotype_full[ID_COL]
    .astype(str)
)

phenotype_full = (
    phenotype_full
    .set_index(ID_COL)
)


KH_DIAG_COL = next(
    c
    for c in (
        "khovanov_q_minus_2t_diagonal_count",
        "kh_diagonal_count",
        "khovanov_diagonal_count",
    )
    if c in phenotype_full.columns
)


val_ids = (
    meta.iloc[val_idx][ID_COL]
    .astype(str)
    .to_numpy()
)

test_ids_local = (
    meta.iloc[test_idx][ID_COL]
    .astype(str)
    .to_numpy()
)

y_val = (
    phenotype_full
    .reindex(val_ids)[KH_DIAG_COL]
    .to_numpy(float)
)

y_test = (
    phenotype_full
    .reindex(test_ids_local)[KH_DIAG_COL]
    .to_numpy(float)
)

if (
    np.isnan(y_val).any()
    or np.isnan(y_test).any()
):
    raise RuntimeError(
        "Missing Khovanov outcome"
    )


def continuous_features(
    indices,
    split_name,
):

    m = meta.iloc[
        indices
    ]

    columns = []

    # all four selection-view amplitudes
    # + withheld Khovanov amplitude
    for name in ALL_NORM_VIEWS:

        columns.append(
            np.asarray(
                score_by_view[name][
                    f"{split_name}_log_norm"
                ],
                dtype=float,
            )
        )

    columns.extend([
        m["number_of_crossings"]
        .to_numpy(float),

        m["is_alternating"]
        .to_numpy(float),

        m["signature"]
        .abs()
        .to_numpy(float),
    ])

    return np.column_stack(
        columns
    )


X_val_nuisance = continuous_features(
    val_idx,
    "val",
)

X_test_nuisance = continuous_features(
    test_idx,
    "test",
)


nuisance_model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=SEED,
)

nuisance_model.fit(
    X_val_nuisance,
    y_val,
)

pred_test = nuisance_model.predict(
    X_test_nuisance
)

residual_test = (
    y_test
    - pred_test
)

observed_residual = float(
    np.mean(
        residual_test[selected]
    )
)


# Fixed-cardinality randomization only inside exact
# structural strata because amplitude is handled continuously.
structural_frame = pd.DataFrame({
    "crossing": crossing,
    "alternating": alternating,
    "signature_abs": signature_abs,
})

structural_strata, _ = pd.factorize(
    pd.MultiIndex.from_frame(
        structural_frame
    ),
    sort=False,
)

structural_sampler, structural_fixed = prepare_sampler(
    structural_strata,
    selected,
)

rng = np.random.default_rng(
    SEED + 1000
)

residual_null = np.empty(
    PAPER_REPS,
    dtype=float,
)

for b in range(PAPER_REPS):

    null_selected = sample_fixed_cardinality(
        structural_sampler,
        rng,
    )

    residual_null[b] = np.mean(
        residual_test[
            null_selected
        ]
    )


n_ge = int(
    np.sum(
        residual_null >= observed_residual
    )
)

continuous_p = (
    n_ge + 1
) / (
    PAPER_REPS + 1
)

continuous_summary = pd.DataFrame([{
    "analysis":
        "relative_error_no_khovanov_n31",

    "selected_n":
        N_SELECT,

    "observed_mean_diagonal":
        float(
            np.mean(
                y_test[selected]
            )
        ),

    "selected_mean_residual":
        observed_residual,

    "residual_null_mean":
        float(
            np.mean(
                residual_null
            )
        ),

    "residual_null_q025":
        float(
            np.quantile(
                residual_null,
                0.025,
            )
        ),

    "residual_null_q975":
        float(
            np.quantile(
                residual_null,
                0.975,
            )
        ),

    "p_emp":
        continuous_p,

    "fixed_selected_structural":
        structural_fixed,
}])


continuous_summary.to_csv(
    OUT /
    "relative_no_khovanov_continuous_amplitude.csv",
    index=False,
)

print(
    "\nCONTINUOUS-AMPLITUDE SENSITIVITY"
)

display(
    continuous_summary
)


# ------------------------------------------------------------
# 10. Save selected knots
# ------------------------------------------------------------

selected_ids = (
    meta.iloc[
        test_idx[selected]
    ][
        [
            ID_COL,
            "number_of_crossings",
            "is_alternating",
            "signature",
        ]
    ]
    .copy()
)

selected_ids[
    "relative_C3"
] = relative_c3[selected]

selected_ids[
    "stored_khovanov_diagonal_count"
] = kh_diag[selected]

selected_ids[
    "stored_khovanov_support_size"
] = kh_support[selected]

selected_ids.to_csv(
    OUT /
    "relative_no_khovanov_selected_n31.csv",
    index=False,
)


print(
    "\nSaved Stage 23C to:",
    OUT,
)