# ============================================================
# Stage 23D
# CONDITIONAL percentile — withheld-Khovanov analysis
# Equal-cardinality held-out benchmark, n = 31
#
# Question:
# Does the size-matched no-Khovanov CONDITIONAL selection
# localize larger stored Khovanov structure?
#
# Run AFTER:
#   stage23_anomaly_score_baselines.py
#
# Design:
#   - same frozen train / validation / test split
#   - Khovanov excluded from selection
#   - n = 31
#   - selection uses Alexander, Jones, HOMFLY-PT, Theta
#   - primary conditional score = 100-bin held-out percentile
#   - fixed-cardinality joint-label randomization
#   - structural controls:
#         crossing number
#         alternation
#         |signature|
#   - amplitude controls:
#         four selection-view norms
#         withheld Khovanov norm
#   - grid trajectory:
#         20, 10, 5, 2, 1 bins
#   - final paper-facing:
#         2 bins, 5,000 randomizations
#   - continuous-amplitude sensitivity
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import HistGradientBoostingRegressor


# ============================================================
# 0. Preconditions
# ============================================================

REQUIRED = (
    "meta",
    "test_idx",
    "val_idx",
    "score_by_view",
    "test_kh_diag",
    "test_kh_support",
    "INVARIANTS",
    "ROOT",
)

missing = [
    x
    for x in REQUIRED
    if x not in globals()
]

if missing:
    raise RuntimeError(
        "Run stage23_anomaly_score_baselines.py first. "
        f"Missing variables: {missing}"
    )


# ============================================================
# 1. Analysis constants
# ============================================================

OUT = (
    Path(ROOT)
    / "23D_conditional_no_khovanov_withheld_n31"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

N_TEST = len(test_idx)
N_SELECT = 31

SEED = 20260901

PAPER_BINS = 2
PAPER_REPS = 5000

GRID_BINS = (
    20,
    10,
    5,
    2,
    1,
)

GRID_REPS = 2000

NO_KHOVANOV = tuple(
    name
    for name in INVARIANTS
    if name != "Khovanov"
)

assert len(NO_KHOVANOV) == 4

print(
    "No-Khovanov views:",
    NO_KHOVANOV,
)


# ============================================================
# 2. Read the PRIMARY 100-bin conditional percentile
# ============================================================
#
# Stage 23 stores:
#
#   test_conditional_percentile_50
#   test_conditional_percentile_100
#   test_conditional_percentile_200
#
# The primary paper-facing held-out benchmark uses 100 bins.
# ============================================================

def get_conditional_test_score(view_name):

    key = "test_conditional_percentile_100"

    if key not in score_by_view[view_name]:

        raise KeyError(
            f"{view_name} does not contain '{key}'.\n"
            f"Available keys:\n"
            f"{list(score_by_view[view_name].keys())}"
        )

    score = np.asarray(
        score_by_view[view_name][key],
        dtype=float,
    )

    if len(score) != N_TEST:

        raise ValueError(
            f"{view_name}: score length "
            f"{len(score)} != N_TEST={N_TEST}"
        )

    if not np.isfinite(score).all():

        raise ValueError(
            f"Non-finite values found in "
            f"{view_name} {key}"
        )

    print(
        f"{view_name}: using "
        f"score_by_view['{view_name}']['{key}']"
    )

    return score


# ============================================================
# 3. Construct conditional C3 WITHOUT Khovanov
# ============================================================
#
# These values are ALREADY held-out conditional percentiles.
# We therefore do NOT re-rank them.
#
# For four views, the third-largest score equals the
# second-smallest score when sorted ascending.
#
# A high C3 therefore means at least 3 of 4 views assign
# a high conditional percentile.
# ============================================================

conditional_score_by_view = {}

for name in NO_KHOVANOV:

    conditional_score_by_view[name] = (
        get_conditional_test_score(name)
    )


score_matrix = np.column_stack([
    conditional_score_by_view[name]
    for name in NO_KHOVANOV
])


# Third-largest among 4 values
conditional_c3 = np.sort(
    score_matrix,
    axis=1,
)[:, 1]


# Stable top-31
order = np.lexsort((
    np.arange(N_TEST),
    -conditional_c3,
))

selected_local_idx = (
    order[:N_SELECT]
)

selected = np.zeros(
    N_TEST,
    dtype=bool,
)

selected[
    selected_local_idx
] = True

assert selected.sum() == N_SELECT

print(
    "\nConditional no-Khovanov "
    f"selected n = {selected.sum()}"
)


# ============================================================
# 4. Withheld-Khovanov outcomes
# ============================================================

kh_diag = np.asarray(
    test_kh_diag,
    dtype=float,
)

kh_support = np.asarray(
    test_kh_support,
    dtype=float,
)

if len(kh_diag) != N_TEST:
    raise ValueError(
        "test_kh_diag length mismatch"
    )

if len(kh_support) != N_TEST:
    raise ValueError(
        "test_kh_support length mismatch"
    )


def endpoint_values(mask):

    diag = kh_diag[mask]
    support = kh_support[mask]

    return {

        "n":
            int(mask.sum()),

        "kh_diagonal_mean":
            float(
                np.mean(diag)
            ),

        "kh_diagonal_ge_3_prop":
            float(
                np.mean(diag >= 3)
            ),

        "kh_diagonal_ge_4_prop":
            float(
                np.mean(diag >= 4)
            ),

        "kh_support_mean":
            float(
                np.mean(support)
            ),
    }


observed = endpoint_values(
    selected
)

print(
    "\nObserved CONDITIONAL "
    "no-Khovanov selection:"
)

display(
    pd.DataFrame([
        observed
    ])
)


# ============================================================
# 5. Stage-23B reproduction check
# ============================================================
#
# Existing size-matched no-Khovanov result:
#
#   Conditional P(W_Kh >= 3) = 0.871 approximately
#
# At n = 31:
#
#   27 / 31 = 0.8709677419
#
# If this does not reproduce, stop.
# ============================================================

EXPECTED_GE3 = (
    27 / 31
)

actual_ge3 = (
    observed[
        "kh_diagonal_ge_3_prop"
    ]
)

print(
    "\nStage-23B reproduction check:"
)

print(
    f"actual   = {actual_ge3:.6f}"
)

print(
    f"expected = {EXPECTED_GE3:.6f}"
)

print(
    f"difference = "
    f"{actual_ge3 - EXPECTED_GE3:+.8f}"
)


if not np.isclose(
    actual_ge3,
    EXPECTED_GE3,
    atol=1e-6,
):

    raise RuntimeError(
        "\nSTOP.\n"
        "The conditional n=31 selection does not reproduce "
        "the Stage-23B size-matched no-Khovanov benchmark.\n"
        "Do not run inference until the selection construction "
        "matches Stage 23B exactly."
    )


print(
    "\nPASS: conditional n=31 "
    "selection reproduces Stage 23B."
)


# ============================================================
# 6. Structural covariates
# ============================================================

test_meta = (
    meta
    .iloc[test_idx]
    .reset_index(drop=True)
)

crossing = (
    test_meta[
        "number_of_crossings"
    ]
    .to_numpy(int)
)

alternating = (
    test_meta[
        "is_alternating"
    ]
    .to_numpy(int)
)

signature_abs = (
    test_meta[
        "signature"
    ]
    .abs()
    .to_numpy()
)


# ============================================================
# 7. Validation-derived amplitude bins
# ============================================================
#
# Conditioning variables:
#
#   Alexander norm
#   Jones norm
#   HOMFLY-PT norm
#   Theta norm
#   Khovanov norm
#
# Khovanov is withheld from selection.
# Its norm is included only as a nuisance control.
# ============================================================

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

    edges = np.unique(
        edges
    )

    if len(edges) <= 2:

        return np.zeros(
            len(test_values),
            dtype=np.int16,
        )

    bins = np.searchsorted(
        edges[1:-1],
        test_values,
        side="right",
    )

    return bins.astype(
        np.int16
    )


def make_joint_strata(
    n_bins,
):

    data = {

        "crossing":
            crossing,

        "alternating":
            alternating,

        "signature_abs":
            signature_abs,
    }

    for name in ALL_NORM_VIEWS:

        if (
            "val_log_norm"
            not in score_by_view[name]
        ):
            raise KeyError(
                f"{name} missing val_log_norm"
            )

        if (
            "test_log_norm"
            not in score_by_view[name]
        ):
            raise KeyError(
                f"{name} missing test_log_norm"
            )

        data[
            f"norm_{name}"
        ] = validation_quantile_bins(

            score_by_view[name][
                "val_log_norm"
            ],

            score_by_view[name][
                "test_log_norm"
            ],

            n_bins=n_bins,
        )

    frame = pd.DataFrame(
        data
    )

    codes, _ = pd.factorize(
        pd.MultiIndex.from_frame(
            frame
        ),
        sort=False,
    )

    return codes.astype(
        np.int32
    )


# ============================================================
# 8. Fixed-cardinality joint-label randomization
# ============================================================

def prepare_sampler(
    strata,
    selected_mask,
):

    strata = np.asarray(
        strata
    )

    selected_mask = np.asarray(
        selected_mask,
        dtype=bool,
    )

    payload = []
    fixed_selected = 0

    for code in np.unique(
        strata
    ):

        members = np.flatnonzero(
            strata == code
        )

        n_selected = int(
            selected_mask[
                members
            ].sum()
        )

        if n_selected == 0:
            continue

        payload.append(
            (
                members,
                n_selected,
            )
        )

        if (
            n_selected
            == len(members)
        ):
            fixed_selected += (
                n_selected
            )

    return (
        payload,
        fixed_selected,
    )


def sample_fixed_cardinality(
    sampler,
    rng,
):

    mask = np.zeros(
        N_TEST,
        dtype=bool,
    )

    for (
        members,
        n_selected,
    ) in sampler:

        if (
            n_selected
            == len(members)
        ):

            chosen = members

        else:

            chosen = rng.choice(
                members,
                size=n_selected,
                replace=False,
            )

        mask[
            chosen
        ] = True

    if (
        mask.sum()
        != N_SELECT
    ):

        raise AssertionError(
            (
                mask.sum(),
                N_SELECT,
            )
        )

    return mask


# ============================================================
# 9. Null engine
# ============================================================

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

    (
        sampler,
        fixed_selected,
    ) = prepare_sampler(
        strata,
        selected,
    )

    selected_indices = (
        np.flatnonzero(
            selected
        )
    )

    selected_cell_sizes = np.asarray([
        np.sum(
            strata
            == strata[i]
        )
        for i
        in selected_indices
    ])

    rng = (
        np.random.default_rng(
            seed
        )
    )

    values = {

        metric:
            np.empty(
                n_reps,
                dtype=float,
            )

        for metric
        in METRICS
    }

    for b in range(
        n_reps
    ):

        null_selected = (
            sample_fixed_cardinality(
                sampler,
                rng,
            )
        )

        result = endpoint_values(
            null_selected
        )

        for metric in METRICS:

            values[
                metric
            ][b] = (
                result[
                    metric
                ]
            )

        if (
            (b + 1)
            % 500
            == 0
        ):

            print(
                f"{n_bins:2d} bins: "
                f"{b + 1:,}/"
                f"{n_reps:,}"
            )

    summary_rows = []

    for metric in METRICS:

        null_values = (
            values[
                metric
            ]
        )

        obs = (
            observed[
                metric
            ]
        )

        n_ge = int(
            np.sum(
                null_values
                >= obs
            )
        )

        p_emp = (
            n_ge + 1
        ) / (
            n_reps + 1
        )

        summary_rows.append({

            "bins_per_view":
                n_bins,

            "metric":
                metric,

            "selected_n":
                N_SELECT,

            "observed":
                obs,

            "null_mean":
                float(
                    np.mean(
                        null_values
                    )
                ),

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

            "n_ge_observed":
                n_ge,

            "n_randomizations":
                n_reps,

            "p_emp":
                p_emp,

            "median_selected_cell_size":
                float(
                    np.median(
                        selected_cell_sizes
                    )
                ),

            "fixed_selected":
                int(
                    fixed_selected
                ),
        })

    return (
        pd.DataFrame(
            summary_rows
        ),
        values,
    )


# ============================================================
# 10. Full grid trajectory
# ============================================================

grid_frames = []

for n_bins in GRID_BINS:

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CONDITIONAL no-Khovanov null: "
        f"{n_bins} bins/view"
    )

    print(
        "=" * 70
    )

    frame, _ = run_null(

        n_bins=n_bins,

        n_reps=GRID_REPS,

        seed=(
            SEED
            + n_bins
        ),
    )

    grid_frames.append(
        frame
    )


grid_summary = pd.concat(
    grid_frames,
    ignore_index=True,
)


grid_summary.to_csv(

    OUT
    / "conditional_no_khovanov_grid_trajectory.csv",

    index=False,
)


print(
    "\nGRID TRAJECTORY — "
    "MEAN KHOVANOV DIAGONAL COUNT"
)


display(

    grid_summary.loc[

        grid_summary[
            "metric"
        ].eq(
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


# ============================================================
# 11. Final paper-facing 2-bin null
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "FINAL CONDITIONAL "
    "2-BIN PAPER-FACING NULL"
)

print(
    "=" * 70
)


(
    paper_summary,
    paper_null,
) = run_null(

    n_bins=PAPER_BINS,

    n_reps=PAPER_REPS,

    seed=SEED,
)


# Holm correction across four endpoints
(
    reject,
    p_holm,
    _,
    _,
) = multipletests(

    paper_summary[
        "p_emp"
    ].to_numpy(),

    method="holm",
)


paper_summary[
    "holm_p"
] = p_holm


paper_summary[
    "holm_reject_005"
] = reject


paper_summary[
    "z_null"
] = (

    paper_summary[
        "observed"
    ]

    -

    paper_summary[
        "null_mean"
    ]

) / paper_summary[
    "null_sd"
]


paper_summary.to_csv(

    OUT
    / "conditional_no_khovanov_final_joint_null.csv",

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


# ============================================================
# 12. Continuous-amplitude sensitivity
# ============================================================

if (
    "phenotype_path"
    not in globals()
):

    matches = list(
        Path(ROOT).rglob(
            "complete_mathematical_phenotype_atlas.parquet"
        )
    )

    if len(matches) != 1:

        raise RuntimeError(
            "Could not uniquely locate "
            "complete_mathematical_phenotype_atlas.parquet. "
            f"Found {len(matches)} candidates."
        )

    phenotype_path = (
        matches[0]
    )


print(
    "\nPhenotype atlas:",
    phenotype_path,
)


phenotype_full = (
    pd.read_parquet(
        phenotype_path
    )
)


# ============================================================
# 13. Determine knot ID column
# ============================================================

if (
    "ID_COL"
    not in globals()
):

    id_candidates = (
        "knot",
        "knot_id",
        "name",
        "identifier",
    )

    ID_COL = next(
        (
            c
            for c
            in id_candidates
            if (
                c in phenotype_full.columns
                and c in meta.columns
            )
        ),
        None,
    )

    if ID_COL is None:

        raise KeyError(
            "Could not determine ID_COL."
        )


print(
    "Using knot identifier:",
    ID_COL,
)


phenotype_full[
    ID_COL
] = (
    phenotype_full[
        ID_COL
    ]
    .astype(str)
)


phenotype_full = (
    phenotype_full
    .set_index(
        ID_COL
    )
)


# ============================================================
# 14. Determine Khovanov diagonal-count column
# ============================================================

KH_DIAG_CANDIDATES = (
    "khovanov_q_minus_2t_diagonal_count",
    "kh_diagonal_count",
    "khovanov_diagonal_count",
)


KH_DIAG_COL = next(
    (
        c
        for c
        in KH_DIAG_CANDIDATES
        if c
        in phenotype_full.columns
    ),
    None,
)


if KH_DIAG_COL is None:

    raise KeyError(
        "Could not identify Khovanov "
        "diagonal-count column."
    )


print(
    "Using Khovanov outcome:",
    KH_DIAG_COL,
)


# ============================================================
# 15. Validation/test outcome arrays
# ============================================================

val_ids = (

    meta
    .iloc[val_idx][
        ID_COL
    ]
    .astype(str)
    .to_numpy()
)


test_ids_local = (

    meta
    .iloc[test_idx][
        ID_COL
    ]
    .astype(str)
    .to_numpy()
)


y_val = (

    phenotype_full
    .reindex(
        val_ids
    )[
        KH_DIAG_COL
    ]
    .to_numpy(
        float
    )
)


y_test = (

    phenotype_full
    .reindex(
        test_ids_local
    )[
        KH_DIAG_COL
    ]
    .to_numpy(
        float
    )
)


if (
    np.isnan(
        y_val
    ).any()
    or
    np.isnan(
        y_test
    ).any()
):

    raise RuntimeError(
        "Missing Khovanov diagonal "
        "outcomes in validation/test."
    )


# ============================================================
# 16. Continuous nuisance features
# ============================================================

def continuous_features(
    indices,
    split_name,
):

    m = (
        meta
        .iloc[indices]
    )

    columns = []

    # All five representation log norms
    for name in ALL_NORM_VIEWS:

        key = (
            f"{split_name}_log_norm"
        )

        if (
            key
            not in score_by_view[
                name
            ]
        ):

            raise KeyError(
                f"{name} missing {key}"
            )

        columns.append(

            np.asarray(
                score_by_view[
                    name
                ][
                    key
                ],
                dtype=float,
            )
        )

    # Structural covariates
    columns.extend([

        m[
            "number_of_crossings"
        ].to_numpy(
            float
        ),

        m[
            "is_alternating"
        ].to_numpy(
            float
        ),

        m[
            "signature"
        ]
        .abs()
        .to_numpy(
            float
        ),
    ])

    return np.column_stack(
        columns
    )


X_val_nuisance = (
    continuous_features(
        val_idx,
        "val",
    )
)


X_test_nuisance = (
    continuous_features(
        test_idx,
        "test",
    )
)


print(
    "\nContinuous nuisance feature shapes:"
)

print(
    "validation:",
    X_val_nuisance.shape
)

print(
    "test:",
    X_test_nuisance.shape
)


# ============================================================
# 17. Fit nuisance model on validation only
# ============================================================

nuisance_model = (
    HistGradientBoostingRegressor(

        learning_rate=0.05,

        max_iter=300,

        max_leaf_nodes=31,

        l2_regularization=1.0,

        random_state=SEED,
    )
)


nuisance_model.fit(
    X_val_nuisance,
    y_val,
)


pred_test = (
    nuisance_model.predict(
        X_test_nuisance
    )
)


residual_test = (
    y_test
    - pred_test
)


observed_residual = float(

    np.mean(
        residual_test[
            selected
        ]
    )
)


print(
    "\nObserved conditional "
    "mean residual:",
    observed_residual,
)


# ============================================================
# 18. Structural-only randomization of residual endpoint
# ============================================================

structural_frame = pd.DataFrame({

    "crossing":
        crossing,

    "alternating":
        alternating,

    "signature_abs":
        signature_abs,
})


(
    structural_strata,
    _,
) = pd.factorize(

    pd.MultiIndex.from_frame(
        structural_frame
    ),

    sort=False,
)


(
    structural_sampler,
    structural_fixed,
) = prepare_sampler(

    structural_strata,
    selected,
)


rng = (
    np.random.default_rng(
        SEED + 1000
    )
)


residual_null = np.empty(
    PAPER_REPS,
    dtype=float,
)


for b in range(
    PAPER_REPS
):

    null_selected = (
        sample_fixed_cardinality(

            structural_sampler,

            rng,
        )
    )

    residual_null[
        b
    ] = np.mean(

        residual_test[
            null_selected
        ]
    )

    if (
        (b + 1)
        % 500
        == 0
    ):

        print(
            "Continuous null: "
            f"{b + 1:,}/"
            f"{PAPER_REPS:,}"
        )


n_ge = int(

    np.sum(
        residual_null
        >= observed_residual
    )
)


continuous_p = (

    n_ge + 1

) / (

    PAPER_REPS + 1

)


continuous_summary = pd.DataFrame([{

    "analysis":
        "conditional_no_khovanov_n31",

    "selected_n":
        N_SELECT,

    "observed_mean_diagonal":
        float(
            np.mean(
                y_test[
                    selected
                ]
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
        int(
            structural_fixed
        ),
}])


continuous_summary.to_csv(

    OUT
    / "conditional_no_khovanov_continuous_amplitude.csv",

    index=False,
)


print(
    "\nCONTINUOUS-AMPLITUDE SENSITIVITY"
)


display(
    continuous_summary
)


# ============================================================
# 19. Save selected knots
# ============================================================

selected_ids = (

    meta
    .iloc[
        test_idx[
            selected
        ]
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
    "conditional_C3"
] = (
    conditional_c3[
        selected
    ]
)


selected_ids[
    "stored_khovanov_diagonal_count"
] = (
    kh_diag[
        selected
    ]
)


selected_ids[
    "stored_khovanov_support_size"
] = (
    kh_support[
        selected
    ]
)


selected_ids.to_csv(

    OUT
    / "conditional_no_khovanov_selected_n31.csv",

    index=False,
)


# ============================================================
# 20. Final compact summary
# ============================================================

print(
    "\n"
    + "=" * 72
)

print(
    "STAGE 23D COMPLETE"
)

print(
    "=" * 72
)


print(
    "\nObserved:"
)

display(
    pd.DataFrame([
        observed
    ])
)


print(
    "\nGrid trajectory:"
)

display(

    grid_summary.loc[

        grid_summary[
            "metric"
        ].eq(
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


print(
    "\nFinal 2-bin inference:"
)

display(
    paper_summary[
        [
            "metric",
            "observed",
            "null_mean",
            "z_null",
            "p_emp",
            "holm_p",
            "fixed_selected",
        ]
    ]
)


print(
    "\nContinuous amplitude:"
)

display(
    continuous_summary
)


print(
    "\nSaved Stage 23D to:"
)

print(
    OUT
)