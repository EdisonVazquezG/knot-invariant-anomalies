# Paper results and code

Numbers in script names are retained as stable provenance identifiers. They need
not appear in publication prose. Paths below are relative to the processed run.

| Result | Source script | Key output |
|---|---|---|
| Five anomaly scores | `stage23_anomaly_score_baselines.py` | `23_anomaly_score_baselines/score_selected_test_ids.csv` |
| Fixed-cardinality score overlap | `stage23B_size_matched_score_sensitivity.py` | `23B_size_matched_score_sensitivity/size_matched_selected_ids.csv` |
| View configurations and 31-member selection | `stage33_view_ablation_and_fibers.py` | `33_view_ablation_and_fibers/heldout_view_ablation_n31_summary.csv` |
| Joint fixed-cardinality null | `stage29b_final_joint_null.py` | `29b_final_joint_null/` |
| Fifteen-crossing amplitude grid | `stage42_crossing15_grid.py` | `42_crossing15_grid_direct/crossing15_grid_summary.csv` |
| Theta preprocessing | `stage41_theta_preprocessing.py` | `41_theta_preprocessing_direct_fixed/theta_sensitivity_summary.csv` |
| Theta numerical precision | `stage44_theta_float64_verification.py` | `44_theta_float64_verification/theta_float64_summary.csv` |
| Cross-partition tuple overlap | `stage43_split_tuple_audit.py` | `43_split_tuple_audit/split_tuple_overlap_summary.csv` |
| Polynomial fibers and width differences | `stage45_exact_fiber_khovanov_audit.py` | `45_exact_fiber_khovanov_audit/fiber_summary.csv` |
| Exact diagram verification | `stage46_independent_diagram_verification.py` | `46_independent_diagram_verification/verification_summary.json` |
| Additional partition grouped by fibers | `stage47_fiber_grouped_validation.py` | `47_fiber_grouped_validation/main_text_comparison.csv` |
| Corrected mirror assignment experiment | `stage31B_corrected_four_view_mirror_withheld_khovanov.py` | `31B_corrected_four_view_mirror_withheld_khovanov/four_view_mirror_withheld_khovanov_summary.csv` |
| Width thresholds | `stage34_turaev_width_consequence_audit.py` | `34_turaev_width_consequence_audit/WKh_ge4_selected_count_audit.csv` |
| Jones/Khovanov convention audit and cancellation | `stage36_euler_cancellation_endpoint.py` | `36_euler_cancellation_endpoint/` |
| Residual-compression sensitivity | `stage38_residual_compression_sensitivity.py` | `38_residual_compression_sensitivity/` |

The historical compression sweep wrote nested sensitivity directories. Preserve
those original outputs and their provenance. The corrected script creates sibling
directories for subsequent runs. The collector records the actual relative paths;
it does not relabel old outputs as newly computed results.

## Interpretation boundaries confirmed from code

- The view comparison changes both the view set and the aggregation rule
  (2 of 2 versus 3 of 4). Its results describe these joint configurations.
- Exact width pairs were identified by fiber enumeration. The diagram archive
  does not assert that the anomaly detector discovered them.
- The diagram code contains exact HOMFLY-to-Alexander specialization for its
  checked examples. The cancellation audit compares Jones and Khovanov across
  the atlas. Neither is an atlas-wide cross-table audit of all four polynomial
  views. Such an audit was not added or claimed here.
- PCA reconstruction uses the fitted transformation and inverse transformation.
  A nonzero test-sample mean alone does not invalidate the centered projection
  interpretation. This packaging pass does not alter centering or scores.
