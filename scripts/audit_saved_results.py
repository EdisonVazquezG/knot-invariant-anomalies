#!/usr/bin/env python3
"""Check internal consistency of supplied result tables, without refitting models.

This audit reconstructs memberships and summaries from saved scores and labels.
It does not recompute polynomial coefficients, homology, PCA fits or calibration.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
ID='knot_id_base'


def audit(root):
    checks=[]
    def record(name, condition, detail):
        checks.append({'check':name,'passed':bool(condition),'detail':detail})
    def read(name):return pd.read_csv(root/name)
    def equal(a,b):return bool(np.allclose(a,b,rtol=1e-10,atol=1e-10))
    # All ordered pairs, both equal-cardinality families, from stable IDs.
    ids=read('23B_size_matched_score_sensitivity/size_matched_selected_ids.csv')
    tab=read('23B_size_matched_score_sensitivity/size_matched_score_jaccard.csv')
    sets={(f,m):set(g[ID]) for (f,m),g in ids.groupby(['family','method'])}
    ok=not ids.duplicated(['family','method',ID]).any()
    for row in tab.itertuples():
        a,b=sets[row.family,row.method_a],sets[row.family,row.method_b]
        expected_n=60 if row.family=='All 5 C3' else 31
        ok &= len(a)==len(b)==expected_n and len(a&b)==row.overlap and equal(len(a&b)/len(a|b),row.jaccard)
    record('fixed_cardinality_jaccard_from_ids',ok,{'comparisons':len(tab)})
    # Mirror membership statistics.
    mirror=read('31B_corrected_four_view_mirror_withheld_khovanov/four_view_mirror_withheld_khovanov_summary.csv')
    mids=read('31B_corrected_four_view_mirror_withheld_khovanov/four_view_mirror_selected_ids.csv')
    ms={k:set(g[ID]) for k,g in mids.groupby('assignment')};base=ms['baseline_canonical']
    ok=not mids.duplicated(['assignment',ID]).any()
    for row in mirror.itertuples():
        a=ms[row.assignment]
        ok &= len(a)==row.selected_n and len(a&base)==row.overlap_with_canonical
        ok &= equal(len(a&base)/len(a|base),row.jaccard_with_canonical)
        ok &= equal(len(a&base)/len(base),row.canonical_recovered_prop)
    record('mirror_memberships_from_ids',ok,{'assignments':len(mirror),'canonical_n':len(base)})
    # Fixed width thresholds from the explicit lists.
    counts=read('34_turaev_width_consequence_audit/WKh_ge4_selected_count_audit.csv')
    lists=read('34_turaev_width_consequence_audit/WKh_ge4_selected_knot_lists.csv')
    ok=not lists.duplicated(['selection',ID]).any()
    for row in counts.itertuples():
        g=lists.loc[lists.selection.eq(row.selection)]
        ok &= len(g)==row.n_WKh_ge_4 and g.stored_F_diagonal_count_WKh.ge(4).all()
        ok &= equal(len(g)/row.selected_n,row.prop_WKh_ge_4)
    record('width_threshold_counts_from_lists',ok,counts[['selection','selected_n','n_WKh_ge_4']].to_dict('records'))
    # Partition separation with the stored exact-input group labels.
    splits=read('47_fiber_grouped_validation/split_assignments.csv')
    crossing=int(splits.groupby('exact_joint_fiber').grouped_partition.nunique().gt(1).sum())
    common=set(splits.loc[splits.frozen_partition.eq('test') & splits.grouped_partition.eq('test'),ID])
    size=read('47_fiber_grouped_validation/split_sizes.csv');ok=len(splits)==313230 and not splits[ID].duplicated().any()
    for row in size.itertuples():
        col='frozen_partition' if row.design=='frozen' else 'grouped_partition'
        ok &= int(splits[col].eq(row.partition).sum())==row.n
    record('partition_sizes_and_joint_group_separation',ok and crossing==0 and len(common)==6982,
           {'atlas_n':len(splits),'spanning_grouped_partitions':crossing,'common_test_n':len(common),
            'scope':'stored exact-input group labels; source coefficient equality not recomputed'})
    # Re-rank all four views within each eligible pool, not just previously selected IDs.
    scores=read('47_fiber_grouped_validation/all_test_scores.csv')
    selected=read('47_fiber_grouped_validation/selected_ids.csv')
    profiles=read('47_fiber_grouped_validation/headline_profiles.csv')
    cols=[v+'_conditional_percentile' for v in ['Alexander','Jones','HOMFLY_PT','Theta']]
    reconstructed={};ok=True
    for design,frame in scores.groupby('design'):
        frame=frame.copy();ok &= not frame[ID].duplicated().any()
        part='grouped_partition' if design=='grouped_refit_float64' else 'frozen_partition'
        expected=set(splits.loc[splits[part].eq('test'),ID]);ok &= set(frame[ID])==expected
        for pool in ['own_test','common_test']:
            f=frame if pool=='own_test' else frame.loc[frame[ID].isin(common)].copy()
            ranks=f[cols].rank(method='average').to_numpy()/len(f)
            c3=np.sort(ranks,axis=1)[:,1] # third largest of four
            if pool=='own_test':ok &= equal(c3,f.C3.to_numpy())
            order=np.lexsort((f[ID].to_numpy(str),-c3))[:31]
            found=set(f.iloc[order][ID]);reconstructed[design,pool]=found
            stored=selected.loc[selected.design.eq(design)&selected.evaluation_pool.eq(pool)]
            ok &= len(stored)==31 and set(stored[ID])==found
    record('six_selections_reranked_from_saved_per_view_scores',ok,{'selections':len(reconstructed),'per_selection_n':31})
    ok=True
    for row in profiles.itertuples():
        f=selected.loc[selected.design.eq(row.design)&selected.evaluation_pool.eq(row.evaluation_pool)]
        ok &= len(f)==row.selected_n and equal(f.W_F.mean(),row.mean_W_F)
        ok &= equal(f.W_F.ge(3).mean(),row.P_W_F_ge_3) and equal(f.W_F.ge(4).mean(),row.P_W_F_ge_4)
        ok &= equal(f.is_alternating.mean(),row.alternating_fraction)
    record('width_and_alternation_profiles_from_members',ok,{'profiles':len(profiles)})
    a=reconstructed['frozen_refit_float64','common_test'];b=reconstructed['grouped_refit_float64','common_test']
    record('common_test_membership_overlap',len(a&b)==28 and len(a|b)==34,
           {'intersection':len(a&b),'union':len(a|b),'jaccard':len(a&b)/len(a|b)})
    # Check the reported finite-simulation floor and within-family Holm correction.
    null=read('29b_final_joint_null/paper_facing_joint_null_results.csv')
    kh=null.loc[null.analysis.eq('noKh_external_Khovanov')]
    record('canonical_null_pvalue_bookkeeping',len(kh)==4 and kh.B.eq(5000).all()
           and equal(kh.empirical_p_upper,1/5001) and equal(kh.Holm_p_within_family,4/5001),
           {'endpoints':len(kh),'scope':'summary bookkeeping only; permutation draws not included'})
    v=json.loads((root/'46_independent_diagram_verification/verification_summary.json').read_text())
    record('recorded_diagram_verification_status',v['status']=='FULL_VERIFIED' and not v['errors']
           and all(p['exact_four_view_pair_verified'] and p['width_difference'] for p in v['pairs']),
           {'status':v['status'],'pairs':len(v['pairs']),'scope':'recorded certificate; no new Sage or homology execution'})
    manifest=json.loads((REPO/'evidence/collected_results_manifest.json').read_text())
    bad=[]
    for row in manifest['included_files']:
        p=root/row['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:bad.append(row['path'])
    record('bundled_result_file_integrity',not bad,{'files':len(manifest['included_files']),'mismatches':bad})
    return {'status':'SAVED_RESULTS_CHECKS_PASS' if all(x['passed'] for x in checks) else 'SAVED_RESULTS_CHECKS_FAIL',
            'fresh_full_execution_verified':False,'models_refitted':False,'checks':checks}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=REPO/'paper_results')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=audit(a.root)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])
    for row in result['checks']:print(('PASS' if row['passed'] else 'FAIL'),row['check'])
    raise SystemExit(0 if result['status']=='SAVED_RESULTS_CHECKS_PASS' else 2)

if __name__=='__main__':main()
