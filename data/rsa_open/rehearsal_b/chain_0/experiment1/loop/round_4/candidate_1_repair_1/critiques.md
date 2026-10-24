# Critique of `bounded_depth_listener`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **points_on_singleton_competitor_with_duplicate_targets** — Mean points allocated to a 2-feature singleton competitor in informative scenes where the heard word is also satisfied by duplicate 1-feature minimal targets. (observed 2.55 vs null mean 4.96, z=-6.75, p=0.002, q=0.00266) [survives FDR]
- **minimal_duplicates_minus_multifeature_singletons** — Total points on duplicate 1-feature targets minus total points on 2-feature singleton competitors in universal scenes where all four objects satisfy the heard word. (observed 4.15 vs null mean 0.145, z=5.60, p=0.002, q=0.00266) [survives FDR]
- **points_on_full_competitor_with_triplicate_targets** — Mean points allocated to the 3-feature singleton competitor in 4-object informative scenes where the heard word is shared with three 1-feature triplicate targets. (observed 2.73 vs null mean 1.54, z=4.81, p=0.002, q=0.00266) [survives FDR]
- **duplicate_targets_minus_singleton_competitor_points** — Total points allocated to duplicate 2-feature targets minus points allocated to a 3-feature singleton competitor in informative scalar implicature scenes where the heard word is true of all three. (observed 1.52 vs null mean -1.75, z=4.55, p=0.002, q=0.00266) [survives FDR]
- **points_on_minimal_target_in_all_match_display** — Mean points allocated to the unique 1-feature minimal target in 4-object informative scenes where all objects satisfy the heard word and competitors have 2 or 3 features. (observed 7.97 vs null mean 6.51, z=4.51, p=0.002, q=0.00266) [survives FDR]
- **points_on_scalar_competitor_with_duplicate_targets** — Mean points allocated to the 3-feature competitor in informative scalar implicature scenes where the heard word is shared with two 1-feature duplicate targets. (observed 3.7 vs null mean 3.01, z=3.83, p=0.002, q=0.00266) [survives FDR]
- **prior_empty_minus_partial_singleton_with_full_duplicates** — Mean point difference between an empty zero-feature singleton and a 2-feature partial singleton in prior trials containing duplicate full-feature objects. (observed 2.12 vs null mean -0.0267, z=3.44, p=0.00599, q=0.00685) [survives FDR]
- **prior_full_minus_sparse_in_all_distinct** — Mean point difference between the full-feature object and sparse objects in 3-object prior trials where all objects are distinct. (observed 1.15 vs null mean 0.018, z=2.31, p=0.028, q=0.028) [survives FDR]
