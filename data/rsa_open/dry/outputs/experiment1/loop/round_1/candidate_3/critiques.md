# Critique of `rsa_l2_singleton_feat_color_valence_l0`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **prior_four_obj_three_duplicate_singleton** — On 4-object prior trials containing a unique singleton alongside three identical duplicate distractors, the mean points allocated to the singleton object. (observed 7.71 vs null mean 6.69, z=6.38, p=0.002, q=0.00228) [survives FDR]
- **prior_competing_sparse_minus_dense** — On prior trials with duplicate 0-feature distractors and two singletons (sparse 1-feature vs dense all-features), the mean points allocated to the sparse singleton minus the dense singleton. (observed -0.625 vs null mean -3.15, z=6.14, p=0.002, q=0.00228) [survives FDR]
- **complex_competitor_minus_distractor_gap** — On communicative trials with both a feature-rich competitor (true of the heard message) and a feature-rich distractor (false of the heard message), the mean point difference between the competitor and the distractor. (observed 1.24 vs null mean 0.55, z=6.10, p=0.002, q=0.00228) [survives FDR]
- **three_object_scalar_competitor_points** — On 3-object communicative trials with a unique minimal target and an uninformative competitor satisfying the message, the mean points allocated to the competitor object. (observed 1.65 vs null mean 1.05, z=4.85, p=0.002, q=0.00228) [survives FDR]
- **prior_duplicate_background_points** — On prior trials containing exactly one singleton among identical duplicate distractors, the total points allocated across all duplicate distractor objects combined. (observed 2.79 vs null mean 3.44, z=-4.78, p=0.002, q=0.00228) [survives FDR]
- **maximal_vs_intermediate_competitor_preference** — On communicative trials where the heard word fits a minimal target, an intermediate competitor, and a maximal-feature competitor, the mean point difference between the maximal competitor and the intermediate competitor. (observed 0.0375 vs null mean -0.531, z=4.30, p=0.002, q=0.00228) [survives FDR]
- **prior_two_singletons_point_gap** — On prior trials containing exactly two singletons alongside duplicate background distractors, the mean absolute difference in points allocated between the two singletons. (observed 2.98 vs null mean 3.69, z=-3.84, p=0.002, q=0.00228) [survives FDR]
- **prior_all_singleton_full_complexity_points** — On prior trials where all objects are distinct singletons with ascending feature counts from zero to all features, the mean points allocated to the all-features object. (observed 1.62 vs null mean 2.6, z=-3.11, p=0.004, q=0.004) [survives FDR]
