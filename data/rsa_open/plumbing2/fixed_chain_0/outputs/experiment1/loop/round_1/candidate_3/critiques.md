# Critique of `rsa_l2_singleton_feat_color_valence_l0`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **singleton_specific_with_sparse_distractors_rate** — Proportion of choices selecting the minimal-feature singleton referent satisfying the heard word, in utterance scenes where all true referents are singletons of differing feature counts and false distractors are sparse (at most 1 feature each). (observed 0.738 vs null mean 0.657, z=3.02, p=0.004, q=0.024) [survives FDR]
- **all_true_single_feature_target_rate** — Proportion of choices selecting a 1-feature referent in utterance scenes where the heard word is true of every object in the display and at least one object has only that single feature. (observed 0.703 vs null mean 0.757, z=-2.82, p=0.00599, q=0.024) [survives FDR]
- **prior_intermediate_singleton_rate** — Proportion of choices selecting a singleton with an intermediate feature count (strictly between the scene minimum and maximum feature counts), on no-word prior trials containing such an object. (observed 0.311 vs null mean 0.267, z=2.51, p=0.00999, q=0.0266) [survives FDR]
- **prior_maximal_feature_clone_rate** — Proportion of choices selecting a maximal-feature duplicate object on no-word prior trials that contain both unique singletons and a duplicate pair of maximal-feature objects. (observed 0.253 vs null mean 0.296, z=-2.40, p=0.026, q=0.0519)
