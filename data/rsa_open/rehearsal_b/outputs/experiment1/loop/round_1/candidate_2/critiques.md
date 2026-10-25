# Critique of `rsa_l2_singleton_feat_color_valence_l0`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **points_on_minimal_targets_when_word_is_universal** — Total points allocated to 1-feature duplicate targets in informative scenes where the heard word is true of all objects and competitors have multiple features. (observed 7.08 vs null mean 4.87, z=6.22, p=0.002, q=0.00266) [survives FDR]
- **scalar_implicature_points_on_full_feature_competitor** — Mean points allocated to the 3-feature competitor object in informative scalar implicature scenes where the heard word is shared between a 1-feature target and a 3-feature competitor. (observed 2.75 vs null mean 1.73, z=5.23, p=0.002, q=0.00266) [survives FDR]
- **points_on_singleton_competitor_with_duplicate_targets** — Mean points allocated to a 2-feature singleton competitor in informative scenes where the heard word is also true of duplicate 1-feature minimal targets. (observed 2.55 vs null mean 4.39, z=-5.10, p=0.002, q=0.00266) [survives FDR]
- **duplicate_targets_minus_singleton_competitor_points** — Total points on duplicate 2-feature targets minus points on a 3-feature singleton competitor in informative scenes where the heard word is true of all three. (observed 1.52 vs null mean -1.36, z=3.85, p=0.002, q=0.00266) [survives FDR]
- **prior_points_partial_minus_full_feature_singleton** — Mean point difference between partial-feature singletons and full-feature singletons in prior trials containing empty duplicate distractors. (observed 0.525 vs null mean -1.22, z=3.75, p=0.002, q=0.00266) [survives FDR]
- **points_on_minimal_singleton_with_duplicate_competitors** — Mean points allocated to a 1-feature minimal singleton in informative scenes where the heard word is true of all objects and competitors are multi-feature duplicates. (observed 8.47 vs null mean 7.88, z=3.44, p=0.002, q=0.00266) [survives FDR]
- **prior_points_empty_object_with_partial_duplicates** — Mean points allocated to the zero-feature empty object in prior trials containing duplicate 1-feature objects and a 2-feature singleton. (observed 4.12 vs null mean 3.12, z=2.96, p=0.00999, q=0.00999) [survives FDR]
- **prior_points_on_empty_singleton_with_full_duplicates** — Mean points allocated to the empty zero-feature singleton object in prior trials where all other objects are identical full-feature duplicates. (observed 6.9 vs null mean 6.31, z=2.81, p=0.00599, q=0.00685) [survives FDR]
