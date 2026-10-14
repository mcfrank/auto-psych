# Critique of `softmax_belief_listener`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **target_rate_large_feature_gap** — Rate of choosing the minimal-feature target in ambiguous utterance trials where the competing matching object has at least 3 more features than the target. (observed 0.697 vs null mean 0.786, z=-4.32, p=0.002, q=0.00266) [survives FDR]
- **prior_intermediate_features_rate** — Rate of choosing intermediate-complexity objects on mumble trials where the display contains both extreme (empty and full) objects and intermediate objects. (observed 0.191 vs null mean 0.318, z=-4.16, p=0.002, q=0.00266) [survives FDR]
- **prior_singleton_rate_with_duplicates** — Rate of choosing any visual singleton object on mumble trials where the display contains both singleton and duplicated objects. (observed 0.557 vs null mean 0.653, z=-3.80, p=0.002, q=0.00266) [survives FDR]
- **prior_dup_full_rate_with_empty_singleton** — Rate of choosing the full duplicated object on mumble trials where the display contains an empty singleton object and a full object with duplicate copies. (observed 0.53 vs null mean 0.411, z=3.24, p=0.002, q=0.00266) [survives FDR]
- **target_rate_2obj** — Rate of choosing the minimal-feature target in 2-object ambiguous utterance trials. (observed 0.676 vs null mean 0.762, z=-3.17, p=0.002, q=0.00266) [survives FDR]
- **uninformative_word_min_feature_rate** — Rate of choosing the minimal-feature object in utterance trials where the heard word is true of every object in the display. (observed 0.66 vs null mean 0.721, z=-2.85, p=0.002, q=0.00266) [survives FDR]
- **target_rate_competitor_duplicated** — Rate of choosing the minimal-feature target in ambiguous utterance trials where at least one competing matching object has an identical duplicate in the display. (observed 0.655 vs null mean 0.718, z=-2.80, p=0.00599, q=0.00685) [survives FDR]
- **target_rate_scaling_4obj_vs_2obj** — Difference in pragmatic target choice rate between 4-object displays and 2-object displays in ambiguous utterance trials. (observed 0.022 vs null mean -0.0566, z=2.55, p=0.014, q=0.014) [survives FDR]
