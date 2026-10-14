# Critique of `rsa_l2_singleton_feat_color_familiar_l0`

5 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **target_rate_competitor_duplicated** — Rate of choosing the minimal-feature target in ambiguous utterance trials where at least one competing matching object has an identical duplicate in the display. (observed 0.655 vs null mean 0.804, z=-7.16, p=0.002, q=0.004) [survives FDR]
- **uninformative_word_min_feature_rate** — Rate of choosing the minimal-feature object in utterance trials where the heard word is true of every object in the display. (observed 0.66 vs null mean 0.774, z=-5.44, p=0.002, q=0.004) [survives FDR]
- **target_rate_large_feature_gap** — Rate of choosing the minimal-feature target in ambiguous utterance trials where the competing matching object has at least 3 more features than the target. (observed 0.697 vs null mean 0.803, z=-5.33, p=0.002, q=0.004) [survives FDR]
- **prior_intermediate_features_rate** — Rate of choosing intermediate-complexity objects on mumble trials where the display contains both extreme (empty and full) objects and intermediate objects. (observed 0.191 vs null mean 0.313, z=-3.97, p=0.002, q=0.004) [survives FDR]
- **prior_singleton_rate_with_duplicates** — Rate of choosing any visual singleton object on mumble trials where the display contains both singleton and duplicated objects. (observed 0.557 vs null mean 0.622, z=-2.66, p=0.00599, q=0.00959) [survives FDR]
