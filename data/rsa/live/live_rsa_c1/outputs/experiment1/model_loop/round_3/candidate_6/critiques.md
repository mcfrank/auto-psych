# Critique of `attentive_display_listener`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **target_rate_competitor_duplicated** — Rate of choosing the minimal-feature target in ambiguous utterance trials where at least one competing matching object has an identical duplicate in the display. (observed 0.655 vs null mean 0.786, z=-5.96, p=0.002, q=0.00266) [survives FDR]
- **target_rate_ambiguity_breadth_diff** — Difference in minimal-feature target choice rate between narrow ambiguity (word fits exactly 2 objects) and broad ambiguity (word fits 3 or more objects). (observed 0.0457 vs null mean -0.0887, z=5.15, p=0.002, q=0.00266) [survives FDR]
- **uninformative_word_min_feature_rate** — Rate of choosing the minimal-feature object in utterance trials where the heard word is true of every object in the display. (observed 0.66 vs null mean 0.765, z=-5.07, p=0.002, q=0.00266) [survives FDR]
- **prior_intermediate_features_rate** — Rate of choosing intermediate-complexity objects on mumble trials where the display contains both extreme (empty and full) objects and intermediate objects. (observed 0.191 vs null mean 0.332, z=-4.51, p=0.002, q=0.00266) [survives FDR]
- **prior_max_feature_rate** — Rate of choosing the maximal-feature object on mumble trials where objects vary in total feature count. (observed 0.587 vs null mean 0.51, z=4.04, p=0.002, q=0.00266) [survives FDR]
- **target_rate_large_feature_gap** — Rate of choosing the minimal-feature target in ambiguous utterance trials where the competing matching object has at least 3 more features than the target. (observed 0.697 vs null mean 0.773, z=-3.60, p=0.002, q=0.00266) [survives FDR]
- **target_rate_2obj** — Rate of choosing the minimal-feature target in 2-object ambiguous utterance trials. (observed 0.676 vs null mean 0.746, z=-2.54, p=0.024, q=0.024) [survives FDR]
- **prior_singleton_rate_with_duplicates** — Rate of choosing any visual singleton object on mumble trials where the display contains both singleton and duplicated objects. (observed 0.557 vs null mean 0.616, z=-2.39, p=0.024, q=0.024) [survives FDR]
