# Critique of `rsa_l2_singleton_distinct_feat_color_l0`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **competitor_with_redundant_features_rate** — Proportion of choices for the overspecified competitor object in utterance trials where the competitor possesses at least two redundant alternative features. (observed 0.295 vs null mean 0.201, z=7.06, p=0.002, q=0.00266) [survives FDR]
- **duplicate_competitor_choice_rate** — Proportion of choices for the competitor class among utterance trials where the heard word is true of a unique singleton target and a competitor object that has duplicate copies in the scene. (observed 0.319 vs null mean 0.194, z=5.49, p=0.002, q=0.00266) [survives FDR]
- **mumble_intermediate_feature_rate** — Proportion of choices for objects with strictly intermediate feature counts among mumble trials where the display contains objects of at least three distinct feature counts. (observed 0.164 vs null mean 0.308, z=-5.22, p=0.002, q=0.00266) [survives FDR]
- **scalar_competitor_choice_rate** — Proportion of trials choosing the overspecified competitor object among utterance trials where the heard word is true of both a minimal target and a competitor with strictly more features. (observed 0.28 vs null mean 0.22, z=4.80, p=0.002, q=0.00266) [survives FDR]
- **mumble_empty_object_rate** — Proportion of choices for the featureless object [0, 0, 0, 0] among four-object mumble trials containing at least one featureless object. (observed 0.355 vs null mean 0.251, z=4.05, p=0.002, q=0.00266) [survives FDR]
- **false_feature_choice_rate** — Proportion of choices for an object of which the heard utterance is false (literal semantic violation) across all utterance trials. (observed 0.0297 vs null mean 0.0165, z=3.57, p=0.002, q=0.00266) [survives FDR]
- **two_object_scalar_target_rate** — Proportion of choices for the minimal target object among two-object utterance trials where the heard word is true of both objects. (observed 0.678 vs null mean 0.781, z=-3.43, p=0.004, q=0.00457) [survives FDR]
- **mumble_pairwise_rich_object_rate** — Proportion of choices for the object with more features among two-object mumble trials with unequal feature counts. (observed 0.728 vs null mean 0.623, z=2.75, p=0.00999, q=0.00999) [survives FDR]
