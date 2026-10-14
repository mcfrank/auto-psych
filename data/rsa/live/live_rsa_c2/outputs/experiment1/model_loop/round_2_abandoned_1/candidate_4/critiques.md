# Critique of `feature_redundancy_discounted_l2`

8 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **mumble_intermediate_feature_rate** — Proportion of choices for objects with strictly intermediate feature counts among mumble trials where the display contains objects of at least three distinct feature counts. (observed 0.164 vs null mean 0.305, z=-5.28, p=0.002, q=0.00266) [survives FDR]
- **competitor_with_redundant_features_rate** — Proportion of choices for the overspecified competitor object in utterance trials where the competitor possesses at least two redundant alternative features. (observed 0.295 vs null mean 0.221, z=4.95, p=0.002, q=0.00266) [survives FDR]
- **duplicate_competitor_choice_rate** — Proportion of choices for the competitor class among utterance trials where the heard word is true of a unique singleton target and a competitor object that has duplicate copies in the scene. (observed 0.319 vs null mean 0.216, z=4.21, p=0.002, q=0.00266) [survives FDR]
- **false_feature_choice_rate** — Proportion of choices for an object of which the heard utterance is false across all utterance trials. (observed 0.0222 vs null mean 0.0121, z=3.62, p=0.002, q=0.00266) [survives FDR]
- **competitor_with_non_redundant_features_rate** — Proportion of choices for the overspecified competitor object in utterance trials where the competitor possesses distinguishing features that are non-redundant. (observed 0.208 vs null mean 0.321, z=-3.57, p=0.002, q=0.00266) [survives FDR]
- **mumble_empty_object_rate** — Proportion of choices for the featureless object among four-object mumble trials containing at least one featureless object. (observed 0.355 vs null mean 0.265, z=3.33, p=0.004, q=0.00457) [survives FDR]
- **mumble_pairwise_rich_object_rate** — Proportion of choices for the object with more features among two-object mumble trials with unequal feature counts. (observed 0.728 vs null mean 0.615, z=3.00, p=0.002, q=0.00266) [survives FDR]
- **two_object_scalar_target_rate** — Proportion of choices for the minimal target object among two-object utterance trials where the heard word is true of both objects. (observed 0.678 vs null mean 0.766, z=-2.73, p=0.014, q=0.014) [survives FDR]
