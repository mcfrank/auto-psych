# Critique of `rsa_l2_singleton_feat_color_valence_l0`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **utterance_all_true_vs_subset_true_rich_singleton_probe_rating_diff** — On utterance trials where second_probe is a true singleton with at least 3 features, mean second_rating when the word is true of all objects (with one singleton) minus when it is true of a proper subset including a 1-feature object. (observed 1.41 vs null mean 0.706, z=7.66, p=0.002, q=0.00533) [survives FDR]
- **all_true_utterance_rich_singleton_choice_rate** — On 4-object utterance trials where the heard word is true of every object and one unique singleton has strictly more features than the three identical clones, the rate of choosing the singleton. (observed 0.417 vs null mean 0.259, z=5.22, p=0.002, q=0.00533) [survives FDR]
- **utterance_featureless_distractor_all_feature_competitor_choice_rate** — On utterance trials containing at least one featureless object (all 0s), at least one 1-feature true object, and a unique all-feature object (all 1s), the rate of choosing the all-feature object. (observed 0.16 vs null mean 0.238, z=-3.51, p=0.002, q=0.00533) [survives FDR]
- **utterance_sparse_singleton_vs_rich_clones_clone_choice_rate** — On utterance trials where the heard word applies to one unique singleton and at least two identical clones with strictly more features than the singleton, the rate of choosing one of the true clones. (observed 0.182 vs null mean 0.145, z=2.66, p=0.012, q=0.024) [survives FDR]
