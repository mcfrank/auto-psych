# Critique of `rsa_l2_singleton_feat_color_valence_l0`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **prior_featureless_object_points** — Mean points allocated on wordless prior trials (heard is NaN) to completely featureless objects (feature_count == 0) in scenes that include a featureless object. (observed 3.42 vs null mean 2.99, z=3.43, p=0.002, q=0.016) [survives FDR]
- **utt_runner_up_complexity_gap** — Mean difference in points per object between intermediate 2-feature literal matches (feature_count == 2) and high-complexity literal matches (feature_count >= 3) on utterance trials that also contain a 1-feature literal match (feature_count == 1). (observed 0.217 vs null mean 0.535, z=-2.92, p=0.00599, q=0.016) [survives FDR]
- **utt_isolated_preempted_match_points** — Mean points allocated on utterance trials to perceptually isolated literal matches (minimum Hamming distance >= 2 to all other display objects) that also possess a unique unmentioned feature (extension == 1), in scenes where another literal match has a close neighbor (minimum Hamming distance <= 1). (observed 4.23 vs null mean 4.04, z=2.69, p=0.00599, q=0.016) [survives FDR]
