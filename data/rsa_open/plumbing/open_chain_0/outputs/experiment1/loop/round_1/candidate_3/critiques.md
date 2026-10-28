# Critique of `rsa_l2_singleton_feat_color_valence_l0`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **heard_all_true_max_point_concentration** — Mean maximum points placed on any single object on trials where the heard word is true of all four objects in the display, testing whether the model under- or over-predicts trial-level betting concentration when semantics rules out no referent. (observed 7.04 vs null mean 6.65, z=7.31, p=0.002, q=0.00799) [survives FDR]
- **heard_duplicate_simple_vs_singleton_complex_diff** — Mean difference in points between the two identical simple true referents (summed) and the singleton complex true referent in heard-word scenes where the word applies to a duplicate pair of fewer-feature objects and a singleton object with more features. (observed 4.24 vs null mean 3.66, z=3.51, p=0.004, q=0.0107) [survives FDR]
- **heard_false_referent_points_leakage** — Mean total points allocated to semantically false objects (objects lacking the heard feature) on trials where a word was heard and at least one object lacks that feature, testing whether the model's uniform lapse and Dirichlet-multinomial link mispredict points given to false referents. (observed 0.426 vs null mean 0.511, z=-2.88, p=0.002, q=0.00799) [survives FDR]
- **heard_max_feature_competitor_points** — Mean points allocated to the most feature-rich true referent in heard-word scenes where multiple objects satisfy the heard word and differ in feature count, testing whether depth-2 RSA over-eliminates complex competitors via scalar implicature. (observed 2.37 vs null mean 2.21, z=2.65, p=0.012, q=0.024) [survives FDR]
