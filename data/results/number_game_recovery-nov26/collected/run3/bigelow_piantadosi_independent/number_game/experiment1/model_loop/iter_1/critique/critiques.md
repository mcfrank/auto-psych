# Critique of `sharpened_map_concept_selection`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **parity_match_effect** — Among sets whose examples all share parity, yes rate for parity-matching targets minus parity-mismatching targets; observed < null means the model over-commits to parity (odd/even) rules relative to people, observed > null means people use parity more. (observed 0.415 vs null mean 0.2, z=10.53, p=0.002, q=0.00799) [survives FDR]
- **inside_vs_outside_range** — Yes rate for targets strictly inside the [min,max] span of the examples minus yes rate for targets outside that span (sets with 2+ examples); observed > null means people treat the example range as a region more than the model does. (observed 0.11 vs null mean 0.0499, z=3.41, p=0.002, q=0.00799) [survives FDR]
- **near_far_similarity_gradient** — Yes rate for targets within 3 of the nearest example minus yes rate for targets 15+ away from every example (excluding targets equal to an example); observed > null means people show more graded nearby-number generalization than the sharpened model produces. (observed 0.0725 vs null mean 0.129, z=-2.50, p=0.016, q=0.0426) [survives FDR]
