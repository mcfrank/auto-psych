# Critique of `magnitude_threshold_concepts`

5 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **wide_minus_mid_span_in_range** — For sets with >=2 examples, in-range non-example yes-rate when span>=50 minus when span is 15-49; observed > model means wide-span sets are filled in more relative to mid spans than the model predicts. (observed 0.125 vs null mean 0.0595, z=3.27, p=0.002, q=0.00799) [survives FDR]
- **mixed_parity_set_in_minus_out_range** — For sets of >=2 examples with mixed parity, yes-rate on non-example targets inside [min,max] minus yes-rate outside; observed > model means people rely more on magnitude range for mixed-parity sets than the model. (observed 0.119 vs null mean 0.0349, z=3.12, p=0.002, q=0.00799) [survives FDR]
- **set_level_sd_nonexample_rate** — Standard deviation across number sets of the per-set yes-rate on non-example targets; observed > model means sets differ more sharply in breadth of generalization than the model produces. (observed 0.23 vs null mean 0.21, z=2.75, p=0.00599, q=0.016) [survives FDR]
- **two_example_out_of_range_rate** — For two-example sets, yes-rate on targets outside [min,max]; observed > model means people generalize beyond the examples range more than the model with two examples. (observed 0.222 vs null mean 0.258, z=-2.47, p=0.018, q=0.032) [survives FDR]
- **four_example_above_minus_below_range** — For 4-example sets, yes-rate on targets above the max minus yes-rate on targets below the min; observed > model means a stronger upward asymmetry than the model predicts. (observed 0.0686 vs null mean 0.00487, z=2.29, p=0.02, q=0.032) [survives FDR]
