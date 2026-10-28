# Critique of `last_example_marks_range_top`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **above_below_asym_max_last** — In sets of >=2 whose largest example is shown last, yes-rate for targets 1-10 above the max minus yes-rate for targets 1-10 below the min; observed below null means people extend upward even less (the model under-credits the top edge), above null means the model over-sharpens the top edge. (observed -0.111 vs null mean -0.222, z=2.41, p=0.018, q=0.144)
