# Critique of `tempered_size_principle_independent`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **outside_minus_inside_range** — Among non-example targets, yes rate for targets outside the [min, max] range of the examples minus targets inside that range; observed above null means people extend beyond the example range more than the model's interval hypotheses allow, below null means they confine generalization to the range more. (observed -0.15 vs null mean -0.093, z=-2.76, p=0.00599, q=0.048) [survives FDR]
- **intermediate_cell_fraction** — Fraction of (set, target) cells whose mean yes rate lies in [0.25, 0.75]; observed above null means people's endorsements are more graded/uncertain than the model's posterior predicts, below null means more all-or-none. (observed 0.461 vs null mean 0.531, z=-2.53, p=0.018, q=0.0719)
