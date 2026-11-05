# Critique of `bigelow_piantadosi_independent`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **in_range_effect** — Among sets with >=2 examples, yes-rate for targets inside [min example, max example] minus yes-rate for targets outside that range; observed above null_mean means people favour interpolation within the example span more than the model (interval-like generalization under-produced), below means the model over-weights range. (observed 0.0848 vs null mean 0.0189, z=2.68, p=0.00599, q=0.04) [survives FDR]
- **last_digit_match_effect** — Yes-rate for targets whose last digit equals the last digit of at least one example minus yes-rate for targets matching none; observed above null_mean means people generalize by shared final digit (even partially/for mixed sets) more than the model, below means less. (observed 0.312 vs null mean 0.25, z=2.60, p=0.00999, q=0.04) [survives FDR]
