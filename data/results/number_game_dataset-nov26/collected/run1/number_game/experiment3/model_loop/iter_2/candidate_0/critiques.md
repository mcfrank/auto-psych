# Critique of `coincidence_decisive_exemplar`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **incoherent_set_cell_polarization** — Among (set, target) cells from single-example sets and incoherent multi-example sets (no shared final digit, parity or multiple-of-5 among examples), fraction of cells whose yes-rate is <=1/9 or >=8/9; observed > null means people answer all-or-none even without a suspicious coincidence (decisiveness is not confined to coherent sets as the model assumes), observed < null means the model is too polarized there. (observed 0.216 vs null mean 0.137, z=3.50, p=0.004, q=0.032) [survives FDR]
