# Critique of `parity_preserving_decisive_route`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **coherent_set_far_violation_rate** — Yes-rate for targets more than 20 from every example that break the set's coincidence (wrong final digit for same-digit sets, else not a multiple of 5 for all-multiple-of-5 sets, else wrong parity for same-parity sets), among 2-4-example coherent sets; observed > null means the model's steepened slope drives these rejections too close to zero (people keep a residual yes floor), observed < null means the model accepts them too often. (observed 0.123 vs null mean 0.154, z=-2.80, p=0.00799, q=0.0586)
- **single_example_cell_polarization** — Among (set, target) cells from one-example sets, fraction of cells whose yes-rate is <=1/9 or >=8/9; observed > null means people answer all-or-none after a lone example even though the model (no coincidence, so its least-steep slope) predicts graded answers there; observed < null means the model is too polarized there. (observed 0.248 vs null mean 0.159, z=2.54, p=0.018, q=0.0586)
- **incoherent_multi_cell_polarization** — Among (set, target) cells from 2-4-example sets with no coincidence (examples differ in parity, are not all multiples of 5, and do not share a final digit), fraction of cells with yes-rate <=1/9 or >=8/9; observed > null means people are all-or-none even after incoherent multi-example sets (decisiveness not confined to coherent sets), observed < null means the model over-polarizes them. (observed 0.187 vs null mean 0.114, z=2.44, p=0.022, q=0.0586)
