# Critique of `short_sequence_pseudoflip_dilution`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **low_switch_pairs_more_switches** — Among pairs of length >= 6 where both sequences have at most 2 switches and they differ in switch count, the rate of choosing the more-switching sequence; observed above null means people reward the first one or two breaks of a streak more than the model predicts, below means less. (observed 0.872 vs null mean 0.839, z=3.01, p=0.00599, q=0.048) [survives FDR]
- **high_switch_pairs_more_switches** — Among length-8 pairs where both sequences have >= 5 switches and they differ in switch count, the rate of choosing the more-switching sequence; observed below null means people penalise over-alternation near the top of the switch range more than the model predicts, above means less. (observed 0.36 vs null mean 0.333, z=1.95, p=0.05, q=0.2)
