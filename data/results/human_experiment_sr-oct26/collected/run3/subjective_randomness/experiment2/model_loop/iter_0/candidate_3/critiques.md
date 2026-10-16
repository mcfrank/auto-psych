# Critique of `length_normalised_chance_vs_motif`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **frac_participants_prefer_fewer_switches** — Fraction of participants who, on pairs differing in switch count, choose the sequence with FEWER switches more than half the time; observed above null_mean means some people hold a reversed (streaky-is-random) preference that the model's shared-sign sensitivity cannot represent. (observed 0.188 vs null mean 0.0503, z=6.17, p=0.002, q=0.016) [survives FDR]
- **participant_side_bias_variance** — Variance across participants of their rate of choosing the LEFT sequence; observed above null_mean means people have idiosyncratic side (left/right) response biases the model lacks. (observed 0.00571 vs null mean 0.00421, z=2.53, p=0.016, q=0.0639)
