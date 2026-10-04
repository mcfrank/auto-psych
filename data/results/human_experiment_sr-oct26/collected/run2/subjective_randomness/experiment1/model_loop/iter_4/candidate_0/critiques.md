# Critique of `person_sensitivity_length_scaled_ideal`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **heads_majority_pref** — Among pairs with equal switch counts whose H-counts differ, rate of choosing the sequence with more H; the incumbent is H/T symmetric (null near 0.5), so observed above/below null_mean means people favour heads-heavy/tails-heavy sequences as more random. (observed 0.567 vs null mean 0.425, z=3.06, p=0.004, q=0.032) [survives FDR]
- **near_periodic_avoidance** — Among pairs with switch-rate difference at most 1/7 where exactly one sequence is near-periodic (period 2-4 with exactly one mismatch, not strictly periodic), rate of choosing the other sequence; observed above null_mean means people penalise almost-repeating patterns that the model's strict periodicity check misses. (observed 0.383 vs null mean 0.315, z=2.65, p=0.00999, q=0.04) [survives FDR]
- **shorter_longest_run_pref** — Among pairs with equal switch counts but different longest-run lengths, rate of choosing the sequence with the shorter longest run; observed above null_mean means the model under-penalises long streaks beyond what switch rate and periodicity explain. (observed 0.588 vs null mean 0.444, z=2.58, p=0.018, q=0.048) [survives FDR]
