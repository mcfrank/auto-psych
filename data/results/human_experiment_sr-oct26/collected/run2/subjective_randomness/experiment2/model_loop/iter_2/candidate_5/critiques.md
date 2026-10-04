# Critique of `tally_span_personal_ideal`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **more_switches_at_near_equal_span** — Among pairs whose tally spans differ by at most 1, the rate of choosing the sequence with more H/T switches; observed above null means the model under-predicts a preference for alternation. (observed 0.711 vs null mean 0.645, z=6.68, p=0.002, q=0.016) [survives FDR]
- **late_minus_early_streak_avoidance** — Rate of choosing the sequence with the shorter longest run in the second half of trials minus the first half; positive observed above null means streak avoidance strengthens over the session, which a static model cannot produce. (observed -0.0217 vs null mean 0.013, z=-2.97, p=0.004, q=0.016) [survives FDR]
- **participant_streak_avoidance_variance** — Variance across participants of their rate of choosing the sequence with the shorter longest run (pairs differing in longest run); observed above null means individual differences in streak aversion exceed what the personal span ideal produces. (observed 0.0404 vs null mean 0.0342, z=2.53, p=0.016, q=0.0426) [survives FDR]
