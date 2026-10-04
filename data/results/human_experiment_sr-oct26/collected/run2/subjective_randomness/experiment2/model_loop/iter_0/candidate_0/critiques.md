# Critique of `heads_default_alternation_ideal`

5 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **balanced_choice_at_equal_switches** — Among pairs whose two sequences have the same number of switches but different |#H-#T| imbalance, the rate of choosing the more balanced sequence; observed above null means the model under-weights heads/tails balance (it only has a linear heads-share term, no symmetric balance term). (observed 0.677 vs null mean 0.527, z=7.22, p=0.002, q=0.00533) [survives FDR]
- **shorter_longest_run_at_equal_switches** — Among pairs with equal switch counts but different longest-run lengths, the rate of choosing the sequence with the shorter longest run; observed above null means the model under-penalizes long streaks beyond what the switching rate implies. (observed 0.625 vs null mean 0.508, z=6.21, p=0.002, q=0.00533) [survives FDR]
- **symmetric_pattern_avoidance** — Among pairs where exactly one sequence is a palindrome or a complement-mirror (e.g. HHTT-like reversals) but neither is periodic, the rate of choosing the non-symmetric sequence; observed above null means people penalize visible symmetry that the model (only periodicity and switching) does not capture. (observed 0.305 vs null mean 0.373, z=-5.12, p=0.002, q=0.00533) [survives FDR]
- **pair_choice_rate_dispersion** — Mean over unordered stimulus pairs of |rate of choosing the lexicographically-first sequence - 0.5|; observed above null means people are more consensual (extreme) on individual pairs than the model predicts, below means the model is overconfident. (observed 0.23 vs null mean 0.215, z=2.53, p=0.00799, q=0.016) [survives FDR]
- **participant_side_bias_variance** — Variance across participants of their proportion of Left choices; observed above null means individual side/response biases (or lapsing) that the model lacks and so under-produces. (observed 0.00507 vs null mean 0.00373, z=2.47, p=0.02, q=0.032) [survives FDR]
