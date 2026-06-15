# Theory Report — Experiment 2

## length_sensitive_prototype

**Motivation:** In experiment 1, all 8 models were statistically indistinguishable (all within
~2·dse of the best ELPD). The top model (`iter0_candidate1`) uses `imbalance` and
`|p_alts - theta_alt|` without any adjustment for sequence length, yet the task pairs sequences
that can differ in length (4–8 flips). Under a fair coin the standard deviation of p_alts
scales as 1/√(n−1), so a fixed deviation from theta_alt carries much more statistical weight
for an 8-flip sequence than for a 4-flip one. No experiment-1 model exploited this.

**Mechanism:** The penalty terms are multiplied by √(n): `imbalance · √n` and
`|p_alts − theta_alt| · √(n−1)`. This is equivalent to using a z-score-style evidence weight:
longer sequences provide stronger evidence of non-randomness for the same absolute deviation.
The rest of the structure (free theta_alt, alt_weight/balance_weight split, softmax with beta
and side_bias) is identical to the winning model, so the only new cognitive claim is that
people's sensitivity scales with evidence strength.

## asymmetric_alternation_prototype

**Motivation:** The winning model uses a symmetric absolute-distance penalty |p_alts − theta_alt|
that treats being too streaky and being too alternating as equally non-random. However, the
gambler's fallacy literature (Rabin 2002; Ayton & Fischer 2004) shows that people react more
strongly to runs than to excess alternation — they expect the next flip to reverse after a streak
but are less surprised by a further alternation. No experiment-1 model tested this asymmetry;
the three top models all used symmetric penalties.

**Mechanism:** The alternation penalty is split into two components:
`streak_k · max(theta_alt − p_alts, 0) + max(p_alts − theta_alt, 0)`. When `streak_k > 1`
the model penalises streakiness more than over-alternation; when `streak_k = 1` it reduces
exactly to the symmetric |p_alts − theta_alt| of the best experiment-1 model. `streak_k` has
a HalfNormal(0, 2) prior so the data can push it toward or away from symmetry. The balance-cue
(imbalance) term is retained unchanged.
