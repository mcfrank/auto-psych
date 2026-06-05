# Theory Report — Experiment 1

## bayesian_fair_coin

**Motivation:** This is the principal model derived from Griffiths & Tenenbaum (2001,
"The Rational Basis of Representativeness"). It provides the theoretically grounded
baseline against which the simpler feature-based models will be compared.

**Mechanism:** For each sequence the model computes the log Bayes factor (LBF)
of a fair-coin hypothesis against a single biased-coin alternative whose bias θ
is inferred from data (Beta(2,2) prior). The sequence with the higher LBF — the
one that is more diagnostic of a fair coin relative to a biased alternative — is
preferred. A softmax temperature τ controls choice stochasticity. Free parameters:
θ (alternative coin bias) and τ.

## alternation_bias

**Motivation:** The gambler's fallacy / negative recency bias is one of the most
robust findings in subjective randomness research (Kahneman & Tversky, 1972).
People systematically overestimate how often fair coins alternate. This model tests
whether alternation proportion alone — a single normalized feature — is sufficient
to account for forced-choice randomness judgements.

**Mechanism:** The model computes p_alts = alternations / (n − 1) for each
sequence and applies a softmax over the difference. It uses no head-count
information, making it identifiable from bayesian_fair_coin in cases where
alternation rate and LBF diverge (e.g., HTHTHT vs. HHTHTT have the same LBF
but different alternation rates). Free parameter: τ only.

## runs_penalty

**Motivation:** Long unbroken runs are a distinct cue to non-randomness that is
not fully captured by alternation proportion. A sequence can have moderate
alternation but a long streak at one end (e.g., HHHHTHT). This model tests
whether the single worst-case run length drives choices, independently of the
overall alternation rate.

**Mechanism:** The model scores each sequence by its maximum run length and
prefers the sequence with the shorter maximum run via softmax on the difference.
It uses only `max_run_a`/`max_run_b`, making its predictions differ from
alternation_bias precisely when the sequences have similar p_alts but different
streak profiles. Free parameter: τ only.
