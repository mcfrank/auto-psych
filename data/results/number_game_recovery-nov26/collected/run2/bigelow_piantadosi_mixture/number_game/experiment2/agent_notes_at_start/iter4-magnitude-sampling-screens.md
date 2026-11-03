---
name: iter4-magnitude-sampling-screens
description: Round 4 cand_0 MAP screens on WOR incumbent (NLL 902.77): magnitude-biased sampling, interval-only evidence weight, rule-vs-range evidence temperature all peak at null
metadata:
  type: project
---
Round 4 candidate_0 (2026-10-30). Baseline is concepts_sampled_without_replacement, MAP NLL 902.77.
- Magnitude-biased sampling P(x|h) ∝ x^-β: β=0 is best. β=±0.1 costs 0.4-0.7 nats, β=0.5 costs 18 and β=-0.5 costs 10.
- An evidence exponent on intervals only reaches ρ≈1, gain 0.1.
- A temperature on the rule-vs-interval log-evidence ratio reaches τ≈0.98, no gain.
- Worst set residual: 83,38,58,68 (~20 nats; mid-range targets at 20-40% yes vs 8% predicted). None of the above fixes it.
I submitted `magnitude_biased_example_sampling` (β ~ TruncNormal(0, 0.2) on ±0.6, Chebyshev tabulated, 1.6 ms/grad, smoke loglik -904.7). It may fail the novelty gate if β shrinks to about 0.

**Why:** this confirms [[number-game-ceiling]] and [[iter3-noise-ceiling-wor]].
**How to apply:** don't re-screen these. The screener is at iter_4/candidate_0/scratch/screen.py.
