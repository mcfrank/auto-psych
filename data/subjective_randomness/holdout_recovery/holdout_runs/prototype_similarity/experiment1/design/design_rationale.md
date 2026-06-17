# Design rationale — prototype_similarity experiment 1

## Models being discriminated

Two cognitive models are competing to explain subjective randomness judgments:

- **bayesian_diagnosticity**: A sequence looks random if it is likely under a fair-coin generator
  relative to alternating, biased, and streaky alternatives. Key features: `n`, `h`, `alts`.
- **encoding_compressibility**: A sequence looks random if it resists simple compression — low
  max-run length, low periodicity, balanced H/T counts. Key features: `max_run_norm`,
  `periodicity`, `imbalance`.

## Candidate pool

200 candidate pairs were generated using six strategies targeting the discriminating axes between the models:

1. **High-alts vs high-max-run** (same length, same head count): directly pits the models'
   key metrics against each other.
2. **Biased vs alternating**: sequences where one is heavily head- or tail-biased and the other
   is near-perfectly alternating.
3. **Periodic but short-run vs. aperiodic but long-run**: targets encoding_compressibility's
   internal periodicity vs. max_run tradeoff, while bayesian_diagnosticity may be indifferent.
4. **Imbalanced+short-run vs balanced+long-run**: encoding penalizes imbalance and max_run
   differently; bayesian penalizes bias strongly.
5. **Diverse sampling across alternation count** for lengths 6–8.
6. **Near-perfect alternators vs. near-perfect streamers**: the sharpest contrast for
   bayesian_diagnosticity's alternating vs. streaky classification.

Pairs were scored by a heuristic combining delta-alts, delta-max_run_norm, delta-periodicity,
and delta-imbalance; the top 200 by this discriminating score were forwarded to EIG.

## EIG results

- **N selected**: 20 stimuli
- **EIG range**: 0.3203 – 0.3327 nats
- The highest-EIG stimuli are alternating-vs-all-same-outcome pairs (e.g., HTHTHTHT vs HHHHHHHH).

  These maximize model disagreement because:
  - `bayesian_diagnosticity` classifies both as highly non-random (alternating vs. biased
    generators), and the relative degree depends on fitted alt_prior and bias_share parameters.
  - `encoding_compressibility` classifies HTHTHTHT as high-periodicity (non-random) but low
    max_run and balanced, while HHHHHHHH has maximum imbalance and max_run — a different
    penalty profile — so the preferred-left prediction can flip depending on fitted weights.

  Responses to these stimuli thus reduce posterior uncertainty about which model family
  (or which parameter regime within a model) governs human judgments.

## Stimulus schema

Each stimulus in `stimuli.json` is a dict:
```json
{"sequence_a": "HTHTHTHT", "sequence_b": "HHHHHHHH", "eig": 0.3327}
```
Sequence lengths range from 4 to 8 characters drawn from {H, T}.
