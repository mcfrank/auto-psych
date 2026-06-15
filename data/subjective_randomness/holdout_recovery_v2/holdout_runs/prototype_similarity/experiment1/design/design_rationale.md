# Design rationale — prototype_similarity experiment 1

## Models competing

- **encoding_compressibility**: penalizes sequences by weighted sum of `max_run_norm` + `periodicity` + `imbalance`. A sequence looks random if it has no long runs, no repeating template structure, and balanced H/T.
- **bayesian_diagnosticity**: scores sequences by log P(fair coin) minus log P(best non-random alternative) (alternating, streaky, biased generators). A sequence looks random if it is more diagnostic of a fair coin than of any structured generator.

## Key discriminating axis

The models diverge most on **imbalanced/streaky vs alternating** pairs:

- **Alternating sequences** (e.g. THTHTH, HTHT): low `max_run_norm` and zero `imbalance` so EC penalizes them only mildly (via `periodicity`). But BD penalizes them strongly — high alternation rate is maximally diagnostic of the alternating generator (switch_prob = 0.95).
- **Imbalanced/streaky sequences** (e.g. TTTH, TTTHTT, HHTHHH): high `max_run_norm` and/or high `imbalance` so EC penalizes heavily. BD also penalizes these (biased or streaky generators), but less severely than it penalizes the perfectly alternating sequences above.

Result: EC tends to prefer alternating sequences over imbalanced ones (low run + low imbalance outweigh periodicity); BD strongly prefers neither, but finds alternating sequences less random than moderately imbalanced ones.

## Candidate pool

- 250 candidate pairs drawn from 8 archetype categories: highly alternating, balanced-moderate, periodic (period > 2), balanced low-alternation, streaky, imbalanced, random-looking, and cross-length pairs.
- Enumerated all H/T sequences of length 4–8 and classified by `p_alts`, `imbalance`, `max_run`, and `periodicity`.

## Final stimuli

- **N = 20** pairs selected by EIG (expected information gain about model identity over 2 models with equal prior weight).
- **EIG range: 0.161 – 0.220** (out of max 1.0 bit for a 2-model prior).
- Top pairs all contrast imbalanced/streaky sequences (TTTH, TTTHTT, HHTHHH) against alternating sequences (THTHTH, THTH, HTHT, HTHTHTH, THHTHTHT).
- Several pairs use very long streaky sequences (TTTTTTT, HHHHHH) to probe EC's strong max_run penalty against BD's more moderate streaky-generator penalty.
- EIG values are substantially higher than in previous runs using intra-category pairs, consistent with the cross-category divergence hypothesis.
