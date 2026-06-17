# Design Rationale — Experiment 3 (encoding_compressibility)

## Summary

- **Candidate pool**: 200 pairs (top-200 by surrogate discrimination score from ~29,000 eligible pairs of length-4 to length-8 sequences)
- **EIG scoring**: prior-predictive EIG over 6 competing PyMC models with uniform prior
- **Selected stimuli**: 20 pairs
- **EIG range**: 0.1145 – 0.1170 (all selected stimuli are near-equal in informativeness)

## Models in competition

| Model | Key feature(s) | Monotonic? |
|---|---|---|
| `prototype_similarity` | imbalance + \|p_alts − θ_alt\| | No (ideal ~0.65) |
| `inner_loop_model` | imbalance² + (p_alts − θ_alt)² | No (squared, steeper) |
| `rle_description_length` | (alts + 1) / n | Yes (more alts always better) |
| `max_run_length` | max_run_norm | Yes (shorter run always better) |
| `bayesian_diagnosticity` | log P(fair) − log P(alternating/streaky/biased) | No |
| `head_balance` *(new)* | imbalance only | Ignores alternation entirely |

## Dominant contrast

The top-EIG stimuli are all instances of the same structure: a **moderately alternating but imbalanced 7-character sequence** (e.g., TTHTTHT: 2H 5T, 4 transitions, max_run=2) versus a **perfectly balanced but nearly sorted 8-character sequence** (TTTTHHHH or HHHHTTTT: 4H 4T, 1 transition, max_run=4).

This contrast maximally pits:
- **head_balance** — predicts the sorted sequence is *more* random (better H/T balance, imbalance=0 vs 0.43)
- **rle_description_length** and **max_run_length** — predict the alternating sequence is *more* random (more transitions, shorter max run)
- **prototype_similarity / inner_loop_model** — moderate alternation (p_alts=0.67) vs. very low alternation (p_alts=0.14); prototype prefers the alternating sequence
- **bayesian_diagnosticity** — sorted sequence resembles a streaky generator, so also predicts alternating sequence is more random

The head_balance model is uniquely diagnostic here: it is the only model that selects the sorted sequence as more random, because it ignores alternation structure entirely.

## Secondary contrast

Stimuli 13–14 (HHHHTTTT vs TTHTTHTT / HHTHHTHH) provide a secondary check: the sorted sequence against a moderately alternating one with slight imbalance, allowing further discrimination of head_balance from the alternation-sensitive models.

Stimuli 15–20 involve shorter (4-character) sequences against TTTTHHHH/HHHHTTTT, which provide additional power at shorter sequence lengths.

## How this design discriminates

The experiment is designed to identify whether participants weight:
1. **H/T balance** as the primary cue (head_balance model)
2. **Alternation rate** as the primary cue (rle, max_run, prototype)
3. A combination of both (prototype, inner_loop, bayesian)

If the ground truth is `encoding_compressibility` (the holdout model from this run), these stimuli should produce response patterns inconsistent with head_balance and consistent with alternation-sensitive models.
