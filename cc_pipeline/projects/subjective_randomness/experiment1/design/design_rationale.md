# Design Rationale — Experiment 1 (subjective_randomness)

## Summary

Selected **30 stimulus pairs** from 600 candidates (sequence lengths 4, 6, 8; same-length and mixed-length pairs), scored by Expected Information Gain (EIG) under uniform prior over three cognitive models.

## Models

| Model | Description |
|-------|-------------|
| `representativeness` | Prefers sequences closer to 50/50 H/T balance (balance heuristic) |
| `alternation` | Prefers sequences with more H-T/T-H transitions (alternation bias) |
| `griffiths_representativeness` | Prefers sequences with higher likelihood under a Markov chain with p_alternation=0.7 (rational basis of representativeness, Griffiths & Tenenbaum) |

## EIG Statistics

- **EIG range**: 0.0551 – 0.2666 bits
- All 30 stimuli have EIG > 0

## Selection Strategy

Stimuli were selected using a diversity-aware round-robin over all three theory pairs:
- `representativeness` vs `alternation`
- `representativeness` vs `griffiths_representativeness`
- `alternation` vs `griffiths_representativeness`

For each pair, the stimulus with highest pairwise EIG (under a 50/50 prior on that pair alone) was selected in rotation, ensuring the design collectively discriminates across all three models rather than optimizing only for the globally highest-EIG stimuli.

## Key Design Features

The highest-EIG stimuli (EIG ≈ 0.267) contrast sequences like `TTTHHH` (low alternations, imbalanced) vs `HTHHTH` (high alternations, near-balanced), which create maximally divergent predictions across models: the `alternation` and `griffiths_representativeness` models strongly prefer the alternating sequence, while `representativeness` is more sensitive to H/T balance. This triangulation is the key diagnostic contrast in the experiment.
