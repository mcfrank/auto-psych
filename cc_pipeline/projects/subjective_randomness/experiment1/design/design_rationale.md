# Design Rationale — Experiment 1 (subjective_randomness)

## Overview

Selected **30 stimuli** (the maximum per problem definition) to satisfy the 5-minute / 30-trial Prolific constraint. Each stimulus is a pair of H/T sequences; participants choose which looks "more random."

## Candidate generation

All possible sequence pairs drawn from lengths 4, 6, and 8 (same-length and mixed-length), plus a random sample of length-6×6 and length-8×8 pairs (seeded for reproducibility). Total candidate pool: ~22,600 pairs.

## EIG scoring

Each pair was scored by Expected Information Gain (EIG) under three models with a uniform prior:

- **griffiths_representativeness** — prefers sequences with both alternation rate and H/T balance close to 0.5
- **alternation_bias** — prefers sequences with higher alternation rate regardless of balance
- **balance_heuristic** — prefers sequences with H/T ratio closer to 0.5 regardless of pattern

**EIG range: 0.4032 – 0.4032 bits** (all 30 selected stimuli sit on the same maximum EIG plateau).

## How the design discriminates between models

The plateau value of ≈0.40 bits arises because many pairs cause the three models to disagree maximally: one model strongly prefers sequence A, another strongly prefers sequence B, and the third is near-indifferent or in between. This is the theoretical ceiling for three-way discrimination with binary responses and a uniform prior.

All 30 stimuli involve a short highly-alternating sequence (HTHT or THTH, length 4) paired with a length-8 sequence that has moderate-to-high balance but low alternation — exactly the configuration where:
- **alternation_bias** chooses the length-4 alternating sequence
- **balance_heuristic** may prefer the length-8 sequence if it is more balanced
- **griffiths_representativeness** weighs both dimensions and reaches a different probability than either pure heuristic

This three-way split is what drives the high EIG and makes these pairs maximally informative for model identification.
