# Design rationale — bayesian_diagnosticity experiment 1

## Overview

20 stimuli selected from a pool of 164 candidate pairs via EIG over the prior-predictive distributions of two competing PyMC models.

## Candidate generation

All H/T sequences of length 4–8 were enumerated (496 total). After filtering to sequences with at least one alternation and imbalance < 0.6, and de-duplicating by H↔T symmetry, 217 sequences remained. Pairs were constructed to maximize model disagreement, yielding 164 candidates across disagreement levels and sequence lengths (same-length and cross-length pairs included).

## EIG scoring

EIG was computed from 200 prior-predictive draws per model per stimulus (uniform model prior). EIG range across the top 20 stimuli: **0.038 – 0.057 bits**.

## What the design discriminates

The two models share only the `imbalance` feature; they diverge on:

- **Prototype similarity** cares about `p_alts` — sequences whose alternation rate is close to the learned ideal (theta_alt ∈ [0.35, 0.95]) look more random.
- **Encoding compressibility** cares about `max_run_norm` and `periodicity` — sequences with shorter runs and no repeating template look more random.

The high-EIG pairs all contrast:
- A sequence with **low alternation but low periodicity** (e.g., HHHHTTTT — all-same-then-all-other style, periodicity ≈ 0)
- A sequence with **higher alternation but high periodicity** (e.g., THHTTHHT — period-4 repeating, periodicity = 1.0)

Prototype similarity tends to prefer the more-alternating sequence (closer to theta_alt), while encoding compressibility penalises the high periodicity of the same sequence. This creates genuine predictive disagreement, maximising the information gain from each participant response.
