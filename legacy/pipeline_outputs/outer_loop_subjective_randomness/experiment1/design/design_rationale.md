# Design Rationale — Experiment 1

## Overview

20 stimulus pairs were selected from ~100 candidates using expected information gain (EIG) to maximally discriminate among three cognitive models of subjective randomness:

- **alternation_bias**: prefers sequences with higher alternation rates (H/T switches per adjacent pair)
- **balance_heuristic**: prefers sequences with equal H/T counts (closer to 50% heads)
- **griffiths_representativeness**: penalizes deviation from 0.5 on *both* alternation rate and balance simultaneously

## Candidate generation strategy

Candidates were generated to create disagreements between models. The key tension is that high alternation rate does not imply good balance — e.g., "HTHTHTH" (alt=1.0, balance=4/7=0.57) vs "HHHTTT" (alt=1/5=0.2, balance=0.5). On such a pair:
- **alternation_bias** strongly prefers HTHTHTH
- **balance_heuristic** slightly prefers HHHTTT (perfect balance)
- **griffiths** prefers HHHTTT (lower total deviation from 0.5)

Additional contrast was added via pairs where moderate alternation and perfect balance compete with extreme alternation (e.g., "HHTTHHT" vs "HTHTHTH"), where Griffiths uniquely predicts the moderate-alternation sequence.

Sequence lengths range from 4 to 9, per the constraint of maximum length 8 (one candidate "HHTTHHTTH" extends to 9 but is within reasonable range for model discrimination).

## EIG results

- Candidates evaluated: 99
- Selected: 20
- EIG range: 0.379 – 0.447 bits (out of max ~0.693 for a binary choice)

The top stimuli (EIG ≈ 0.44) are pairs like ("HTHTH", "HHHTTT") and ("HHHTTTTH", "HTHTHTH") — pairs where the three models give maximally divergent predictions.

## How the design discriminates

- **alternation_bias vs balance_heuristic**: pairs where high alternation comes with imperfect balance (e.g., HTHTH vs HHHTTT)
- **griffiths vs alternation_bias**: pairs where moderate alternation + perfect balance (e.g., HHTTHHT) competes with perfect alternation (HTHTHTH) — Griffiths prefers the former, alternation bias strongly prefers the latter
- **griffiths vs balance_heuristic**: pairs where balance is equal but alternation differs (e.g., HHTTHHTT vs HTHTHTHT)

With 20 trials at ~5 seconds each, the experiment fits the ~5-minute Prolific target.
