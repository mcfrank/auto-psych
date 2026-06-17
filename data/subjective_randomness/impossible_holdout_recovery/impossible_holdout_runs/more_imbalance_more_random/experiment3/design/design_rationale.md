# Design Rationale

**Number of stimuli:** 20
**EIG Range:** 0.3900 – 0.5067

## Rationale

The top candidate stimuli chosen by EIG primarily contrast highly homogeneous, imbalanced sequences (e.g., `HHHHH`, `TTTTT`, `TTTTTTT`) against sequences with balanced outcome frequencies and moderate-to-high alternation rates (e.g., `THTTHTHH`, `THTHHTH`). 

In the previous experiment, the highest posterior probability was assigned to `inner_loop_model` and `iter1_candidate2`, which assert that people paradoxically perceive sequences with greater imbalance as more random. These hypotheses are highly distinct from alternative models like `prototype_similarity` (which favors balanced sequences with ideal alternation rates) or `raw_alternation_count` (which strictly relies on transitions). By contrasting extreme imbalance with high-alternation/balanced sequences, the new design aims to definitively distinguish whether the perceived randomness is driven by extreme class imbalance or if other structural properties (like alternations or streaks) are necessary.

The stimuli lengths vary between 5 and 8 to provide variation in length and enable discrimination against models that normalize by sequence length.
