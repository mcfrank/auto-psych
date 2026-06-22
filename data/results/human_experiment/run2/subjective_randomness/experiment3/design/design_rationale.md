# Experiment 3 Design Rationale

**Number of stimuli:** 32 pairs of coin-flip sequences.

**EIG Range:** The top 32 selected stimuli have Expected Information Gain (EIG) values ranging from 0.253 to 0.390.

**Candidate Generation Strategy:** We generated an initial tractable pool of 300 candidate pairs using random sampling from all possible sequences of length 2 to 8, while deliberately enforcing structural diversity:
- Fully random pairs
- Pairs of the exact same length
- Pairs of the same length but varying in their number of alternations
- Pairs of different lengths

**Discrimination between models:** The candidate pairs heavily vary over critical dimensions postulated by the competing cognitive models (length differences, alternation rates, imbalance, and streakiness). By selecting the pairs with the highest EIG, the pipeline guarantees that the resulting set represents those cases where the prior-predictive predictions of the competing models are most divergent. This optimized subset of 32 provides the strongest discriminatory power to pull apart models that weight these structural features differently (e.g., prototype similarity vs evidence accumulation vs encoding compressibility).
