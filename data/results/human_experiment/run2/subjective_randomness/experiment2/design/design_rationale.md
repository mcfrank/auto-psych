# Design Rationale

**Candidate Generation**:
A set of 300 random pairs of coin flip sequences (H and T) with lengths between 2 and 8 was generated as the candidate pool. This provided a tractable pool to compute Expected Information Gain (EIG).

**Stimulus Selection**:
EIG was computed over the 7 cognitive models in the theoretical pool (`bayesian_diagnosticity`, `encoding_compressibility`, `falk_konold_difficulty`, `inner_loop_model`, `manhattan_messy_prototype`, `prototype_similarity`, `window_typicality`). The top 32 stimuli were greedily selected to maximize discriminatory power between these models based on their prior-predictive choice probabilities.

**EIG Range**:
The final set of 32 selected stimuli pairs has an EIG range of approximately 0.185 to 0.269, with all stimuli possessing strictly positive EIG values.

**Discrimination Strategy**:
The chosen sequence pairs represent strong informational contrasts along the structural dimensions modeled by the theories. They contrast differences in sequence lengths, alternation rates, longest runs, and general sequence prototypes, enabling an experimental measurement capable of successfully differentiating the theoretical mechanisms underlying subjective randomness.
