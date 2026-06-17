# Design Rationale

- **Number of stimuli:** 30 pairs of coin flip sequences
- **EIG Range:** 0.1681 to 0.2837

## Discrimination Strategy
The design uses Expected Information Gain (EIG) to select 30 sequence pairs out of a pool of randomly generated candidates (lengths 3 to 8). The selected sequence pairs tend to pit sequences with highly alternating patterns (e.g., "HTHT", "THT", "HTHTH") against highly structured, non-alternating sequences or sequences of different lengths (e.g., "HHH", "TTT", "HHHH"). 

These pairs act to maximally discriminate between the current cognitive models (`bayesian_diagnosticity`, `encoding_compressibility`, and `prototype_similarity`). For instance, models relying on compressibility will score sequences like "HHH" vastly differently than models relying on prototype similarity or alternation rates. By selecting pairs that differ significantly across these feature dimensions, the experiment can tease apart which cognitive mechanism best predicts human subjective perception of randomness.