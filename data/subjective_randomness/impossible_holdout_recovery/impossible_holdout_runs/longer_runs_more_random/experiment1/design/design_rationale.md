# Design Rationale

**Number of stimuli:** 30 pairs.
**EIG range:** 0.1570 to 0.2225.

**Design Generation:**
We uniformly sampled 300 sequence pairs with lengths varying between 4 and 8 characters (using H and T). By generating diverse lengths and compositions, we ensured a tractable candidate pool. 

**Model Discrimination:**
The stimuli were selected by maximizing Expected Information Gain (EIG) over the prior predictive distributions of the three candidate cognitive models (`bayesian_diagnosticity`, `encoding_compressibility`, and `prototype_similarity`). Pairs with the highest EIG (such as `THTH` vs `TTTHTTTT` or `HTHTTH` vs `HHHHH`) strongly discriminate the models because they pit highly alternating or representative sequences against highly unbalanced or constant sequences. Different models value sequence length, alternation rate, and imbalance differently, meaning their predictions diverge most strongly on these extreme pairs, maximizing the information gained from the participant's choices.
