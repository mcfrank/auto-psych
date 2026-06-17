# Experiment 2 Design Rationale

**Number of stimuli:** 30 stimulus pairs were selected using Expected Information Gain (EIG) over a pool of 300 randomly generated candidate pairs (lengths 3 to 8).
**EIG Range:** The final selected 30 stimuli have an EIG ranging from 0.1661 to 0.3030.

**Design Strategy & Discriminability:**
The selected stimuli effectively contrast sequences with extreme properties against balanced, more "prototypical" sequences. For example, pairs like `"TTTT"` vs `"HTHTTHTH"` and `"HTHTHTTH"` vs `"TTTTTTTT"` pit highly streak-heavy or periodic sequences against highly alternating, low-streak ones. 

This design strongly discriminates between the current set of competing cognitive theories:
- **`absolute_ideal_run` vs `inner_loop_model` / `ideal_run_proportion`**: By comparing full-streak sequences of different lengths (e.g., length 4 vs length 8), the design separates whether people penalize streaks based on their absolute length or their normalized proportion relative to the total sequence length.
- **`pure_periodicity_penalty`**: Contrasts sequences based purely on periodic repeating patterns versus unstructured ones.
- **`bayesian_diagnosticity`, `encoding_compressibility`, and `prototype_similarity`**: These theories weight run lengths, alternations, and sequence balance differently. 

By employing pairs with varying lengths and contrasting maximum runs, alternations, and imbalance directly, the stimuli maximize the expected information gain across the theoretical landscape.