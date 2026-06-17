# Experiment 1 Design Rationale

**Number of stimuli:** 30 pairs
**EIG Range:** 0.1443 – 0.2999

## Design Methodology
We generated a pool of 300 random candidate stimulus pairs (each sequence in the pair was drawn uniformly from all possible sequences of lengths 3 through 8). We then scored each pair by computing Expected Information Gain (EIG) over the theorist's PyMC models (`bayesian_diagnosticity` and `encoding_compressibility`) using prior-predictive sampling. The top 30 pairs with the highest EIG were selected for the experiment.

## Discrimination Strategy
The selected stimulus pairs exhibit strong contrasts that differentiate the candidate cognitive models. For instance, many pairs match highly uniform or extreme sequences (e.g., `HHHHHHHH`, `TTTTTTTT`) against highly alternating or balanced sequences (e.g., `THHTHTHT`, `THTHHTHT`), often with differing lengths (e.g., length 4 vs length 8). 

These specific comparisons effectively discriminate between the models because the theories weigh sequence length, alternation rate, and run lengths differently. By maximizing EIG, the design pinpoints the sequence pairs where `bayesian_diagnosticity` and `encoding_compressibility` make the most divergent predictions regarding which sequence participants will find "more random."
