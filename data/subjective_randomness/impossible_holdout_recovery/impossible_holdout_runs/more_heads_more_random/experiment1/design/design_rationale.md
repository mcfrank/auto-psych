# Experiment 1 Design Rationale

**Number of stimuli:** 30
**EIG range:** 0.1384 – 0.2511

## How the design discriminates between models

The generated design selects 30 stimulus pairs with the highest Expected Information Gain (EIG) from a diverse candidate pool of varying lengths (3-8) and binary sequences. 

The selected pairs frequently contrast highly predictable, low-complexity sequences (e.g., "TTTT", "HHHHHHHH") against highly alternating, irregular sequences (e.g., "HTHTTHTH", "THTHTHHT"). 

This design discriminates effectively between the provided cognitive models (`bayesian_diagnosticity`, `encoding_compressibility`, `prototype_similarity`) by exposing their differing sensitivities to sequence length, repetition vs. alternation biases, and absolute compression difficulty. For instance, `encoding_compressibility` will heavily penalize uniform sequences of any length because they compress easily, while `prototype_similarity` and `bayesian_diagnosticity` will have specific sensitivities to the length-adjusted binomial probabilities and run-length statistics that diverge significantly on these extreme pairs. By pitting extremes of alternation and length against each other, the design ensures the theories will make strongly diverging predictions.
