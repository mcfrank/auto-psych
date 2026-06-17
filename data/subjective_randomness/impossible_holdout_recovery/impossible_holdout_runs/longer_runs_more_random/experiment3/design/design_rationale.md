# Experiment 3 Design Rationale

**Number of Stimuli:** 30
**Expected Information Gain (EIG) Range:** 0.2669 – 0.4398

**Rationale:**
To optimally discriminate between the 10 candidate cognitive models (e.g., `surprising_run_length`, `bayesian_diagnosticity`, `prototype_similarity`, `pure_periodicity_penalty`, etc.), we generated a candidate pool of 300 unique pairs composed of sequences varying in length between 4 and 8 coin flips. 

By computing the prior-predictive Expected Information Gain (EIG) for each pair, we selected the top 30 pairs. The resulting design pairs sequences that provoke maximal disagreement among the theorists' predictions. For example, comparing an extreme monotonic sequence like `"TTTTTTTT"` to a short, mixed sequence like `"HTTHT"` distinguishes models that heavily penalize long runs from those that are driven by overall head proportion, sequence length, or alternation rates. This ensures that the choices made by participants in these 30 trials will yield the highest possible statistical leverage for model selection and parameter estimation in the subsequent analysis loop.