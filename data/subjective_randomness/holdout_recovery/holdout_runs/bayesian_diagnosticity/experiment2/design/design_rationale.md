# Experiment 2 Design Rationale

**Number of stimuli:** 30 pairs
**EIG Range:** 0.1081 to 0.3278

## Selection Process
We generated a candidate pool of 60 random sequence pairs, drawing from all possible sequences of lengths 3 through 8. For each candidate pair, we computed the Expected Information Gain (EIG) based on prior-predictive samples from the five competing models:
- `prototype_similarity`
- `encoding_compressibility`
- `inner_loop_model`
- `inferred_mixture_model`
- `unnormalized_mixture_model`

The top 30 most informative pairs were selected as the final design to adhere to the target experiment duration of approximately 5 minutes.

## Model Discrimination
The highest EIG pair in the design compares "HHHHHH" (an extreme contiguous streak) with "HTHTHTHT" (a perfect alternation). This highlights the core tension among the candidate theories:
- **`prototype_similarity`** models randomness based on proximity to an ideal alternation rate.
- **`encoding_compressibility`** penalizes simple descriptions like long runs and strict periodicity.
- **`inferred_mixture_model`** considers discrete generating hypotheses (streaky Markov vs alternating Markov).

The other selected pairs (e.g., "HTHTTTHH" vs "THTTTTTT") further probe the boundaries between sequence lengths and substring patterns. By prioritizing pairs with high variance in predicted choice probabilities across the five models, this design maximally distinguishes their differing computational accounts of subjective randomness.
