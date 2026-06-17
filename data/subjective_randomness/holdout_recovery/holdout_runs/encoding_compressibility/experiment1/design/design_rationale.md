# Experiment 1 Design Rationale

**Number of stimuli:** 30 pairs  
**EIG Range:** 0.1212 – 0.2373  

## Design Strategy
To maximally discriminate between the `bayesian_diagnosticity` and `prototype_similarity` models, the expected information gain (EIG) pipeline evaluated candidate pairs of coin flip sequences (lengths 3 to 8). 

The optimal design consists of sequence pairs that create strong diverging predictions across the two models' features. Specifically:
- `bayesian_diagnosticity` evaluates sequences using Bayesian evidence across fair, alternating, streaky, and biased generators. It responds sensitively to raw counts (`n`, `h`, `alts`) and heavily penalizes overly regular patterns through alternative hypotheses.
- `prototype_similarity` scores randomness strictly based on distances from a balanced (`imbalance`) and alternating (`p_alts`) prototype. 

The selected stimuli frequently contrast highly alternating sequences (e.g., `HTHTHTTH`) with extremely streaky or biased sequences (e.g., `TTTTTTTT`). Such pairs force the models apart: prototype similarity treats the alternating sequence as closer to the "alternating prototype," while bayesian diagnosticity assigns different relative evidences based on the prior probability of streakiness versus alternating generative processes. By sampling pairs that maximize this tension, the 30 selected trials are expected to sharply distinguish which model better matches human subjective randomness judgments.
