# Design Rationale

## Overview
This design selects 32 pairs of binary sequences (H/T) to maximize Expected Information Gain (EIG) for discriminating between cognitive models of subjective randomness. 

## Candidate Pool
We generated an initial tractable pool of 300 candidate pairs by randomly sampling sequences of lengths 2 through 8. To ensure diversity, half of the candidate pairs consist of sequences of the same length, and the other half consist of sequences of different lengths. This captures both within-length and cross-length evaluations, providing a rich basis for model discrimination.

## Selection and EIG
The pool of 300 candidates was scored by EIG across the prior predictive distributions of all theoretical models present in the `cognitive_models` directory (using the current model registry constraints). The 32 candidate pairs with the highest joint discriminatory value were selected for the final experiment.

**Key Design Properties:**
- **Number of Stimuli**: 32 pairs.
- **EIG Range**: 0.148 to 0.245 bits per pair.
- **Discrimination Strategy**: By spanning sequences of varying lengths (2–8), varying alternation rates, and varying lengths of "streaks" (like `"TTTTTHHH"` vs `"HTHHHHHH"`), this set provides maximal distinguishing power. The diverse mix of within-length and cross-length pairs isolates features where the theoretical models make divergent predictions regarding sequence typicality, alternation biases, and structural complexity.
