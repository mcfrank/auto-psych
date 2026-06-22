# Theory Report — Experiment 2

## accumulated_alternation_typicality
**Hypothesis:** People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but this typicality is based entirely on how closely the sequence's alternation rate matches a mental prototype, completely ignoring the overall proportion of heads and tails.
**Motivation:** The best-fitting model from Experiment 1's inner loop (accumulated typicality) accumulates typicality based on deviations in both alternation rate and head proportion. This model isolates the alternation component to test whether head proportion is strictly necessary for the typicality accumulator, or if subjective randomness is predominantly driven by streakiness/switching alone.
**Mechanism:** The model calculates per-event typicality using the squared deviation of the empirical alternation rate from an inferred ideal alternation rate, with no term for head proportion. This event typicality is then multiplied by sequence length to produce the final randomness score.

## linear_accumulated_typicality
**Hypothesis:** People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but their penalty for deviating from the ideal head proportion and alternation rate is linear (absolute difference) rather than quadratic, treating extreme deviations proportionally to small ones.
**Motivation:** The winning inner-loop model from Experiment 1 (iter1_candidate0) assumes that the penalty for deviating from mental prototypes is quadratic, meaning people disproportionately heavily penalize sequences with extreme imbalances or run lengths. This refinement tests if people instead respond to deviations more linearly.
**Mechanism:** The model calculates per-event typicality as a baseline minus the weighted absolute difference between empirical rates and ideal rates (for both proportion and alternations), rather than the squared difference, and scales it by length. This is a direct variation of the functional form of the accumulated typicality hypothesis.
