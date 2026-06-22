# Experiment 3 Design Rationale

- **Number of stimuli**: 32 pairs of coin flip sequences.
- **EIG range**: 0.0834 to 0.1887.

## Rationale
The design process evaluated a randomly generated set of candidate sequence pairs to find the subset that maximizes the Expected Information Gain (EIG) across the current set of models. By selecting the top 32 high-EIG pairs, the experiment is targeted precisely to differentiate the competing models of subjective randomness perception. 

The selected pairs provide high EIG by creating scenarios where models diverge strongly in their predictions of which sequence will be judged "more random." Some sequences might feature different string lengths, varying proportion of heads vs tails, alternation rates, and max run lengths, isolating specific features used by the different models (such as accumulated typicality models vs asymmetric alternation typicality).