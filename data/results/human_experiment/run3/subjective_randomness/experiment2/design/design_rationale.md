# Experiment 2 Design Rationale

## Stimuli Selection

We generated a candidate pool of ~300 pairs of H/T sequences with lengths ranging from 2 to 8. We then scored these pairs based on Expected Information Gain (EIG) over the 7 cognitive models defined for this experiment (including `smoothed_prototype_distance`, `falk_konold_complexity`, `inner_loop_model`, and other baselines), assuming a uniform prior over the models. The top 32 stimuli with the highest EIG were selected for the experiment.

## EIG Range and Discrimination

- **Number of stimuli**: 32
- **EIG range**: 0.187 to 0.328 nats

The selected design is optimized to discriminate between the candidate models by identifying regions where the models make divergent predictions. For instance, `smoothed_prototype_distance` predicts length-dependent tolerance to imbalance through Bayesian smoothing, while `falk_konold_complexity` evaluates complexity based on alternating and repeating sub-sequence rates. By pitting sequences of varying lengths, balances, and streakiness against one another (e.g., comparing short highly-imbalanced sequences to longer sequences with specific structural motifs), the chosen stimulus set will provide the most informative possible test to distinguish these competing hypotheses.
