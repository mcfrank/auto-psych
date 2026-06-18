# Experiment 1 Design Rationale

**Number of stimuli**: 32 pairs
**EIG range**: 0.0851 – 0.1809

## Design Generation Process
1. **Candidate Pool**: We uniformly sampled 300 sequence pairs to form the candidate pool. The individual sequences in each pair range in length from 4 to 8, generated with equal chance across this length distribution.
2. **Scoring**: Each pair was processed through a feature extraction pipeline and then scored by its Expected Information Gain (EIG) over the theorist's PyMC models, which used the prior-predictive `p_left` to evaluate their discriminative power.
3. **Selection**: The top 32 sequence pairs with the highest EIG were selected as the final experimental stimuli.

## Model Discrimination
By choosing sequence pairs that maximize Expected Information Gain (EIG), the selected design directly focuses on trials where the prior predictions of the various cognitive models in `cognitive_models/` differ most. Some sequences have highly imbalanced head/tail counts, varying maximum run lengths, and different transition patterns (alternation rates). Providing sequences that pull the model predictions apart ensures that participants' responses in a forced-choice task will strongly provide evidence for or against each candidate cognitive model.