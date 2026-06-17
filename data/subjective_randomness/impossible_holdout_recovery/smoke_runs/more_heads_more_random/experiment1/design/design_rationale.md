# Experiment 1 Design Rationale

**Number of stimuli:** 30 pairs
**EIG range:** 0.1108 – 0.2766

## Model Discrimination Strategy
This design selects 30 sequence pairs (drawn from a random candidate pool of 250 sequence pairs with lengths from 2 to 8) to maximally discriminate between three cognitive models of subjective randomness:
1. **Prototype Similarity:** Evaluates closeness to balanced H/T counts and an ideal alternation rate.
2. **Encoding Compressibility:** Penalizes simple features like long runs, periodicity, and imbalance.
3. **Bayesian Diagnosticity:** Evaluates evidence of a fair coin against non-random alternatives like biased, streaky, or alternating generators.

The selected sequences (such as `HT` vs `TTTTTTHH`) offer contrasting features across the models—varying drastically in terms of longest runs, length, alternation proportion, and balance. These differing feature profiles cause the models' prior-predictive predictions to diverge, yielding high Expected Information Gain (EIG) and thus providing strong discriminative power to identify which model best explains human judgments of subjective randomness.
