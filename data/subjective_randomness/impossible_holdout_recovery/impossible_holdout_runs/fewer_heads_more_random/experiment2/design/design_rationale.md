# Experiment 2 Design Rationale

**Number of stimuli:** 30 pairs of sequences
**EIG range:** 0.3887 – 0.5663

## Rationale

We generated a candidate pool of 300 random pairs of coin flip sequences (up to length 8, accommodating different lengths as per the problem definition). From this pool, we used the Expected Information Gain (EIG) over the theorist's PyMC models to select the most informative 30 pairs. 

These top 30 sequence pairs maximize the predictive divergence among our current set of 6 models (`prototype_similarity`, `encoding_compressibility`, `bayesian_diagnosticity`, `inner_loop_model`, `fewer_heads_proportion`, and `short_streaks`). By querying participants on these highly discriminative stimuli, we will force the competing models to make mutually exclusive predictions, providing the strongest evidence to arbitrate between their varying mechanisms—such as sensitivity to pure proportion, streak lengths, prototype similarity, compressibility, or diagnosticity.
