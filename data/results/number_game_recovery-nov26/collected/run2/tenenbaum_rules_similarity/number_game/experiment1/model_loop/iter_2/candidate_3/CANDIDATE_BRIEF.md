# Candidate Brief

Refine the incumbent: `inferred_example_noise_concepts` — rank 0, the best model on this data, ELPD-LOO -1212.1. Its hypothesis, as its author stated it:

> A refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but they do not assume a fixed 10% of examples are random noise; how often they believe the program emits an example from outside its concept is a free quantity inferred from the data. The one change is this inferred example-noise rate, which governs both how sharply the size principle accumulates over examples and how readily a rule is kept despite an example it does not cover.

Its source is `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment1/model_loop/models/inferred_example_noise_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
