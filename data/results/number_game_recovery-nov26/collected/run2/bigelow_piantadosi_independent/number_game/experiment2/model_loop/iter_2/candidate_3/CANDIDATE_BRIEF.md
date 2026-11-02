# Candidate Brief

Refine the incumbent: `capacity_limited_evidence_integration` — rank 0, the best model on this data, ELPD-LOO -2140.2. Its hypothesis, as its author stated it:

> People infer the hidden concept by Bayesian learning over mathematical rules (with separate prior plausibilities for even, odd and the other kinds of rule) and numerical intervals, but they have a limited capacity for integrating examples: the examples of a set compete for the same limited working memory, so each additional example adds less evidence than the one before, and the total evidence grows more slowly than the number of examples. This predicts that larger sets are generalized more broadly (broad rules and wide ranges keep more weight) than an ideal learner, which counts every example fully, would allow, while single-example sets are judged as the ideal learner does.

Its source is `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/capacity_limited_evidence_integration.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
