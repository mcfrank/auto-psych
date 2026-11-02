# Candidate Brief

Refine the incumbent: `recency_weighted_even_odd_family_prior` — rank 0, the best model on this data, ELPD-LOO -871.3. Its hypothesis, as its author stated it:

> Refinement of `even_odd_split_family_prior`: people do the same Bayesian concept learning over the same 45 rules (with separate prior plausibilities for even, odd and the other kinds of rule) and all intervals of 1-100 with a tempered size principle, but they do not weigh every example equally — they read the examples in order and each example's evidence counts more the later it was shown (a fitted order gradient, with the total amount of evidence held fixed). The single change is grafting this recency weighting of the examples (from `recency_weighted_concept_learning`) onto the incumbent, motivated by the critique that targets near the last-shown example get more yeses than targets near the first-shown one, which an order-blind learner cannot produce.

Its source is `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/recency_weighted_even_odd_family_prior.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
