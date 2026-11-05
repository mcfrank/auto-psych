# Candidate Brief

Refine the incumbent: `tempered_size_principle_independent` — rank 0, the best model on this data, ELPD-LOO -1599.7. Its hypothesis, as its author stated it:

> Refinement of `bigelow_piantadosi_independent`: people do the same Bayesian concept learning over 45 rules (each with its own prior weight) and all intervals, but they under-trust the size principle — they treat the examples as weaker evidence than a fully rational learner would, as if the likelihood of the examples were tempered by an inferred exponent (sampling-assumption strength), so their posterior is less concentrated on the smallest consistent rule. The one change is this free evidence-strength exponent (1 recovers the original model); it predicts lower endorsement of minimal-rule members and graded endorsement among them, because members also covered by competing concepts (nearby intervals, other rules) keep more support.

Its source is `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment1/model_loop/models/tempered_size_principle_independent.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
