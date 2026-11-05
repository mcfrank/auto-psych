# Candidate Brief

Refine the incumbent: `factor_kin_multiple_route` — rank 0, the best model on this data, ELPD-LOO -2717.2. Its hypothesis, as its author stated it:

> Refinement of `multiple_route_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity, divisibility by the set's common divisor, all sharpened when the examples share a suspicious coincidence). Being a multiple of an example is still a distance-free route to similarity. The one change is that this arithmetic route is broader than exact multiples. A target that shares a factor of 3 or more with an example also gains a separate, smaller fixed amount of similarity to it, at any distance, even when it is neither a multiple nor a divisor of that example (12 or 27 after "15", 9 after "6", 28 after "42"). So people generalize to the factor-kin of an example, not only to its multiples, while targets sharing no such factor with any example get no boost.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/factor_kin_multiple_route.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
