# Candidate Brief

Refine the incumbent: `range_bounded_parity_concepts` — rank 0, the best model on this data, ELPD-LOO -3184.0. Its hypothesis, as its author stated it:

> Refinement of `parity_restricted_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, and their odd-only and even-only versions. The one change is a single added component, taken from `bounded_range_rule_learning`: people can also restrict a rule to a range of numbers that covers the examples (e.g. "multiples of 3 between 20 and 100"), and these range-bounded rules get an inferred share of the prior for rules (share 0 recovers `parity_restricted_concepts`). A bounded rule is smaller than the full rule, so the size principle favours it. This predicts that people endorse members of the examples' rule that lie outside the examples' min-max range less than members inside it, and that this gap is larger than the rule/interval mixture allows (63,27,99,33 → 3 below 45).

Its source is `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/range_bounded_parity_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
