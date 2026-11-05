# Candidate Brief

Refine the incumbent: `rule_within_range_conjunction` — rank 0, the best model on this data, ELPD-LOO -1626.1. Its hypothesis, as its author stated it:

> People infer a conjunctive concept: a mathematical rule (e.g. "even", "multiples of 7", or "any number") restricted to a range of magnitudes, and they infer both the rule and the range from the examples by Bayesian size-principle reasoning. Because the range can be tight around the examples, a target that fits the rule but lies far outside the span of the examples is judged less likely to be produced than one inside it.

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/rule_within_range_conjunction.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
