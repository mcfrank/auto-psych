# Candidate Brief

Refine the incumbent: `shared_property_rule_matching` — rank 0, the best model on this data, ELPD-LOO -1674.5. Its hypothesis, as its author stated it:

> People ignore magnitude and similarity entirely and judge by rule matching: they note which arithmetic properties all the examples have in common (even, odd, multiple of 3/4/5/10, a shared final digit, prime, square, power of 2) and say "yes" to the degree that the target has those same properties, regardless of how far it lies from the examples. So rule-consistent targets are accepted even when numerically distant, rule violators are rejected even when close, and a single example (which carries all its own properties) generalizes as readily as several.

Its source is `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/shared_property_rule_matching.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
