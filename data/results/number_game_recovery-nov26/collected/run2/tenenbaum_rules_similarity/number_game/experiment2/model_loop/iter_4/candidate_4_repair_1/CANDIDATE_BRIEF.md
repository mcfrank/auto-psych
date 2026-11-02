# Candidate Brief

Refine the incumbent: `span_restricted_rule_concepts` — rank 0, the best model on this data, ELPD-LOO -1698.5. Its hypothesis, as its author stated it:

> People do the same Bayesian concept learning as the current best model (45 independently weighted rules plus multiples of 13-50, ranges of 1-100 with a preferred typical width, an inferred share of random examples, examples drawn without replacement), but they also entertain rules confined to a stretch of the number line — "even numbers from 42 to 92", "numbers ending in 4 from 64 up" — i.e. any unbroken run of consecutive members of a familiar rule, every such run equally likely a priori and the family taking an inferred share of the rule prior. Because such a run is smaller than the whole rule, the size principle favours it when rule-following examples are clustered in magnitude, predicting "yes" to rule members inside or near the examples' span and fewer "yes" answers to members of the same rule far from them (e.g. 52 but not 12 after 72, 92, 82, 42).

Its source is `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment2/model_loop/models/span_restricted_rule_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
