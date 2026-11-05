# Candidate Brief

Refine the incumbent: `rule_ambiguity_single_doubt` — rank 0, the best model on this data, ELPD-LOO -4738.2. Its hypothesis, as its author stated it:

> Refinement of the incumbent `single_example_random_program_doubt` by one change to its doubt mechanism: how often people set a lone example aside and answer at a flat rate depends on how many candidate rules that example fits. A number that belongs to many salient categories at once (64: even, a square, a cube, a power of 2, 4 and 8, a multiple of 4 and 8, ...) gives no way to tell which of them the program follows, so people more often give up on rule-based judgment and answer flatly; a number with few salient properties (43) points to a definite rule and is judged by its profile more often. Everything else is the incumbent unchanged (the salience-weighted share of the examples' shared profile, with the interval similarity on multi-example sets), so the breadth of generalization differs from one lone example to another, which the incumbent's single global doubt rate cannot produce (the n1_set_mean_variance critique).

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/rule_ambiguity_single_doubt.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
