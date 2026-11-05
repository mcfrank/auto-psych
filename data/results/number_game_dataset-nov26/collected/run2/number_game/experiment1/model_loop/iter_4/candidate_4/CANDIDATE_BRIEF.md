# Candidate Brief

Refine the incumbent: `one_sided_magnitude_bound_overlap` — rank 0, the best model on this data, ELPD-LOO -1497.0. Its hypothesis, as its author stated it:

> Refinement of `decimal_multiple_salience_overlap`: people still note every salient property all the examples share (each property family with its own salience, base-ten multiples their own family, no size principle) and say "yes" in proportion to the salience-weighted share of that profile the target also has. The one change is to the magnitude concepts people consider: instead of one two-sided "within the examples' span" property, they entertain two separate one-sided bounds — "at least as large as the smallest example" and "no larger than the largest example" — each with its own salience, so if the open-ended "this big or bigger" bound is the more salient, targets above all the examples are endorsed more readily than equally distant targets below them.

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment1/model_loop/models/one_sided_magnitude_bound_overlap.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
