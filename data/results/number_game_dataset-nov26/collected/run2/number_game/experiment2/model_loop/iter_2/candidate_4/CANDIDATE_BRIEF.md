# Candidate Brief

Refine the incumbent: `gap_step_progression_overlap` — rank 0, the best model on this data, ELPD-LOO -3177.9. Its hypothesis, as its author stated it:

> People's space of candidate concepts includes an arithmetic-progression concept that the current models lack. When the examples are evenly spaced (e.g. 8 and 16, or 2 and 8), people entertain "start at the smallest example and keep adding the gap between the examples" (8, 16, 24, 32, ...; 2, 8, 14, 20, ...) as one more property of the set, with its own inferred prior salience. Otherwise they judge the target as before, by the salience-weighted share of the examples' shared property profile it has, so targets on the upward continuation of the progression are endorsed more even when they break other shared properties such as being a power of 2.

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment2/model_loop/models/gap_step_progression_overlap.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
