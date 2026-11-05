# Candidate Brief

Refine the incumbent: `whole_set_round_category` — rank 0, the best model on this data, ELPD-LOO -4181.6. Its hypothesis, as its author stated it:

> People judge a target by its similarity to the stored examples (closeness in magnitude, final digit, parity, the set's common divisor, sharpened by suspicious coincidences, plus distance-free routes for multiples and factor-kin, with support partly averaged over examples), but the familiar category "round numbers" is only brought to mind when the whole example set fits it: a multiple-of-ten target gains distance-free similarity to the examples only when every example ends in 0 or 5 (e.g. 30 or 100 after "5", 50 after "45, 90"). A lone 5- or 0-ending example inside a mixed set (e.g. "15, 11" or "6, 25") gives round targets no boost, unlike in the current best model, where each round-ish example lends kinship to round targets on its own.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment3/model_loop/models/whole_set_round_category.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
