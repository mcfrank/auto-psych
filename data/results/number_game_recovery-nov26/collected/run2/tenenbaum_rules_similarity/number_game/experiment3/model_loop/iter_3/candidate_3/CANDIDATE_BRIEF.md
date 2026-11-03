# Candidate Brief

Refine the incumbent: `graded_ratio_ceiling_criterion` — rank 0, the best model on this data, ELPD-LOO -2006.9. Its hypothesis, as its author stated it:

> I chose above_max_stricter_criterion and refined it. People do the same Bayesian concept learning as the best model (arithmetic and final-digit rules plus multiples of 13-50, rule runs confined to a stretch of the number line, ranges of a typical width, an inferred share of random examples, examples drawn without replacement, no "contains digit d" concepts), and they treat the largest example as a soft ceiling when they turn belief into a yes/no answer. The one change is that the extra certainty they demand is not a fixed step but grows with how far beyond the largest example the target lies, judged as a ratio (Weber-like "how much bigger"), so a target just above the largest example is barely penalised, one twice as large needs much stronger evidence, and the same absolute overshoot counts for more above small examples than above large ones.

Its source is `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment3/model_loop/models/graded_ratio_ceiling_criterion.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
