# Candidate Brief

Refine the incumbent: `exception_tolerant_tempered_mixture` — rank 0, the best model on this data, ELPD-LOO -808.6. Its hypothesis, as its author stated it:

> Refinement of `tempered_size_principle_mixture`: people do the same tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they believe the program often slips and produces a number outside its concept, so they tolerate exceptions — the fraction of examples assumed to be off-concept noise is a fitted parameter instead of a fixed 10%. The one change is this fitted exception rate: when most but not all examples share a regularity (e.g. three of four end in 2), the rule that covers the majority survives the exception, so people extend a partly shared last digit or arithmetic property to distant targets, and lean less on tight numerical intervals, than a learner who treats every example as strictly from the concept.

Its source is `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment1/model_loop/models/exception_tolerant_tempered_mixture.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
