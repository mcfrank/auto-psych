# Candidate Brief

Refine the incumbent: `no_digit_containment_span_concepts` — rank 0, the best model on this data, ELPD-LOO -2008.4. Its hypothesis, as its author stated it:

> People do the same Bayesian concept learning as the current best model (arithmetic and final-digit rules plus multiples of 13-50, rule runs confined to a stretch of the number line, ranges of a typical width, an inferred share of random examples, examples drawn without replacement), except that they never entertain "contains the digit d" concepts: a digit that merely appears somewhere in the written numbers (such as a shared tens digit) is not noticed as a pattern. This simpler concept space predicts that near targets get no extra "yes" for sharing an example's decade and that sets like 8, 80, 48 do not generalise to other numbers containing an 8.

Its source is `/scr/wurgaft/auto-psych/agent_trees/962b9e6d929c436d/repo/_runs/cell_1/experiment3/model_loop/models/no_digit_containment_span_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
