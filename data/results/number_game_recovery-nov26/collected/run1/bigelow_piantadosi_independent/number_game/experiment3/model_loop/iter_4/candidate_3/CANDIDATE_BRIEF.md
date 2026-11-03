# Candidate Brief

Refine the incumbent: `salient_contained_digit_rules` — rank 0, the best model on this data, ELPD-LOO -2598.1. Its hypothesis, as its author stated it:

> Refinement of `gamma_shaped_range_width_prior`: people do the same tempered, exception-tolerant Bayesian concept learning (salient final-digit rules, ranges on a compressed number line with a gamma-shaped width prior, and rule-plus-neighbourhood exceptions anchored on an off-pattern example and emitted through a side channel), but rules about a digit the numeral contains ("has a 4 in it") are also perceptually salient. They are read straight off the written numerals, so they get their own fitted prior boost instead of starting no more likely than an arbitrary arithmetic rule. The one change is this contained-digit salience: when every example contains a common digit in different places, people extend that digit to distant numbers that contain it (especially numbers in that digit's decade, such as 47 after 34, 43, 74), more than the incumbent predicts (the critique's shared-contained-digit tens statistic).

Its source is `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment3/model_loop/models/salient_contained_digit_rules.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
