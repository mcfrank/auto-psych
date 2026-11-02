# Candidate Brief

Refine the incumbent: `digit_sum_open_ended_range_discount` — rank 0, the best model on this data, ELPD-LOO -1928.2. Its hypothesis, as its author stated it:

> Refinement of `digit_sum_concepts` (the incumbent): people do Bayesian concept learning over the same arithmetic and digit rules, the 18 "digits add up to s" concepts and every numeric range of 1-100, reading the examples as distinct draws with some strays, sometimes guessing, and favouring narrow ranges. The one change, the component taken from `open_ended_range_discount`, is that a range running to an end of the number line ("anything up to 30", "anything from 70 on") is a threshold rule rather than a bounded stretch, and people hold such open-ended ranges at their own prior weight relative to bounded ones (inferred; the data suggest lower). This predicts fewer "yes" answers to numbers far below small examples or far above large ones, all the way to 1 or 100, while digit-sum and rule generalization and "yes" to numbers just beyond the examples' span are kept.

Its source is `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment2/model_loop/models/digit_sum_open_ended_range_discount.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
