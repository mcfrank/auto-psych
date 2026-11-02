# Candidate Brief

Refine the incumbent: `margin_beyond_examples_range_prior` — rank 0, the best model on this data, ELPD-LOO -1926.9. Its hypothesis, as its author stated it:

> Refinement of `digit_sum_open_ended_range_discount` (the incumbent): people do Bayesian concept learning over the same arithmetic and digit rules, the 18 digit-sum concepts and every numeric range of 1-100, reading the examples as distinct draws with some strays, sometimes guessing, favouring narrow ranges and discounting open-ended ranges that run to 1 or 100. The one change is a margin expectation: people do not expect the few numbers they were shown to sit right at the edges of the program's range, so they credit a candidate range by how far it reaches past the examples on either side (strength inferred; the data suggest more credit for ranges with room beyond the examples). This predicts more "yes" to numbers a little beyond either end of a cluster (e.g. 57 or 67 after 60, 62, 64) than the incumbent gives, symmetrically on both sides and regardless of the order the examples were shown in, while rule and digit-sum generalization are unchanged.

Its source is `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment2/model_loop/models/margin_beyond_examples_range_prior.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
