# Candidate Brief

Refine the incumbent: `heading_weber_with_upward_counting` — rank 0, the best model on this data, ELPD-LOO -3197.9. Its hypothesis, as its author stated it:

> Refinement of the incumbent `heading_extrapolation_weber_width`. People do Bayesian concept learning over the same arithmetic, digit and digit-sum rules and every numeric range of 1-100. They read the examples as distinct draws with some strays, sometimes guess, judge a range's breadth Weber-like (relative to its magnitude), and let the direction of the last step stretch the ranges toward where the sequence is heading. The one change, the component taken from `upward_room_range_prior`, is a counting habit that does not depend on order. Whatever order the examples came in, people expect the program's numbers to carry on upward past the largest example, so they credit a candidate range by how far it reaches above the examples (strength inferred). This predicts more "yes" to numbers a little above a cluster than below it even for single examples and for sets whose last step points down (e.g. 67 versus 57 after 64, 62, 60), on top of the order-dependent heading effect.

Its source is `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment3/model_loop/models/heading_weber_with_upward_counting.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
