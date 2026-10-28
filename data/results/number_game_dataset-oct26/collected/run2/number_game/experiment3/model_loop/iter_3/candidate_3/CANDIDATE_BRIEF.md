# Candidate Brief

Refine the incumbent: `upward_count_start_boundary` — rank 0, the best model on this data, ELPD-LOO -4687.8. Its hypothesis, as its author stated it:

> Refinement of saturating_criterion_relaxation (all else unchanged: recency-weighted similarity on size, final digit, parity and graded shared factor with coherence-stretched attention, the "multiples of g", "ends in d" and fixed-list concepts, the filled-in span, sequence momentum, zigzag breadth and the saturating relaxation of the criterion). The single change is that people think of the program as counting upward from a starting number: when the examples go up overall (the last one shown is larger than the first), they take the first example as where the program started and treat any number below it as one the program never reached, so they reject it far more than its similarity alone predicts. Sets that go down overall have no such starting point. So the same target, just below the smaller example, is rejected after 8, 16 but not after 16, 8.

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment3/model_loop/models/upward_count_start_boundary.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
